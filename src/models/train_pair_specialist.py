"""Binary specialist classifiers for confusable class pairs.

Deep EDA showed that the model's largest confusion is L2 → L1 (230/358
true-L2 files predicted as L1, centroid distance 0.225). Second is L5 → L1
(centroid distance 0.110). A specialist trained ONLY on the relevant
two-class subset learns sharper boundary decisions there, and its
probability output stacks as a feature into the combo model.

This script trains a binary LightGBM on a {class_a, class_b} subset of
the train data with 5-fold GroupKFold, and produces:
- OOF probabilities P(class_b) on all 11k files (predicted at inference
  time from the per-fold model on samples not in that fold)
- Test probabilities P(class_b) for all 6849 test files

Both saved as 1-D arrays (P(b), so 1 - P(b) = P(a)) ready to inject as
features into a stacked LGBM.

Usage:
    python -m src.models.train_pair_specialist --pair 1 2 --name l1_v_l2
    python -m src.models.train_pair_specialist --pair 1 5 --name l1_v_l5

Outputs:
    oof/pair_<name>_oof.npy        (N_train,) — P(class_b) for all train (per-fold OOF + zero for non-pair files)
    oof/pair_<name>_test.npy       (N_test,)  — P(class_b) for all test
    oof/pair_<name>_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.utils.cv import make_folds

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=int, nargs=2, required=True, metavar=("CLASS_A", "CLASS_B"),
                   help="Two class indices (e.g. 1 2 for L1-vs-L2).")
    p.add_argument("--name", required=True, help="Suffix for output files.")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--num-boost-round", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    a, b = args.pair
    print(f"Training binary specialist: L{a} (class 0) vs L{b} (class 1)")

    # Load 271-feature cache
    cache_train = ROOT / "data" / "feat_train_none.parquet"
    cache_test = ROOT / "data" / "feat_test_none.parquet"
    if not (cache_train.exists() and cache_test.exists()):
        raise SystemExit("Missing 271-feature cache. Run train_lgbm_full.py with --cache-features first.")

    Xtr_df = pd.read_parquet(cache_train)
    Xte_df = pd.read_parquet(cache_test)
    fcols = [c for c in Xtr_df.columns if c != "file_id"]
    X = Xtr_df[fcols].values.astype(np.float64)
    Xte = Xte_df[fcols].values.astype(np.float64)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    y_full = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values

    pair_mask = (y_full == a) | (y_full == b)
    print(f"Pair samples: a={int((y_full == a).sum())}  b={int((y_full == b).sum())}  total={int(pair_mask.sum())}")

    # Binary y: 0 = class_a, 1 = class_b
    y_pair = (y_full == b).astype(np.int64)

    # Per-fold training: use ONLY pair samples in train fold; predict on ALL files in val fold
    # This gives OOF P(b) for every file in train (on the val fold of the GKF).
    folds = make_folds(groups, n_splits=5)
    oof_probs = np.zeros(len(y_full), dtype=np.float32)

    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
        is_unbalance=True,  # binary balance
        verbose=-1, seed=SEED, num_threads=16,
    )
    if args.gpu:
        params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)

    for k, (tr_idx, va_idx) in enumerate(folds):
        # Train fold: only pair samples within tr_idx
        tr_pair_idx = tr_idx[pair_mask[tr_idx]]
        ds = lgb.Dataset(X[tr_pair_idx], label=y_pair[tr_pair_idx])
        model = lgb.train(params, ds, num_boost_round=args.num_boost_round)

        # Predict P(b) for ALL files in val fold (pair and non-pair)
        # For non-pair files in val, the prediction is meaningless individually
        # but useful as a "the model thinks this signature is L_b-like" feature.
        val_pred = model.predict(X[va_idx])
        oof_probs[va_idx] = val_pred
        # Per-fold check: AUC on pair-only val samples
        from sklearn.metrics import roc_auc_score
        va_pair_idx = va_idx[pair_mask[va_idx]]
        if len(va_pair_idx) > 0 and len(np.unique(y_pair[va_pair_idx])) == 2:
            auc = roc_auc_score(y_pair[va_pair_idx], val_pred[pair_mask[va_idx]])
            print(f"  Fold {k}: pair-AUC = {auc:.4f}  (n_pair_val = {len(va_pair_idx)})")

    # Final model on all pair train data, predict test
    print("Training final model on all pair samples...")
    pair_idx = np.where(pair_mask)[0]
    ds_full = lgb.Dataset(X[pair_idx], label=y_pair[pair_idx])
    final_model = lgb.train(params, ds_full, num_boost_round=args.num_boost_round)
    test_probs = final_model.predict(Xte).astype(np.float32)

    np.save(ROOT / "oof" / f"pair_{args.name}_oof.npy", oof_probs)
    np.save(ROOT / "oof" / f"pair_{args.name}_test.npy", test_probs)

    # Sidecar
    sidecar = {
        "name": args.name,
        "class_a": int(a),
        "class_b": int(b),
        "n_a": int((y_full == a).sum()),
        "n_b": int((y_full == b).sum()),
        "n_total_pair": int(pair_mask.sum()),
        "params": params,
        "num_boost_round": args.num_boost_round,
        "oof_pair_mask_p_mean": float(oof_probs[pair_mask].mean()),
        "test_pred_distribution": {
            "p_b_mean": float(test_probs.mean()),
            "p_b_above_0.5": int((test_probs > 0.5).sum()),
        },
    }
    with open(ROOT / "oof" / f"pair_{args.name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, default=str)

    print(f"\nSaved oof: oof/pair_{args.name}_oof.npy  shape={oof_probs.shape}")
    print(f"Saved test: oof/pair_{args.name}_test.npy  shape={test_probs.shape}")
    print(f"Saved meta: oof/pair_{args.name}_meta.json")


if __name__ == "__main__":
    main()
