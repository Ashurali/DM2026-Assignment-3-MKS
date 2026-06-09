"""Stage 2a: binary L1 vs L2 classifier — the dedicated learner for the
confused-pair boundary identified by diagnose_bottleneck.py (167 of 223 L2
misses go to L1; 119 L1→L2 false positives).

Trains only on rows where y ∈ {L1, L2} (5053 rows, 13:1 imbalance).
Per-fold model: trained on tr ∩ {y ∈ {1,2}}, predicts on FULL val (so OOF is
defined for all train rows — values for y∉{1,2} are meaningless on their own
but are used in composition where they'll be multiplied by P(C_walking|x)≈0).
Test: average of 5 fold-models trained on each tr ∩ {y ∈ {1,2}}.

Outputs:
  oof/hier_fine_walk_oof.npy        (N_train, 2) — [P(L1), P(L2)] | within walking
  oof/hier_fine_walk_test_probs.npy (N_test, 2)

Usage:
  python scripts/train_hier_fine_walk.py [--gpu]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("=== train_hier_fine_walk.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hier_common import build_feature_blocks, make_class_weights, ROOT
from src.utils.cv import make_folds


def main():
    import lightgbm as lgb
    from sklearn.metrics import f1_score, classification_report

    gpu = "--gpu" in sys.argv

    X, Xte, block_names = build_feature_blocks(include_combo_oof=True)
    print(f"Features: {block_names}", flush=True)
    print(f"X {X.shape}  Xte {Xte.shape}", flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    y6 = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)

    walk_mask = (y6 == 1) | (y6 == 2)
    print(f"Walking subset: {walk_mask.sum()} rows  ({(y6==1).sum()} L1 / {(y6==2).sum()} L2)", flush=True)

    # Map labels: L1 → 0, L2 → 1
    y_bin = np.zeros_like(y6)
    y_bin[y6 == 2] = 1

    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=15,
        verbose=-1, seed=42, num_threads=16,
    )
    if gpu:
        params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)
    NBR = 400

    # Per-fold OOF + test prediction (averaged)
    oof_p_l2 = np.zeros(len(y6), dtype=np.float64)  # P(L2 | x, walking)
    test_p_l2_sum = np.zeros(len(Xte), dtype=np.float64)
    fold_f1s = []
    for k, (tr, va) in enumerate(folds):
        # Restrict TRAINING to walking rows; prediction stays on full val
        tr_walk = tr[walk_mask[tr]]
        print(f"  Fold {k}: tr_walk={len(tr_walk)} (full tr={len(tr)})  va={len(va)}", flush=True)
        # Aggressive class weight on L2: scale to ~1:1 within binary
        n1 = int((y_bin[tr_walk] == 0).sum())
        n2 = int((y_bin[tr_walk] == 1).sum())
        w_l1 = 1.0
        w_l2 = max(1.0, n1 / max(n2, 1))
        weights = np.where(y_bin[tr_walk] == 0, w_l1, w_l2).astype(np.float64)
        ds = lgb.Dataset(X[tr_walk], label=y_bin[tr_walk], weight=weights)
        m = lgb.train(params, ds, num_boost_round=NBR)

        # Predict on full validation slice (incl. y∉{1,2})
        oof_p_l2[va] = m.predict(X[va])
        # Test: average across folds
        test_p_l2_sum += m.predict(Xte)

        # Fold F1 on the walking subset of val
        va_walk = va[walk_mask[va]]
        if len(va_walk) > 0:
            preds_va = (oof_p_l2[va_walk] >= 0.5).astype(int)
            f = float(f1_score(y_bin[va_walk], preds_va, average="macro"))
            fold_f1s.append(f)
            print(f"    fold L1-vs-L2 F1 on walking val: {f:.4f}  "
                  f"(L1 recall={(preds_va[y_bin[va_walk]==0]==0).mean():.3f}, "
                  f"L2 recall={(preds_va[y_bin[va_walk]==1]==1).mean():.3f})", flush=True)

    test_p_l2 = test_p_l2_sum / len(folds)

    # ── OOF report on walking subset ──
    walk_idx = np.where(walk_mask)[0]
    oof_preds_walk = (oof_p_l2[walk_idx] >= 0.5).astype(int)
    overall_f1 = float(f1_score(y_bin[walk_idx], oof_preds_walk, average="macro"))
    print(f"\nOOF L1-vs-L2 binary F1-macro on walking subset: {overall_f1:.4f}", flush=True)
    print(f"  fold mean: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)
    print(classification_report(y_bin[walk_idx], oof_preds_walk, target_names=["L1", "L2"], digits=4))

    # Save 2-column probs: [P(L1), P(L2)] within walking
    oof_2col = np.stack([1 - oof_p_l2, oof_p_l2], axis=1).astype(np.float32)
    test_2col = np.stack([1 - test_p_l2, test_p_l2], axis=1).astype(np.float32)
    np.save(ROOT / "oof" / "hier_fine_walk_oof.npy", oof_2col)
    np.save(ROOT / "oof" / "hier_fine_walk_test_probs.npy", test_2col)
    print(f"\nSaved hier_fine_walk_oof.npy {oof_2col.shape}", flush=True)
    print(f"Saved hier_fine_walk_test_probs.npy {test_2col.shape}", flush=True)


if __name__ == "__main__":
    main()
