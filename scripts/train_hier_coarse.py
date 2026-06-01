"""Stage 1: coarse 3-way classifier on super-classes.

C_static  = {L0}        (4643 train rows, 42%)
C_walking = {L1, L2}    (5053 train rows, 46%)
C_other   = {L3, L4, L5} (1324 train rows, 12%)

Inputs:  811 features (805 base + 6-way combo OOF/test as a feature block).
Output:  P(super | x) for train (5-fold OOF) and test.

Saves:
  oof/hier_coarse_oof.npy            (N_train, 3)
  oof/hier_coarse_test_probs.npy     (N_test, 3)

Usage:
  python scripts/train_hier_coarse.py [--gpu]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("=== train_hier_coarse.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hier_common import build_feature_blocks, make_class_weights, label_to_super, ROOT
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
    y3 = label_to_super(y6)
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)
    print(f"Super-class counts: {np.bincount(y3)}  (C_static, C_walking, C_other)", flush=True)

    params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
        verbose=-1, seed=42, num_threads=16,
    )
    if gpu:
        params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)
    NBR = 500

    oof = np.zeros((len(y3), 3), dtype=np.float64)
    fold_f1s = []
    for k, (tr, va) in enumerate(folds):
        print(f"  Fold {k}: tr={len(tr)} va={len(va)}", flush=True)
        w_tr = make_class_weights(y3[tr], n_classes=3)
        ds = lgb.Dataset(X[tr], label=y3[tr], weight=w_tr)
        m = lgb.train(params, ds, num_boost_round=NBR)
        probs = m.predict(X[va])
        oof[va] = probs
        f = float(f1_score(y3[va], probs.argmax(axis=1), average="macro"))
        fold_f1s.append(f)
        print(f"    fold F1: {f:.4f}", flush=True)

    oof_pred = oof.argmax(axis=1)
    coarse_f1 = float(f1_score(y3, oof_pred, average="macro"))
    print(f"\nCoarse 3-way OOF F1-macro: {coarse_f1:.4f}  fold mean: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}",
          flush=True)
    print(classification_report(y3, oof_pred, target_names=["C_static", "C_walking", "C_other"], digits=4))

    # ── Audit super-class recall on the constituent fine classes ──
    print("\nSuper-class recall by fine class (OOF):")
    for c in range(6):
        mask = (y6 == c)
        if mask.sum() == 0:
            continue
        super_c = label_to_super(np.array([c]))[0]
        recovered = ((y3[mask] == oof_pred[mask]) & (oof_pred[mask] == super_c)).sum()
        print(f"  L{c} (n={int(mask.sum())}): {recovered}/{int(mask.sum())} = "
              f"{recovered / mask.sum() * 100:.1f}% routed to correct super-class")

    # Final fit, predict test
    print("\nFinal model fit on all train, predicting test...", flush=True)
    w_full = make_class_weights(y3, n_classes=3)
    full_ds = lgb.Dataset(X, label=y3, weight=w_full)
    final_model = lgb.train(params, full_ds, num_boost_round=NBR)
    test_probs = final_model.predict(Xte)

    np.save(ROOT / "oof" / "hier_coarse_oof.npy", oof.astype(np.float32))
    np.save(ROOT / "oof" / "hier_coarse_test_probs.npy", test_probs.astype(np.float32))
    print(f"\nSaved hier_coarse_oof.npy {oof.shape}", flush=True)
    print(f"Saved hier_coarse_test_probs.npy {test_probs.shape}", flush=True)


if __name__ == "__main__":
    main()
