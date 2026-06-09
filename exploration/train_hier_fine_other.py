"""Stage 2b: ternary L3 vs L4 vs L5 classifier within C_other.

Trains only on rows where y ∈ {L3, L4, L5} (1324 rows: 656 L3 / 142 L4 / 526 L5).
Per-fold model: trained on tr ∩ {y ∈ {3,4,5}}, predicts on FULL val.
Test: average of 5 fold-models.

Outputs:
  oof/hier_fine_other_oof.npy        (N_train, 3) — P(L3, L4, L5 | x, C_other)
  oof/hier_fine_other_test_probs.npy (N_test, 3)

Usage:
  python scripts/train_hier_fine_other.py [--gpu]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("=== train_hier_fine_other.py starting ===", flush=True)

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

    other_mask = (y6 == 3) | (y6 == 4) | (y6 == 5)
    print(f"Other subset: {int(other_mask.sum())} rows  "
          f"({(y6==3).sum()} L3 / {(y6==4).sum()} L4 / {(y6==5).sum()} L5)", flush=True)

    # Map labels: L3 → 0, L4 → 1, L5 → 2
    y_local = np.zeros_like(y6)
    y_local[y6 == 3] = 0
    y_local[y6 == 4] = 1
    y_local[y6 == 5] = 2

    params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=31, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=10,
        verbose=-1, seed=42, num_threads=16,
    )
    if gpu:
        params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)
    NBR = 400

    oof_probs = np.zeros((len(y6), 3), dtype=np.float64)
    test_probs_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    fold_f1s = []
    for k, (tr, va) in enumerate(folds):
        tr_other = tr[other_mask[tr]]
        print(f"  Fold {k}: tr_other={len(tr_other)}  va={len(va)}", flush=True)
        # Inverse-frequency weights within the ternary task
        weights = make_class_weights(y_local[tr_other], n_classes=3)
        ds = lgb.Dataset(X[tr_other], label=y_local[tr_other], weight=weights)
        m = lgb.train(params, ds, num_boost_round=NBR)

        oof_probs[va] = m.predict(X[va])
        test_probs_sum += m.predict(Xte)

        va_other = va[other_mask[va]]
        if len(va_other) > 0:
            preds = oof_probs[va_other].argmax(axis=1)
            f = float(f1_score(y_local[va_other], preds, average="macro"))
            fold_f1s.append(f)
            print(f"    fold L3/L4/L5 F1 on other val: {f:.4f}", flush=True)

    test_probs = test_probs_sum / len(folds)

    other_idx = np.where(other_mask)[0]
    oof_preds_other = oof_probs[other_idx].argmax(axis=1)
    overall_f1 = float(f1_score(y_local[other_idx], oof_preds_other, average="macro"))
    print(f"\nOOF L3/L4/L5 ternary F1-macro on other subset: {overall_f1:.4f}", flush=True)
    print(f"  fold mean: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)
    print(classification_report(y_local[other_idx], oof_preds_other, target_names=["L3", "L4", "L5"], digits=4))

    np.save(ROOT / "oof" / "hier_fine_other_oof.npy", oof_probs.astype(np.float32))
    np.save(ROOT / "oof" / "hier_fine_other_test_probs.npy", test_probs.astype(np.float32))
    print(f"\nSaved hier_fine_other_oof.npy {oof_probs.shape}", flush=True)
    print(f"Saved hier_fine_other_test_probs.npy {test_probs.shape}", flush=True)


if __name__ == "__main__":
    main()
