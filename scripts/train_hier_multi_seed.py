"""Train all 3 hierarchical stages with multi-seed averaging + XGBoost partner
for fine_walk (the L1-vs-L2 binary, our identified bottleneck).

What this builds (saved to oof/):
  hier_coarse_ms_oof.npy / _test_probs.npy            — 3 seeds of LGBM, averaged
  hier_fine_walk_ms_oof.npy / _test_probs.npy         — 3 seeds LGBM + 3 seeds XGB, all averaged
  hier_fine_other_ms_oof.npy / _test_probs.npy        — 3 seeds of LGBM, averaged

Why these specific changes (and not others):
  - Multi-seed averaging within each stage: hierarchical stages each see less
    data than the joint 6-way (especially stage 2a with 5053 walking rows and
    stage 2b with 1324 other-cohort rows). LGBM bagging already adds variance,
    so per-seed variation is real and averaging-after-cal helps generalization.
    This is DIFFERENT from the failed combo_v2 multi-seed because there:
    we averaged at the level-2 stacker level on a saturated 805-feature set
    (where averaging diluted the L2 minority signal). Here we average each
    stage independently, then compose, then blend — so L2 minority signal
    survives within the binary stage 2a where it isn't fighting majority
    classes.

  - XGBoost partner for fine_walk only: stage 2a is the one that handles the
    L1↔L2 boundary that drove LB 0.7984 → 0.8107. Diversifying the model
    family there (LGBM + XGB) gives the most leverage. Other stages are
    not bottlenecks; LGBM alone is fine.

Usage:
  python scripts/train_hier_multi_seed.py --gpu --seeds 17 23 41
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("=== train_hier_multi_seed.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hier_common import build_feature_blocks, make_class_weights, label_to_super, ROOT
from src.utils.cv import make_folds
from src.utils.lgbm import lgbm_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    p.add_argument("--num-boost-coarse", type=int, default=500)
    p.add_argument("--num-boost-walk", type=int, default=400)
    p.add_argument("--num-boost-other", type=int, default=400)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Stage 1: coarse 3-way (multi-seed LGBM)
# ─────────────────────────────────────────────────────────────────────
def train_coarse_one_seed(X, Xte, y3, folds, seed, gpu, nbr):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
        verbose=-1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
        data_random_seed=seed + 3, num_threads=16,
    )
    params.update(**lgbm_device())  # auto-GPU when available, else CPU
    oof = np.zeros((len(y3), 3), dtype=np.float64)
    test_probs_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for k, (tr, va) in enumerate(folds):
        w_tr = make_class_weights(y3[tr], n_classes=3)
        ds = lgb.Dataset(X[tr], label=y3[tr], weight=w_tr)
        m = lgb.train(params, ds, num_boost_round=nbr)
        oof[va] = m.predict(X[va])
        test_probs_sum += m.predict(Xte)
    test_probs = test_probs_sum / len(folds)
    print(f"    [coarse seed={seed}] OOF F1: "
          f"{f1_score(y3, oof.argmax(1), average='macro'):.4f}", flush=True)
    return oof, test_probs


def train_coarse_ms(X, Xte, y6, groups, folds, seeds, gpu, nbr):
    print(f"\n── Stage 1: coarse 3-way (seeds={seeds}) ──", flush=True)
    y3 = label_to_super(y6)
    oof_sum = np.zeros((len(y3), 3), dtype=np.float64)
    test_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for seed in seeds:
        oof_s, test_s = train_coarse_one_seed(X, Xte, y3, folds, seed, gpu, nbr)
        oof_sum += oof_s
        test_sum += test_s
    oof = oof_sum / len(seeds)
    test_probs = test_sum / len(seeds)
    oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
    test_probs = test_probs / np.clip(test_probs.sum(axis=1, keepdims=True), 1e-12, None)

    from sklearn.metrics import f1_score
    print(f"  Coarse MS OOF F1: {f1_score(y3, oof.argmax(1), average='macro'):.4f}", flush=True)
    np.save(ROOT / "oof" / "hier_coarse_ms_oof.npy", oof.astype(np.float32))
    np.save(ROOT / "oof" / "hier_coarse_ms_test_probs.npy", test_probs.astype(np.float32))
    return oof, test_probs


# ─────────────────────────────────────────────────────────────────────
# Stage 2a: fine_walk binary L1-vs-L2 (multi-seed LGBM + multi-seed XGBoost)
# ─────────────────────────────────────────────────────────────────────
def fine_walk_lgbm_seed(X, Xte, y_bin, walk_mask, folds, seed, gpu, nbr):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=15,
        verbose=-1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
        data_random_seed=seed + 3, num_threads=16,
    )
    params.update(**lgbm_device())  # auto-GPU when available, else CPU
    oof = np.zeros(len(y_bin), dtype=np.float64)
    test_sum = np.zeros(len(Xte), dtype=np.float64)
    for k, (tr, va) in enumerate(folds):
        tr_walk = tr[walk_mask[tr]]
        n1 = int((y_bin[tr_walk] == 0).sum())
        n2 = int((y_bin[tr_walk] == 1).sum())
        weights = np.where(y_bin[tr_walk] == 0, 1.0, max(1.0, n1 / max(n2, 1))).astype(np.float64)
        ds = lgb.Dataset(X[tr_walk], label=y_bin[tr_walk], weight=weights)
        m = lgb.train(params, ds, num_boost_round=nbr)
        oof[va] = m.predict(X[va])
        test_sum += m.predict(Xte)
    test_probs = test_sum / len(folds)
    walk_idx = np.where(walk_mask)[0]
    f = float(f1_score(y_bin[walk_idx], (oof[walk_idx] >= 0.5).astype(int), average="macro"))
    print(f"    [walk LGBM seed={seed}] OOF binary F1 on walking: {f:.4f}", flush=True)
    return oof, test_probs


def fine_walk_xgb_seed(X, Xte, y_bin, walk_mask, folds, seed, gpu, nbr):
    import xgboost as xgb
    from sklearn.metrics import f1_score
    params = dict(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
        min_child_weight=4, reg_lambda=1.0, seed=seed,
        tree_method="hist", verbosity=0,
    )
    if gpu:
        params["device"] = "cuda"
    oof = np.zeros(len(y_bin), dtype=np.float64)
    test_sum = np.zeros(len(Xte), dtype=np.float64)
    for k, (tr, va) in enumerate(folds):
        tr_walk = tr[walk_mask[tr]]
        n1 = int((y_bin[tr_walk] == 0).sum())
        n2 = int((y_bin[tr_walk] == 1).sum())
        scale_pos_weight = max(1.0, n1 / max(n2, 1))
        p = dict(params)
        p["scale_pos_weight"] = scale_pos_weight
        dtrain = xgb.DMatrix(X[tr_walk], label=y_bin[tr_walk])
        m = xgb.train(p, dtrain, num_boost_round=nbr)
        oof[va] = m.predict(xgb.DMatrix(X[va]))
        test_sum += m.predict(xgb.DMatrix(Xte))
    test_probs = test_sum / len(folds)
    walk_idx = np.where(walk_mask)[0]
    f = float(f1_score(y_bin[walk_idx], (oof[walk_idx] >= 0.5).astype(int), average="macro"))
    print(f"    [walk XGB  seed={seed}] OOF binary F1 on walking: {f:.4f}", flush=True)
    return oof, test_probs


def train_fine_walk_ms(X, Xte, y6, groups, folds, seeds, gpu, nbr):
    print(f"\n── Stage 2a: fine_walk L1-vs-L2 (seeds={seeds}, LGBM+XGB) ──", flush=True)
    walk_mask = (y6 == 1) | (y6 == 2)
    y_bin = np.zeros_like(y6)
    y_bin[y6 == 2] = 1

    pieces_oof = []
    pieces_test = []
    for seed in seeds:
        oof_l, test_l = fine_walk_lgbm_seed(X, Xte, y_bin, walk_mask, folds, seed, gpu, nbr)
        pieces_oof.append(oof_l)
        pieces_test.append(test_l)
        oof_x, test_x = fine_walk_xgb_seed(X, Xte, y_bin, walk_mask, folds, seed, gpu, nbr)
        pieces_oof.append(oof_x)
        pieces_test.append(test_x)

    p_l2_oof = np.mean(pieces_oof, axis=0)
    p_l2_test = np.mean(pieces_test, axis=0)

    from sklearn.metrics import f1_score, classification_report
    walk_idx = np.where(walk_mask)[0]
    preds = (p_l2_oof[walk_idx] >= 0.5).astype(int)
    f_overall = float(f1_score(y_bin[walk_idx], preds, average="macro"))
    print(f"  Fine_walk MS OOF binary F1 on walking: {f_overall:.4f}", flush=True)
    print(classification_report(y_bin[walk_idx], preds, target_names=["L1", "L2"], digits=4))

    oof_2col = np.stack([1 - p_l2_oof, p_l2_oof], axis=1).astype(np.float32)
    test_2col = np.stack([1 - p_l2_test, p_l2_test], axis=1).astype(np.float32)
    np.save(ROOT / "oof" / "hier_fine_walk_ms_oof.npy", oof_2col)
    np.save(ROOT / "oof" / "hier_fine_walk_ms_test_probs.npy", test_2col)
    return oof_2col, test_2col


# ─────────────────────────────────────────────────────────────────────
# Stage 2b: fine_other ternary L3/L4/L5 (multi-seed LGBM)
# ─────────────────────────────────────────────────────────────────────
def fine_other_seed(X, Xte, y_local, other_mask, folds, seed, gpu, nbr):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=31, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=10,
        verbose=-1, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
        data_random_seed=seed + 3, num_threads=16,
    )
    params.update(**lgbm_device())  # auto-GPU when available, else CPU
    oof = np.zeros((len(y_local), 3), dtype=np.float64)
    test_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for k, (tr, va) in enumerate(folds):
        tr_other = tr[other_mask[tr]]
        weights = make_class_weights(y_local[tr_other], n_classes=3)
        ds = lgb.Dataset(X[tr_other], label=y_local[tr_other], weight=weights)
        m = lgb.train(params, ds, num_boost_round=nbr)
        oof[va] = m.predict(X[va])
        test_sum += m.predict(Xte)
    test_probs = test_sum / len(folds)
    other_idx = np.where(other_mask)[0]
    preds = oof[other_idx].argmax(axis=1)
    f = float(f1_score(y_local[other_idx], preds, average="macro"))
    print(f"    [other seed={seed}] OOF ternary F1 on other: {f:.4f}", flush=True)
    return oof, test_probs


def train_fine_other_ms(X, Xte, y6, groups, folds, seeds, gpu, nbr):
    print(f"\n── Stage 2b: fine_other L3/L4/L5 (seeds={seeds}) ──", flush=True)
    other_mask = (y6 == 3) | (y6 == 4) | (y6 == 5)
    y_local = np.zeros_like(y6)
    y_local[y6 == 3] = 0
    y_local[y6 == 4] = 1
    y_local[y6 == 5] = 2

    oof_sum = np.zeros((len(y_local), 3), dtype=np.float64)
    test_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for seed in seeds:
        oof_s, test_s = fine_other_seed(X, Xte, y_local, other_mask, folds, seed, gpu, nbr)
        oof_sum += oof_s
        test_sum += test_s
    oof = oof_sum / len(seeds)
    test_probs = test_sum / len(seeds)
    oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
    test_probs = test_probs / np.clip(test_probs.sum(axis=1, keepdims=True), 1e-12, None)

    from sklearn.metrics import f1_score, classification_report
    other_idx = np.where(other_mask)[0]
    preds = oof[other_idx].argmax(axis=1)
    f = float(f1_score(y_local[other_idx], preds, average="macro"))
    print(f"  Fine_other MS OOF ternary F1: {f:.4f}", flush=True)
    print(classification_report(y_local[other_idx], preds, target_names=["L3", "L4", "L5"], digits=4))

    np.save(ROOT / "oof" / "hier_fine_other_ms_oof.npy", oof.astype(np.float32))
    np.save(ROOT / "oof" / "hier_fine_other_ms_test_probs.npy", test_probs.astype(np.float32))
    return oof, test_probs


def main():
    args = parse_args()

    X, Xte, block_names = build_feature_blocks(include_combo_oof=True)
    print(f"Features: {block_names}", flush=True)
    print(f"X {X.shape}  Xte {Xte.shape}", flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    y6 = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)

    train_coarse_ms(X, Xte, y6, groups, folds, args.seeds, args.gpu, args.num_boost_coarse)
    train_fine_walk_ms(X, Xte, y6, groups, folds, args.seeds, args.gpu, args.num_boost_walk)
    train_fine_other_ms(X, Xte, y6, groups, folds, args.seeds, args.gpu, args.num_boost_other)
    print("\n=== train_hier_multi_seed.py done ===", flush=True)
    print("Next: python scripts/hier_compose_blend_v2.py", flush=True)


if __name__ == "__main__":
    main()
