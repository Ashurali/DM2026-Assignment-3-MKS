"""Train XGBoost and CatBoost on the same 271-feature engineered catalog.

Different tree-construction algorithms vs LightGBM → real (small) decorrelation
in the ensemble. Reuses the same feature cache (data/feat_train_none.parquet)
that train_lgbm_full.py builds, so this is fast if the cache is warm.

Usage:
    python -m src.models.train_xgb_cat --gpu --models xgb cat
    # or just one:
    python -m src.models.train_xgb_cat --gpu --models cat

Outputs (per model):
    oof/<model>_v1_oof.npy          (N_train, 6)
    oof/<model>_v1_test_probs.npy   (N_test, 6)
    oof/<model>_v1_meta.json
    submissions/sub_<model>_v1.csv
    + appended row in submissions/log.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from src.utils.cv import cv_score, to_submission
from src.features.build import build_dataset

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42


def fix_server_path(local_path) -> Path:
    win = PureWindowsPath(str(local_path))
    parts = win.parts
    if "data" in parts:
        idx = parts.index("data")
        return ROOT / Path(*parts[idx:])
    return Path(local_path)


def make_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return inv[y]


def load_features():
    cache_train = ROOT / "data" / "feat_train_none.parquet"
    cache_test = ROOT / "data" / "feat_test_none.parquet"
    if cache_train.exists() and cache_test.exists():
        print("Loading cached 271-feature parquet")
        return pd.read_parquet(cache_train), pd.read_parquet(cache_test)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    meta_train["path"] = meta_train["path"].apply(fix_server_path)
    meta_test["path"] = meta_test["path"].apply(fix_server_path)
    print("Building 271-feature catalog from CSVs (this is the slow path)...")
    Xtr_df = build_dataset(meta_train["path"].tolist(), meta_train["file_id"].tolist())
    Xte_df = build_dataset(meta_test["path"].tolist(), meta_test["file_id"].tolist())
    Xtr_df.to_parquet(cache_train, index=False)
    Xte_df.to_parquet(cache_test, index=False)
    return Xtr_df, Xte_df


# ─── XGBoost ─────────────────────────────────────────────────────────────────
def train_xgb(X, y, groups, Xte, args):
    import xgboost as xgb
    params = dict(
        objective="multi:softprob",
        num_class=N_CLASSES,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=600,
        tree_method="hist",
        device="cuda" if args.gpu else "cpu",
        random_state=SEED,
        verbosity=0,
    )

    def fit_predict(Xtr, ytr, Xva):
        w_tr = make_class_weights(ytr, N_CLASSES)
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr, sample_weight=w_tr)
        probs = m.predict_proba(Xva)
        return probs.argmax(axis=1), probs

    print("\n== XGBoost 5-fold GroupKFold ==")
    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=N_CLASSES,
        checkpoint_name="xgb_v1_final",
    )

    # Final model on all data
    print("Training XGBoost on full train...")
    w_full = make_class_weights(y, N_CLASSES)
    m = xgb.XGBClassifier(**params)
    m.fit(X, y, sample_weight=w_full)
    test_probs = m.predict_proba(Xte)
    return mean, std, oof_preds, oof_probs, test_probs


# ─── CatBoost ────────────────────────────────────────────────────────────────
def train_cat(X, y, groups, Xte, args):
    from catboost import CatBoostClassifier
    params = dict(
        loss_function="MultiClass",
        classes_count=N_CLASSES,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        iterations=600,
        random_seed=SEED,
        task_type="GPU" if args.gpu else "CPU",
        devices="0" if args.gpu else None,
        auto_class_weights="Balanced",  # CatBoost has built-in balancing
        verbose=False,
    )

    def fit_predict(Xtr, ytr, Xva):
        m = CatBoostClassifier(**params)
        m.fit(Xtr, ytr, verbose=False)
        probs = m.predict_proba(Xva)
        return probs.argmax(axis=1), probs

    print("\n== CatBoost 5-fold GroupKFold ==")
    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=N_CLASSES,
        checkpoint_name="cat_v1_final",
    )

    print("Training CatBoost on full train...")
    m = CatBoostClassifier(**params)
    m.fit(X, y, verbose=False)
    test_probs = m.predict_proba(Xte)
    return mean, std, oof_preds, oof_probs, test_probs


# ─── Driver ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--models", nargs="+", default=["xgb", "cat"], choices=["xgb", "cat"])
    return p.parse_args()


def report_and_save(name: str, mean, std, oof_preds, oof_probs, test_probs, meta_test, y, feat_cols):
    from sklearn.metrics import classification_report, f1_score
    oof_macro = float(f1_score(y, oof_preds, average="macro"))
    per_class_f1 = f1_score(y, oof_preds, average=None)
    print(f"\n{name} fold-mean F1: {mean:.4f} ± {std:.4f}  |  OOF macro: {oof_macro:.4f}")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: F1 = {f:.4f}  (n={int((y == c).sum())})")
    print(classification_report(y, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"{name}_v1_oof.npy", oof_probs)
    np.save(ROOT / "oof" / f"{name}_v1_test_probs.npy", test_probs)
    test_preds = test_probs.argmax(axis=1)
    sub_path = ROOT / "submissions" / f"sub_{name}_v1.csv"
    to_submission(meta_test["file_id"].values, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_{name}_v1 | "
                f"{name.upper()} ({len(feat_cols)} features, class-weighted) | "
                f"{mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")

    sidecar = {
        "model": f"{name}_v1",
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "cv_f1_mean": mean, "cv_f1_std": std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "seed": SEED,
    }
    with open(ROOT / "oof" / f"{name}_v1_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


def main():
    args = parse_args()
    Xtr_df, Xte_df = load_features()
    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    X = Xtr_df[feat_cols].values.astype(np.float64)
    Xte = Xte_df[feat_cols].values.astype(np.float64)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values

    print(f"Features: {X.shape}  Test: {Xte.shape}")

    if "xgb" in args.models:
        mean, std, oof_preds, oof_probs, test_probs = train_xgb(X, y, groups, Xte, args)
        report_and_save("xgb", mean, std, oof_preds, oof_probs, test_probs, meta_test, y, feat_cols)

    if "cat" in args.models:
        mean, std, oof_preds, oof_probs, test_probs = train_cat(X, y, groups, Xte, args)
        report_and_save("cat", mean, std, oof_preds, oof_probs, test_probs, meta_test, y, feat_cols)


if __name__ == "__main__":
    main()
