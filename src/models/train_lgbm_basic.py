"""Day-1 submission #2: LightGBM on the basic feature catalog.

5-fold GroupKFold by user (the canonical CV harness — see src/utils/cv.py),
class-weighted via sample_weight derived from inverse class frequency to
handle the 33× imbalance the EDA surfaced.

Outputs:
- `oof/lgbm_basic_v1_oof.npy`   — out-of-fold probabilities, (N_train, 6)
- `submissions/sub02_lgbm_basic.csv`
- log row appended to `submissions/log.md`
"""
from __future__ import annotations
try:  # optional Intel scikit-learn speedup; not required (absent from requirements.txt)
    from sklearnex import patch_sklearn
    patch_sklearn()
except Exception:
    pass
import json
from pathlib import Path, PureWindowsPath
from datetime import date

import numpy as np
import pandas as pd
import lightgbm as lgb

import sys as _sys, pathlib as _pathlib  # repo-root bootstrap so `python src/models/<file>.py` works (not just -m)
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
from src.utils.cv import cv_score, to_submission
from src.utils.lgbm import lgbm_device
from src.features.basic import build_dataset

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return a (N,) sample_weight array via inverse class frequency.

    Equivalent to sklearn `class_weight='balanced'` translated to per-sample
    weights so we can pass it to `lgb.Dataset(weight=...)`.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    # Avoid div-by-zero if a class is missing from a fold
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)  # sklearn's "balanced" formula
    return inv[y]


def fit_predict_factory(params: dict, num_boost_round: int):
    def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray):
        w_tr = make_class_weights(y_tr, N_CLASSES)
        ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        model = lgb.train(params, ds, num_boost_round=num_boost_round)
        probs = model.predict(X_va)
        preds = probs.argmax(axis=1)
        return preds, probs
    return fit_predict


def main() -> None:
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet", engine="pyarrow")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet", engine="pyarrow")
    def fix_server_path(local_path):
        """Correctly parses Windows strings on Linux and rebases to server ROOT."""
        # PureWindowsPath recognizes '\' even when running on Linux
        win_path = PureWindowsPath(local_path)
        parts = win_path.parts
        
        if "data" in parts:
            data_index = parts.index("data")
            # Create a relative path from 'data' onwards
            relative_data_path = Path(*parts[data_index:])
            # Join it with the server's ROOT (e.g., /home/nycu813/mike/...)
            return ROOT / relative_data_path
        return Path(local_path)

    print("Adjusting paths for server environment...")
    meta_train["path"] = meta_train["path"].apply(fix_server_path)
    meta_test["path"] = meta_test["path"].apply(fix_server_path)
    
    print(f"Train rows: {len(meta_train)} | Test rows: {len(meta_test)}")
    print("Building train features...")
    Xtr_df = build_dataset(meta_train["path"].tolist(), meta_train["file_id"].tolist())
    print("Building test features...")
    Xte_df = build_dataset(meta_test["path"].tolist(), meta_test["file_id"].tolist())

    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    X = Xtr_df[feat_cols].values
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values

    print(f"Feature matrix: {X.shape}")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols[:4]} ... {feat_cols[-3:]}")

    params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=20,
        verbose=-1,
        seed=SEED,
        **lgbm_device(),  # auto-GPU when a GPU-enabled LightGBM build is present, else CPU
        gpu_device_id=0,      
        num_threads=16,
        gpu_use_dp=False
    )
    num_boost_round = 500

    fit_predict = fit_predict_factory(params, num_boost_round)
    print("\nRunning 5-fold GroupKFold CV...")
    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=N_CLASSES
    )

    # Per-class CV F1 for diagnostic / minority-class tracking
    from sklearn.metrics import f1_score, classification_report
    per_class_f1 = f1_score(y, oof_preds, average=None)
    print("\nPer-class CV F1-macro:")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: F1 = {f:.4f}  (n={int((y == c).sum())})")
    print("\nClassification report (OOF):")
    print(classification_report(y, oof_preds, digits=4))

    oof_path = ROOT / "oof" / "lgbm_basic_v1_oof.npy"
    np.save(oof_path, oof_probs)
    print(f"Saved OOF probs to {oof_path}  shape={oof_probs.shape}")

    # Train on full train, predict test
    print("\nTraining final model on full train set...")
    w_full = make_class_weights(y, N_CLASSES)
    full_ds = lgb.Dataset(X, label=y, weight=w_full)
    final_model = lgb.train(params, full_ds, num_boost_round=num_boost_round)
    test_probs = final_model.predict(Xte_df[feat_cols].values)
    test_preds = test_probs.argmax(axis=1)

    sub_path = ROOT / "submissions" / "sub02_lgbm_basic.csv"
    to_submission(meta_test["file_id"].values, test_preds, str(sub_path))

    # Save final-model test probs for later blending
    np.save(ROOT / "oof" / "lgbm_basic_v1_test_probs.npy", test_probs)

    # Append a log row
    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"| {date.today().isoformat()} | sub02_lgbm_basic | "
            f"LGBM ({len(feat_cols)} basic features, class-weighted) | "
            f"{mean:.4f} | _pending_ | _pending_ | "
            f"first competitive; per-class F1: {[round(float(x), 4) for x in per_class_f1]} |\n"
        )
    print(f"\nLogged to {log_path}")

    # Also save a small JSON sidecar for reproducibility
    sidecar = {
        "model": "lgbm_basic_v1",
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "params": params,
        "num_boost_round": num_boost_round,
        "cv_f1_mean": mean,
        "cv_f1_std": std,
        "per_class_f1": [float(x) for x in per_class_f1],
        "seed": SEED,
    }
    with open(ROOT / "oof" / "lgbm_basic_v1_meta.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)


if __name__ == "__main__":
    main()
