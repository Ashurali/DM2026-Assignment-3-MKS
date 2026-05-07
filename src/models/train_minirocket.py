"""MiniRocket (Dempster et al. 2021) on raw 6×300 sequences.

MiniRocket transforms each sequence into ~9,996 random-convolutional features
in seconds, then a linear classifier (or LGBM for probabilities) maps to labels.
Often top-3 on the UCR/UEA time-series benchmark suites; very fast to train.

Different inductive bias from CNN-BiLSTM/Transformer (PPV-pooled fixed kernels,
no learned weights) → real ensemble decorrelation. Probability output is
essential for blending.

Usage:
    pip install sktime
    python -m src.models.train_minirocket --name v1 [--lr-classifier] [--tune]

Outputs:
    oof/minirocket_<name>_oof.npy
    oof/minirocket_<name>_test_probs.npy
    oof/minirocket_<name>_meta.json
    submissions/sub_minirocket_<name>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.cv import cv_score, to_submission

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42


def make_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return inv[y]


def load_seq_cache():
    cx = ROOT / "data" / "seq_train.npy"
    cy = ROOT / "data" / "seq_y_train.npy"
    cxte = ROOT / "data" / "seq_test.npy"
    cti = ROOT / "data" / "seq_test_ids.npy"
    if not all(p.exists() for p in [cx, cy, cxte, cti]):
        raise SystemExit(
            "Sequence cache missing — run train_cnn_bilstm.py once to build it."
        )
    return np.load(cx), np.load(cy), np.load(cxte), np.load(cti)


def fit_minirocket(X_seq: np.ndarray) -> tuple:
    """Fit MiniRocket on (N, 6, 300) array and return (transformer, transformed)."""
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    print(f"Fitting MiniRocket on {X_seq.shape}...")
    t0 = time.time()
    # MiniRocketMultivariate expects (N, n_channels, series_len) numpy array OR a 3D nested DataFrame
    # In recent sktime versions it accepts numpy.ndarray of shape (N, C, T).
    rocket = MiniRocketMultivariate(num_kernels=9996, random_state=SEED)
    rocket.fit(X_seq)
    transformed = rocket.transform(X_seq)
    if hasattr(transformed, "values"):
        transformed = transformed.values
    print(f"  fit + transform train: {time.time() - t0:.1f}s   features: {transformed.shape}")
    return rocket, transformed.astype(np.float32)


def transform_minirocket(rocket, X_seq: np.ndarray) -> np.ndarray:
    t0 = time.time()
    transformed = rocket.transform(X_seq)
    if hasattr(transformed, "values"):
        transformed = transformed.values
    print(f"  transform: {time.time() - t0:.1f}s  shape: {transformed.shape}")
    return transformed.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--lr-classifier", action="store_true",
                   help="Use Logistic Regression instead of LightGBM as the head.")
    p.add_argument("--tune", action="store_true", help="Optuna for the LGBM head.")
    p.add_argument("--n-trials", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Xtr_seq, ytr, Xte_seq, test_ids = load_seq_cache()

    print(f"Train seqs: {Xtr_seq.shape}  Test seqs: {Xte_seq.shape}")
    rocket, X = fit_minirocket(Xtr_seq)
    Xte = transform_minirocket(rocket, Xte_seq)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta_train["user_id"].values

    if args.lr_classifier:
        print("\n== MiniRocket + Logistic Regression (5-fold GKF) ==")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        def fit_predict(Xtr, ytr, Xva):
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED, C=1.0,
            ).fit(sc.transform(Xtr), ytr)
            probs = clf.predict_proba(sc.transform(Xva))
            return probs.argmax(axis=1), probs

        mean, std, oof_preds, oof_probs = cv_score(
            fit_predict, X, ytr, groups, n_splits=5, n_classes=N_CLASSES,
            checkpoint_name=f"minirocket_{args.name}_final",
        )

        # Final model
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
        ).fit(sc.transform(X), ytr)
        test_probs = clf.predict_proba(sc.transform(Xte))
    else:
        print("\n== MiniRocket + LightGBM (5-fold GKF) ==")
        import lightgbm as lgb

        params = dict(
            objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
            learning_rate=0.05, num_leaves=63, feature_fraction=0.7,
            bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
            verbose=-1, seed=SEED, num_threads=16,
        )
        num_boost_round = 500

        if args.tune:
            from src.models.train_lgbm_full import maybe_tune
            tuned = maybe_tune(X, ytr, groups, params, smote=False, n_trials=args.n_trials,
                                study_name=f"minirocket_{args.name}")
            num_boost_round = int(tuned.pop("__optuna_best_nbr", 500))
            params = tuned

        def fit_predict(Xtr, ytr, Xva):
            w_tr = make_class_weights(ytr, N_CLASSES)
            ds = lgb.Dataset(Xtr, label=ytr, weight=w_tr)
            m = lgb.train(params, ds, num_boost_round=num_boost_round)
            probs = m.predict(Xva)
            return probs.argmax(axis=1), probs

        mean, std, oof_preds, oof_probs = cv_score(
            fit_predict, X, ytr, groups, n_splits=5, n_classes=N_CLASSES,
            checkpoint_name=f"minirocket_{args.name}_final",
        )

        w_full = make_class_weights(ytr, N_CLASSES)
        full_ds = lgb.Dataset(X, label=ytr, weight=w_full)
        final_model = lgb.train(params, full_ds, num_boost_round=num_boost_round)
        test_probs = final_model.predict(Xte)

    from sklearn.metrics import f1_score, classification_report
    oof_macro = float(f1_score(ytr, oof_preds, average="macro"))
    per_class_f1 = f1_score(ytr, oof_preds, average=None)
    print(f"\nFold-mean F1: {mean:.4f} ± {std:.4f}  |  OOF macro: {oof_macro:.4f}")
    print(classification_report(ytr, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"minirocket_{args.name}_oof.npy", oof_probs.astype(np.float32))
    np.save(ROOT / "oof" / f"minirocket_{args.name}_test_probs.npy", test_probs.astype(np.float32))
    test_preds = test_probs.argmax(axis=1)
    sub_path = ROOT / "submissions" / f"sub_minirocket_{args.name}.csv"
    to_submission(test_ids, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    head = "MiniRocket + LR" if args.lr_classifier else "MiniRocket + LightGBM"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_minirocket_{args.name} | "
                f"{head} (~9996 random conv features) | "
                f"{mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")

    sidecar = {
        "model": f"minirocket_{args.name}",
        "head": "lr" if args.lr_classifier else "lgbm",
        "n_features": X.shape[1],
        "cv_f1_mean": mean, "cv_f1_std": std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "seed": SEED,
    }
    with open(ROOT / "oof" / f"minirocket_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


if __name__ == "__main__":
    main()
