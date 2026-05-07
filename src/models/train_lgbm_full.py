"""Phase-3 LightGBM on the full engineered feature catalog.

Adds, vs. `train_lgbm_basic.py`:
- `src/features/build.py` (~287 features instead of 60)
- Per-fold SMOTE oversampling of minority classes (optional, --smote)
- Feature-group ablation (--exclude GROUP1,GROUP2,...)
- `--name SUFFIX` for differentiating ablation runs in the OOF folder
- Optional Optuna HP tuning (--tune; default: skip)

Outputs (with `--name v1` by default):
- `oof/lgbm_full_<name>_oof.npy`
- `oof/lgbm_full_<name>_test_probs.npy`
- `oof/lgbm_full_<name>_meta.json`
- `submissions/sub_lgbm_full_<name>.csv`
- A row appended to `submissions/log.md`

Server-side path adjustment (handled in `train_lgbm_basic.py`) is applied
identically here so the meta_*.parquet generated on Windows is portable.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.utils.cv import cv_score, to_submission
from src.features.build import build_dataset, ALL_GROUPS

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42


def fix_server_path(local_path) -> Path:
    """Reroot a Windows-style path stored in parquet to the current machine.

    Same logic as in train_lgbm_basic.py: find the segment after 'data' and
    reattach it under this script's ROOT.
    """
    win = PureWindowsPath(str(local_path))
    parts = win.parts
    if "data" in parts:
        idx = parts.index("data")
        return ROOT / Path(*parts[idx:])
    return Path(local_path)


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)  # sklearn-balanced
    return inv[y]


def fit_predict_factory(params: dict, num_boost_round: int, smote: bool):
    """Return a fold-level fit_predict closure. SMOTE is applied INSIDE the
    train fold only — never on val. This keeps the GroupKFold honest."""
    def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray):
        if smote:
            from imblearn.over_sampling import SMOTE
            # k_neighbors=3 is conservative; label-4 only has ~28/fold so we
            # need < its smallest in-fold count. SMOTE auto-fails if too few.
            min_count = int(np.bincount(y_tr).min())
            k = max(1, min(3, min_count - 1))
            try:
                sm = SMOTE(random_state=SEED, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except ValueError as e:
                # If a class has too few samples, fall back to no-SMOTE silently
                print(f"  SMOTE skipped this fold: {e}")
        w_tr = make_class_weights(y_tr, N_CLASSES)
        ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        model = lgb.train(params, ds, num_boost_round=num_boost_round)
        probs = model.predict(X_va)
        preds = probs.argmax(axis=1)
        return preds, probs

    return fit_predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 LGBM training")
    p.add_argument("--exclude", default="", help="Comma-separated feature groups to exclude (for ablation).")
    p.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling per fold.")
    p.add_argument("--name", default="v1", help="Suffix for output files (e.g. 'v1', 'no_fft', 'smote_v1').")
    p.add_argument("--tune", action="store_true", help="Run an Optuna HP search.")
    p.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials (default: 30).")
    p.add_argument(
        "--gpu", action="store_true",
        help="Use device='cuda'/'gpu' for LightGBM training.")
    p.add_argument(
        "--cache-features", action="store_true",
        help="Save the assembled feature matrices to data/feat_*.parquet for reuse across runs.")
    return p.parse_args()


def get_features(meta_train: pd.DataFrame, meta_test: pd.DataFrame, exclude: list[str], cache: bool):
    # Sort exclude list so cache key is deterministic regardless of CLI order
    cache_key = "_".join(sorted(exclude)) if exclude else "none"
    cache_train = ROOT / "data" / f"feat_train_{cache_key}.parquet"
    cache_test = ROOT / "data" / f"feat_test_{cache_key}.parquet"

    if cache and cache_train.exists() and cache_test.exists():
        print(f"Loading cached features: {cache_train.name}")
        Xtr_df = pd.read_parquet(cache_train)
        Xte_df = pd.read_parquet(cache_test)
    else:
        print(f"Building train features (excluding: {exclude or 'none'})...")
        Xtr_df = build_dataset(meta_train["path"].tolist(), meta_train["file_id"].tolist(), exclude=exclude)
        print("Building test features...")
        Xte_df = build_dataset(meta_test["path"].tolist(), meta_test["file_id"].tolist(), exclude=exclude)
        if cache:
            Xtr_df.to_parquet(cache_train, index=False)
            Xte_df.to_parquet(cache_test, index=False)
            print(f"Cached features → {cache_train.name}, {cache_test.name}")

    return Xtr_df, Xte_df


def maybe_tune(X: np.ndarray, y: np.ndarray, groups: np.ndarray, base_params: dict, smote: bool, n_trials: int = 30) -> dict:
    """Run an Optuna search optimising fold-mean F1-macro."""
    import optuna
    from src.utils.cv import cv_score

    def objective(trial: "optuna.trial.Trial") -> float:
        params = dict(base_params)
        params.update(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 50),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        )
        nbr = trial.suggest_int("num_boost_round", 200, 800)
        fp = fit_predict_factory(params, nbr, smote)
        mean, _, _, _ = cv_score(fp, X, y, groups, n_splits=5, n_classes=N_CLASSES, verbose=False)
        return mean

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nOptuna best CV F1-macro: {study.best_value:.4f}")
    print(f"Optuna best params: {study.best_params}")
    out = dict(base_params)
    out.update({k: v for k, v in study.best_params.items() if k != "num_boost_round"})
    out["__optuna_best_nbr"] = study.best_params.get("num_boost_round", 500)
    return out


def main() -> None:
    args = parse_args()
    exclude = [g.strip() for g in args.exclude.split(",") if g.strip()]
    if exclude:
        unknown = [g for g in exclude if g not in ALL_GROUPS]
        if unknown:
            raise SystemExit(f"Unknown feature groups: {unknown}. Known: {ALL_GROUPS}")

    print(f"Run name: {args.name}")
    print(f"Exclude groups: {exclude or 'none'}")
    print(f"SMOTE: {args.smote}")
    print(f"GPU: {args.gpu}")
    print(f"Tune: {args.tune}")

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet", engine="pyarrow")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet", engine="pyarrow")
    meta_train["path"] = meta_train["path"].apply(fix_server_path)
    meta_test["path"] = meta_test["path"].apply(fix_server_path)

    Xtr_df, Xte_df = get_features(meta_train, meta_test, exclude, cache=args.cache_features)

    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    X = Xtr_df[feat_cols].values.astype(np.float64)
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values

    print(f"\nFeature matrix: {X.shape}")
    print(f"Feature count: {len(feat_cols)}")

    base_params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=20,
        lambda_l1=0.0,
        lambda_l2=0.0,
        verbose=-1,
        seed=SEED,
        num_threads=16,
    )
    if args.gpu:
        # LightGBM with CUDA (per the user's working server config)
        base_params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)

    num_boost_round = 500
    if args.tune:
        tuned = maybe_tune(X, y, groups, base_params, args.smote, n_trials=args.n_trials)
        num_boost_round = int(tuned.pop("__optuna_best_nbr", 500))
        params = tuned
    else:
        params = base_params

    print("\nRunning 5-fold GroupKFold CV...")
    fp = fit_predict_factory(params, num_boost_round, args.smote)
    mean, std, oof_preds, oof_probs = cv_score(fp, X, y, groups, n_splits=5, n_classes=N_CLASSES)

    from sklearn.metrics import f1_score, classification_report
    per_class_f1 = f1_score(y, oof_preds, average=None)
    print("\nPer-class CV F1-macro (OOF):")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: F1 = {f:.4f}  (n={int((y == c).sum())})")
    print("\nClassification report (OOF):")
    print(classification_report(y, oof_preds, digits=4))

    oof_path = ROOT / "oof" / f"lgbm_full_{args.name}_oof.npy"
    np.save(oof_path, oof_probs)
    print(f"Saved OOF probs → {oof_path}  shape={oof_probs.shape}")

    print("\nTraining final model on full train set (no SMOTE)...")
    # We deliberately do NOT apply SMOTE on the final-model train set: SMOTE
    # was a CV-fold tool to validate the technique. For the final model we use
    # all real data with class weighting only.
    w_full = make_class_weights(y, N_CLASSES)
    full_ds = lgb.Dataset(X, label=y, weight=w_full)
    final_model = lgb.train(params, full_ds, num_boost_round=num_boost_round)
    test_probs = final_model.predict(Xte_df[feat_cols].values.astype(np.float64))
    test_preds = test_probs.argmax(axis=1)

    sub_path = ROOT / "submissions" / f"sub_lgbm_full_{args.name}.csv"
    to_submission(meta_test["file_id"].values, test_preds, str(sub_path))
    np.save(ROOT / "oof" / f"lgbm_full_{args.name}_test_probs.npy", test_probs)

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        notes = [f"per-class F1 {[round(float(x), 4) for x in per_class_f1]}"]
        if exclude:
            notes.append(f"excluded {exclude}")
        if args.smote:
            notes.append("SMOTE")
        if args.tune:
            notes.append("Optuna 50 trials")
        f.write(
            f"| {date.today().isoformat()} | sub_lgbm_full_{args.name} | "
            f"LGBM full ({len(feat_cols)} features) | "
            f"{mean:.4f} | _pending_ | _pending_ | "
            f"{'; '.join(notes)} |\n"
        )
    print(f"Logged to {log_path}")

    sidecar = {
        "model": f"lgbm_full_{args.name}",
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "params": params,
        "num_boost_round": num_boost_round,
        "exclude_groups": exclude,
        "smote": args.smote,
        "tuned": args.tune,
        "cv_f1_mean": mean,
        "cv_f1_std": std,
        "per_class_f1": [float(x) for x in per_class_f1],
        "seed": SEED,
    }
    with open(ROOT / "oof" / f"lgbm_full_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"Saved sidecar → oof/lgbm_full_{args.name}_meta.json")


if __name__ == "__main__":
    main()
