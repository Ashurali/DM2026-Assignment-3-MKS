"""Train LGBM on combined feature catalogs.

Combines the existing 271-feature catalog with optional add-ons:
- catch22 features (132)         → total ~403
- DL embeddings (256/CNN, 128/Transformer)  → total ~659+
- All combined                    → total ~787

Usage:
    # 271 + catch22:
    python -m src.models.train_lgbm_combo --gpu --use-catch22 --name combo_v1

    # 271 + CNN-BiLSTM v1 embeddings:
    python -m src.models.train_lgbm_combo --gpu --use-cnn-emb cnn_bilstm_v1 --name combo_dl_v1

    # All add-ons + Optuna tuning:
    python -m src.models.train_lgbm_combo --gpu --use-catch22 \
        --use-cnn-emb cnn_bilstm_v1 --use-transformer-emb transformer_v1 --tune \
        --name combo_full_v1

The DL-embedding add-ons require a saved final.pt and a build_or_load_seq_cache
to extract embeddings on train+test. The script extracts the penultimate-layer
features by hooking the model's final FC layer.
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


def load_271_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_train = ROOT / "data" / "feat_train_none.parquet"
    cache_test = ROOT / "data" / "feat_test_none.parquet"
    if not (cache_train.exists() and cache_test.exists()):
        raise SystemExit(
            "Missing 271-feature cache. Run train_lgbm_full.py with --cache-features first."
        )
    return pd.read_parquet(cache_train), pd.read_parquet(cache_test)


def load_catch22_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_train = ROOT / "data" / "feat_catch22_train.parquet"
    cache_test = ROOT / "data" / "feat_catch22_test.parquet"
    if not (cache_train.exists() and cache_test.exists()):
        print("catch22 cache not found — building it now...")
        from src.features.catch22_features import main as build_catch22
        build_catch22()
    return pd.read_parquet(cache_train), pd.read_parquet(cache_test)


def extract_dl_embeddings(run_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features (post-pool/CLS, pre-fc) for train+test.

    Cached at oof/<run>_emb_train.npy / _emb_test.npy after first build.
    """
    cache_tr = ROOT / "oof" / f"{run_name}_emb_train.npy"
    cache_te = ROOT / "oof" / f"{run_name}_emb_test.npy"
    if cache_tr.exists() and cache_te.exists():
        print(f"Loading cached embeddings for '{run_name}'")
        return np.load(cache_tr), np.load(cache_te)

    print(f"Extracting embeddings from final.pt for '{run_name}'...")
    import torch
    from torch.utils.data import DataLoader

    # Load sequence cache
    seq_train = np.load(ROOT / "data" / "seq_train.npy")
    seq_test = np.load(ROOT / "data" / "seq_test.npy")

    final_ckpt = ROOT / "checkpoints" / run_name / "final.pt"
    if not final_ckpt.exists():
        raise SystemExit(f"No final.pt for '{run_name}' at {final_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if run_name.startswith("cnn_bilstm_"):
        from src.models.cnn_bilstm import CNNBiLSTM, SeqDataset
        model = CNNBiLSTM(n_classes=N_CLASSES)
        # Penultimate features = post-attention-pool, post-dropout, pre-fc.
        # Easiest: hook the input to model.fc.
        target_layer = "fc"
    elif run_name.startswith("transformer_"):
        from src.models.transformer import TransformerHAR
        from src.models.cnn_bilstm import SeqDataset  # SeqDataset is shared
        model = TransformerHAR(n_classes=N_CLASSES)
        target_layer = "fc"
    else:
        raise SystemExit(f"Unknown run prefix in '{run_name}'")

    ck = torch.load(final_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # Forward hook on the fc layer's input
    captured: list[torch.Tensor] = []

    def hook(_module, inp, _out):
        captured.append(inp[0].detach().cpu())

    handle = getattr(model, target_layer).register_forward_hook(hook)

    def forward_all(seq_array: np.ndarray) -> np.ndarray:
        ds = SeqDataset(seq_array, training=False)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        captured.clear()
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)):
                    xb = xb[0]
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    _ = model(xb)
        emb = torch.cat(captured, dim=0).float().numpy()
        return emb

    emb_tr = forward_all(seq_train)
    emb_te = forward_all(seq_test)
    handle.remove()

    np.save(cache_tr, emb_tr)
    np.save(cache_te, emb_te)
    print(f"Saved embeddings: train {emb_tr.shape}, test {emb_te.shape}")
    return emb_tr, emb_te


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="combo_v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--use-catch22", action="store_true")
    p.add_argument("--use-cnn-emb", default=None, help="e.g. cnn_bilstm_v1")
    p.add_argument("--use-transformer-emb", default=None, help="e.g. transformer_v1")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-trials", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Xtr_df, Xte_df = load_271_features()
    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    blocks_train: list[np.ndarray] = [Xtr_df[feat_cols].values.astype(np.float64)]
    blocks_test: list[np.ndarray] = [Xte_df[feat_cols].values.astype(np.float64)]
    block_names: list[str] = [f"engineered({len(feat_cols)})"]

    if args.use_catch22:
        c22_tr, c22_te = load_catch22_features()
        c22_cols = [c for c in c22_tr.columns if c != "file_id"]
        blocks_train.append(c22_tr[c22_cols].values.astype(np.float64))
        blocks_test.append(c22_te[c22_cols].values.astype(np.float64))
        block_names.append(f"catch22({len(c22_cols)})")

    if args.use_cnn_emb:
        emb_tr, emb_te = extract_dl_embeddings(args.use_cnn_emb)
        blocks_train.append(emb_tr.astype(np.float64))
        blocks_test.append(emb_te.astype(np.float64))
        block_names.append(f"cnn_emb({emb_tr.shape[1]})")

    if args.use_transformer_emb:
        emb_tr, emb_te = extract_dl_embeddings(args.use_transformer_emb)
        blocks_train.append(emb_tr.astype(np.float64))
        blocks_test.append(emb_te.astype(np.float64))
        block_names.append(f"transformer_emb({emb_tr.shape[1]})")

    X = np.concatenate(blocks_train, axis=1)
    Xte = np.concatenate(blocks_test, axis=1)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values

    print(f"\nFeature blocks used: {block_names}")
    print(f"Combined X: {X.shape}  Test: {Xte.shape}")

    import lightgbm as lgb
    base_params = dict(
        objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
        learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
        verbose=-1, seed=SEED, num_threads=16,
    )
    if args.gpu:
        base_params.update(device="cuda", gpu_device_id=0, gpu_use_dp=False)

    num_boost_round = 500
    params = base_params

    if args.tune:
        # Reuse Optuna tuner from train_lgbm_full
        from src.models.train_lgbm_full import maybe_tune
        # No SMOTE
        tuned = maybe_tune(X, y, groups, base_params, smote=False, n_trials=args.n_trials, study_name=f"lgbm_combo_{args.name}")
        num_boost_round = int(tuned.pop("__optuna_best_nbr", 500))
        params = tuned

    def fit_predict(Xtr, ytr, Xva):
        w_tr = make_class_weights(ytr, N_CLASSES)
        ds = lgb.Dataset(Xtr, label=ytr, weight=w_tr)
        m = lgb.train(params, ds, num_boost_round=num_boost_round)
        probs = m.predict(Xva)
        return probs.argmax(axis=1), probs

    print("\nRunning 5-fold GroupKFold CV...")
    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=N_CLASSES,
        checkpoint_name=f"lgbm_combo_{args.name}_final",
    )

    from sklearn.metrics import f1_score, classification_report
    per_class_f1 = f1_score(y, oof_preds, average=None)
    oof_macro = float(f1_score(y, oof_preds, average="macro"))
    print(f"\nFold-mean F1: {mean:.4f} ± {std:.4f}  |  OOF macro: {oof_macro:.4f}")
    print(classification_report(y, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"lgbm_combo_{args.name}_oof.npy", oof_probs)

    # Train final on all data
    w_full = make_class_weights(y, N_CLASSES)
    full_ds = lgb.Dataset(X, label=y, weight=w_full)
    final_model = lgb.train(params, full_ds, num_boost_round=num_boost_round)
    test_probs = final_model.predict(Xte)
    test_preds = test_probs.argmax(axis=1)
    np.save(ROOT / "oof" / f"lgbm_combo_{args.name}_test_probs.npy", test_probs)

    sub_path = ROOT / "submissions" / f"sub_lgbm_combo_{args.name}.csv"
    to_submission(meta_test["file_id"].values, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_lgbm_combo_{args.name} | "
                f"LGBM combo ({' + '.join(block_names)}) | "
                f"{mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")

    sidecar = {
        "model": f"lgbm_combo_{args.name}",
        "feature_blocks": block_names,
        "n_features_total": X.shape[1],
        "params": params, "num_boost_round": num_boost_round,
        "tuned": args.tune,
        "cv_f1_mean": mean, "cv_f1_std": std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "seed": SEED,
    }
    with open(ROOT / "oof" / f"lgbm_combo_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


if __name__ == "__main__":
    main()
