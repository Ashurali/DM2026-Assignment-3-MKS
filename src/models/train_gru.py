"""Plain end-to-end CNN-BiGRU (NO Evidential Alignment) — an ensemble base.

The EA experiment degraded the BiGRU (OOF 0.65 / LB 0.72) by freezing the
backbone and retraining a linear evidential head. But the backbone itself
reached ~0.71 in-distribution, so a clean end-to-end BiGRU is worth having as a
decorrelated GRU base alongside the CNN-BiLSTM / Transformer / InceptionTime.

This trains the CNN-BiGRU exactly like train_cnn_bilstm (GroupKFold(5) by
user_id, augmentations, mixup, AMP, cosine LR, early-stop on the val fold),
reusing the validated `erm_train` loop. Outputs match the other bases:

    oof/gru_<name>_oof.npy            (N_train, 6)
    oof/gru_<name>_test_probs.npy     (N_test, 6)
    oof/gru_<name>_meta.json
    submissions/sub_gru_<name>.csv

Usage (SERVER, env dm2026-a3):
    python -m src.models.train_gru --gpu --name v1 [--gru-hidden 128] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from src.utils.cv import make_folds, to_submission
from src.utils.checkpoint import run_dir, save_fold, load_fold
from src.models.cnn_bigru import CNNBiGRU, SeqDataset, worker_init_fn  # noqa: F401
from src.models.train_cnn_bilstm import (
    build_or_load_seq_cache, make_class_weights, mixup_batch,  # noqa: F401
)
from src.models.train_gru_evidential import erm_train  # reuse the end-to-end CNN-BiGRU loop

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6


@torch.no_grad()
def predict_probs(model, X, args, device) -> np.ndarray:
    ds = SeqDataset(X, training=False, per_file_norm=args.per_file_norm,
                    concat_stats=args.concat_stats)
    ld = DataLoader(ds, batch_size=args.batch * 2, shuffle=False,
                    num_workers=args.n_workers, pin_memory=True)
    model.eval()
    out = []
    for xb in ld:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--erm-val-frac", type=float, default=0.1,
                   help="random holdout for the FINAL model's early stopping")
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--concat-stats", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Run: gru_{args.name}  device={device}  seed={args.seed}  "
          f"gru_hidden={args.gru_hidden}", flush=True)

    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    print(f"Xtr {Xtr.shape}  ytr {ytr.shape}  Xte {Xte.shape}", flush=True)
    groups = pd.read_parquet(ROOT / "data" / "meta_train.parquet")["user_id"].values
    folds = make_folds(groups, n_splits=5)
    class_w = make_class_weights(ytr)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    name = f"gru_{args.name}"
    run_dir(name)
    oof = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
    fold_f1s = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        cached = load_fold(name, k)
        if cached is not None:
            _, probs_k = cached
            oof[va_idx] = probs_k
            fold_f1s.append(float(f1_score(ytr[va_idx], probs_k.argmax(1), average="macro")))
            print(f"\nFold {k}: F1={fold_f1s[-1]:.4f} (resumed)", flush=True)
            continue
        print(f"\n=== Fold {k}: train={len(tr_idx)} val={len(va_idx)} ===", flush=True)
        # End-to-end CNN-BiGRU; early-stop on the val fold (matches CNN-BiLSTM).
        model = erm_train(Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx],
                          args, class_w, device, tag=f"fold{k}")
        probs_k = predict_probs(model, Xtr[va_idx], args, device)
        oof[va_idx] = probs_k
        save_fold(name, k, probs_k.argmax(1), probs_k)
        fold_f1s.append(float(f1_score(ytr[va_idx], probs_k.argmax(1), average="macro")))
        print(f"Fold {k}: F1-macro = {fold_f1s[-1]:.4f}", flush=True)

    oof_macro = float(f1_score(ytr, oof.argmax(1), average="macro"))
    per_class = f1_score(ytr, oof.argmax(1), average=None)
    print(f"\nCV F1 (fold mean): {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)
    print(f"OOF F1-macro: {oof_macro:.4f}", flush=True)
    print("Per-class OOF F1: " + " ".join(f"c{c}={f:.4f}" for c, f in enumerate(per_class)), flush=True)
    print(classification_report(ytr, oof.argmax(1), digits=4), flush=True)
    (ROOT / "oof").mkdir(exist_ok=True)
    np.save(ROOT / "oof" / f"{name}_oof.npy", oof)

    # ── Final model on all data → test ──
    print("\n=== Final model: all train → test ===", flush=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ytr))
    nv = max(1, int(round(args.erm_val_frac * len(ytr))))
    model = erm_train(Xtr[perm[nv:]], ytr[perm[nv:]], Xtr[perm[:nv]], ytr[perm[:nv]],
                      args, class_w, device, tag="final")
    test_probs = predict_probs(model, Xte, args, device)
    np.save(ROOT / "oof" / f"{name}_test_probs.npy", test_probs)
    sub_path = ROOT / "submissions" / f"sub_{name}.csv"
    to_submission(test_ids, test_probs.argmax(1), str(sub_path))

    with open(ROOT / "submissions" / "log.md", "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_{name} | "
                f"CNN-BiGRU end-to-end (no EA) | "
                f"{np.mean(fold_f1s):.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class]} |\n")
    with open(ROOT / "oof" / f"{name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"model": name, "oof_f1_macro": oof_macro,
                   "per_class_f1": [float(x) for x in per_class],
                   "fold_f1s": fold_f1s, "args": vars(args)}, fh, indent=2)
    print(f"\nDone. OOF F1={oof_macro:.4f}  submission={sub_path}", flush=True)


if __name__ == "__main__":
    main()
