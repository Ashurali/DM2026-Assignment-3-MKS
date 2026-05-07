"""Phase-6 — train Transformer encoder with 5-fold GroupKFold, AMP, full checkpointing.

Identical training scaffold to train_cnn_bilstm.py (resumable per-fold + per-epoch
checkpointing per PROJECT_PLAN §4.5). Differences:

- Model: TransformerHAR (4-layer encoder, d_model=128, nhead=4) instead of CNN-BiLSTM
- Learning rate: 3e-4 (vs 1e-3) — Transformers are sensitive to LR
- Warmup: linear 5 epochs → cosine to end
- Mixup α: 0.4 default (vs 0.2) — Transformers benefit from stronger regularization
- Augmentation defaults same as CNN; can be overridden via flags

Usage:
    python -m src.models.train_transformer --gpu --name v1 [--epochs 50] [--batch 64]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    print("ERROR: PyTorch not installed.", file=sys.stderr)
    raise

from sklearn.metrics import classification_report, f1_score

from src.utils.cv import make_folds, to_submission
from src.utils.checkpoint import run_dir, save_fold, load_fold
from src.models.cnn_bilstm import SeqDataset, worker_init_fn
from src.models.transformer import TransformerHAR

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42

# Re-use CNN-BiLSTM script's helpers via local copy (kept identical for safety)
CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def fix_server_path(local_path) -> Path:
    win = PureWindowsPath(str(local_path))
    parts = win.parts
    if "data" in parts:
        idx = parts.index("data")
        return ROOT / Path(*parts[idx:])
    return Path(local_path)


def build_or_load_seq_cache():
    """Same cache as train_cnn_bilstm — reuses data/seq_*.npy if already built."""
    cache_x_train = ROOT / "data" / "seq_train.npy"
    cache_x_test = ROOT / "data" / "seq_test.npy"
    cache_y_train = ROOT / "data" / "seq_y_train.npy"
    cache_test_ids = ROOT / "data" / "seq_test_ids.npy"

    if all(p.exists() for p in [cache_x_train, cache_x_test, cache_y_train, cache_test_ids]):
        print("Loading sequence cache from data/seq_*.npy ...")
        return (
            np.load(cache_x_train),
            np.load(cache_y_train),
            np.load(cache_x_test),
            np.load(cache_test_ids),
        )

    print("Building sequence cache (one-time, ~1-2 min)...")
    from tqdm.auto import tqdm
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    meta_train["path"] = meta_train["path"].apply(fix_server_path)
    meta_test["path"] = meta_test["path"].apply(fix_server_path)

    def _read(path):
        df = pd.read_csv(path).sort_values("index").reset_index(drop=True)
        return df[CHANNELS].values.T.astype(np.float32)

    Xtr = np.zeros((len(meta_train), 6, 300), dtype=np.float32)
    for i, p in enumerate(tqdm(meta_train["path"].tolist(), desc="train")):
        Xtr[i] = _read(p)
    ytr = meta_train["label"].values.astype(np.int64)
    Xte = np.zeros((len(meta_test), 6, 300), dtype=np.float32)
    for i, p in enumerate(tqdm(meta_test["path"].tolist(), desc="test ")):
        Xte[i] = _read(p)
    test_ids = meta_test["file_id"].values.astype(np.int64)
    np.save(cache_x_train, Xtr); np.save(cache_x_test, Xte)
    np.save(cache_y_train, ytr); np.save(cache_test_ids, test_ids)
    return Xtr, ytr, Xte, test_ids


def make_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return torch.tensor(inv, dtype=torch.float32)


def mixup_batch(x, y, alpha, rng):
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def make_warmup_cosine(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup for warmup_epochs, then cosine annealing to 0."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def train_one_fold(fold_k, Xtr, ytr, Xva, yva, args, class_w, device, ckpt_dir):
    aug_probs = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    train_ds = SeqDataset(Xtr, ytr, training=True, seed=SEED + fold_k, aug_probs=aug_probs)
    val_ds = SeqDataset(Xva, yva, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                               worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    model = TransformerHAR(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = make_warmup_cosine(optimizer, args.warmup, args.epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    rng = np.random.default_rng(SEED + fold_k * 17)

    latest_ckpt = ckpt_dir / f"fold_{fold_k}_latest.pt"
    best_ckpt = ckpt_dir / f"fold_{fold_k}_best.pt"
    start_epoch, best_val_f1, no_improve = 0, -1.0, 0
    history = []

    if latest_ckpt.exists():
        print(f"  Resuming fold {fold_k} from {latest_ckpt.name}")
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val_f1 = ck.get("best_val_f1", -1.0)
        no_improve = ck.get("no_improve", 0)
        history = ck.get("history", [])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.mixup_alpha > 0:
                xb, ya, yb_perm, lam = mixup_batch(xb, yb, args.mixup_alpha, rng)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_perm)
            else:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            # Gradient clipping for Transformer stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
                val_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_preds = val_probs.argmax(axis=1)
        val_f1 = float(f1_score(yva, val_preds, average="macro"))
        elapsed = time.time() - t0
        print(f"  Fold {fold_k} ep {epoch+1:>2d}/{args.epochs}: train_loss={train_loss:.4f}  val_f1={val_f1:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_f1": val_f1, "lr": float(scheduler.get_last_lr()[0])})

        tmp = latest_ckpt.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val_f1": best_val_f1, "no_improve": no_improve, "history": history}, tmp)
        tmp.replace(latest_ckpt)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            tmp = best_ckpt.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "val_f1": best_val_f1, "epoch": epoch}, tmp)
            tmp.replace(best_ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Fold {fold_k}: early stopping at epoch {epoch+1} (best={best_val_f1:.4f})")
                break

    print(f"  Fold {fold_k}: reloading best-val state for OOF (best={best_val_f1:.4f})")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
            val_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    val_probs = np.concatenate(val_probs)
    return val_probs.argmax(axis=1), val_probs, {"history": history, "best_val_f1": best_val_f1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Run name: transformer_{args.name}")
    print(f"Device: {device} (cuda available: {torch.cuda.is_available()})")
    print(f"epochs={args.epochs} warmup={args.warmup} batch={args.batch} lr={args.lr} mixup_α={args.mixup_alpha}")

    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    print(f"Xtr {Xtr.shape}, Xte {Xte.shape}")

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)

    class_w = make_class_weights(ytr)
    print(f"Class weights: {class_w.numpy().round(3).tolist()}")

    ckpt_dir = run_dir(f"transformer_{args.name}")
    oof_probs = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
    oof_preds = np.zeros(len(ytr), dtype=np.int64)
    fold_f1s = []

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    for k, (tr_idx, va_idx) in enumerate(folds):
        cached = load_fold(f"transformer_{args.name}", k)
        if cached is not None:
            preds_k, probs_k = cached
            f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
            print(f"\nFold {k}: F1-macro = {f1_k:.4f}  (resumed)")
            oof_preds[va_idx] = preds_k
            oof_probs[va_idx] = probs_k
            fold_f1s.append(f1_k)
            continue

        print(f"\n=== Fold {k}: train={len(tr_idx)} val={len(va_idx)} ===")
        preds_k, probs_k, _ = train_one_fold(k, Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx],
                                             args, class_w, device, ckpt_dir)
        save_fold(f"transformer_{args.name}", k, preds_k, probs_k)
        f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
        print(f"Fold {k}: F1-macro = {f1_k:.4f}")
        oof_preds[va_idx] = preds_k
        oof_probs[va_idx] = probs_k
        fold_f1s.append(f1_k)

    cv_mean = float(np.mean(fold_f1s))
    cv_std = float(np.std(fold_f1s))
    oof_macro = float(f1_score(ytr, oof_preds, average="macro"))
    per_class_f1 = f1_score(ytr, oof_preds, average=None)
    print(f"\nCV F1-macro (fold mean): {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"OOF F1-macro (concat): {oof_macro:.4f}")
    print("Per-class OOF F1:")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: {f:.4f}  (n={int((ytr == c).sum())})")
    print("\nClassification report (OOF):")
    print(classification_report(ytr, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"transformer_{args.name}_oof.npy", oof_probs)

    print("\n=== Final model retrain on full train ===")
    final_ckpt = ckpt_dir / "final.pt"
    if final_ckpt.exists():
        print("Loading existing final.pt")
        model = TransformerHAR(n_classes=N_CLASSES).to(device)
        ck = torch.load(final_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    else:
        epoch_budget = max(20, int(np.median([
            torch.load(ckpt_dir / f"fold_{k}_best.pt", map_location="cpu", weights_only=False)["epoch"]
            for k in range(5)
        ])) + 1)
        print(f"Retraining final model for {epoch_budget} epochs")
        full_ds = SeqDataset(Xtr, ytr, training=True, seed=SEED, aug_probs={
            "p_rot": args.p_rot, "p_jitter": args.p_jitter,
            "p_scale": args.p_scale, "p_warp": args.p_warp,
        })
        full_loader = DataLoader(full_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                                  worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
        model = TransformerHAR(n_classes=N_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = make_warmup_cosine(optimizer, args.warmup, epoch_budget)
        scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
        criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
        rng = np.random.default_rng(SEED * 7919)
        for ep in range(epoch_budget):
            model.train()
            t0 = time.time()
            total = 0.0
            for xb, yb in full_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if args.mixup_alpha > 0:
                    xb, ya, yb_perm, lam = mixup_batch(xb, yb, args.mixup_alpha, rng)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits = model(xb)
                        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_perm)
                else:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * xb.size(0)
            scheduler.step()
            print(f"  Final ep {ep+1}/{epoch_budget}: loss={total/len(full_ds):.4f}  ({time.time()-t0:.1f}s)")
        torch.save({"model": model.state_dict(), "epochs": epoch_budget}, final_ckpt)

    test_ds = SeqDataset(Xte, training=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    model.eval()
    test_probs = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
            test_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    test_probs = np.concatenate(test_probs).astype(np.float32)
    test_preds = test_probs.argmax(axis=1)
    np.save(ROOT / "oof" / f"transformer_{args.name}_test_probs.npy", test_probs)
    sub_path = ROOT / "submissions" / f"sub_transformer_{args.name}.csv"
    to_submission(test_ids, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_transformer_{args.name} | "
                f"Transformer encoder (4 layers, d_model=128) | "
                f"{cv_mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")
    sidecar = {
        "model": f"transformer_{args.name}",
        "epochs": args.epochs, "warmup": args.warmup, "batch": args.batch, "lr": args.lr,
        "mixup_alpha": args.mixup_alpha, "patience": args.patience,
        "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "fold_best_val_f1": fold_f1s, "seed": SEED,
    }
    with open(ROOT / "oof" / f"transformer_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"\nLogged to {log_path}")


if __name__ == "__main__":
    main()
