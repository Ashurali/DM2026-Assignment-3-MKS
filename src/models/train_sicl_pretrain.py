"""SICL contrastive pretraining + linear-classifier fine-tune (Tier B.3).

Two stages:

1. Pretraining: encoder + projection head trained with SICL loss for
   ~50-100 epochs on all train data (any subject, any class). Two
   augmentation views per sample, sicl_loss with q_s=0.5, τ=0.1.
2. Fine-tuning (5-fold GKF): freeze encoder, train a linear classifier
   on top with class-weighted CE for each fold. Get OOF predictions
   for the combo to use as features.

Encoder = CNN-BiLSTM backbone (no class head). Stage-1 outputs the
final.pt of the encoder. Stage-2 outputs per-fold linear classifier
probabilities (OOF) and a final-classifier on full data for test.

Usage:
    python -m src.models.train_sicl_pretrain --gpu --name sicl_v1 \
        [--pretrain-epochs 80] [--ft-epochs 30] [--per-file-norm]
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
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("ERROR: PyTorch not installed.", file=sys.stderr)
    raise

from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.utils.cv import make_folds, to_submission
from src.utils.checkpoint import run_dir, save_fold, load_fold
from src.models.cnn_bilstm import SeqDataset, worker_init_fn, augment_sample, DEFAULT_AUG_PROBS, CNNBiLSTM
from src.models.sicl import ContrastiveProjectionHead, sicl_loss

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


def build_or_load_seq_cache():
    cx = ROOT / "data" / "seq_train.npy"
    cy = ROOT / "data" / "seq_y_train.npy"
    cxte = ROOT / "data" / "seq_test.npy"
    cti = ROOT / "data" / "seq_test_ids.npy"
    if all(p.exists() for p in [cx, cy, cxte, cti]):
        return np.load(cx), np.load(cy), np.load(cxte), np.load(cti)
    raise SystemExit("Missing seq cache; run train_cnn_bilstm.py first.")


def make_class_weights(y, n_classes=N_CLASSES):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return torch.tensor(inv, dtype=torch.float32)


# ─── Encoder wrapper that exposes the (B, 256) embedding ────────────────────
class CNNBiLSTMEncoder(nn.Module):
    """Same backbone as CNNBiLSTM but returns the 256-d embedding instead
    of class logits. We reuse the existing CNNBiLSTM and just take its
    pre-fc activation."""

    def __init__(self, in_channels: int = 6, lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.backbone = CNNBiLSTM(n_classes=N_CLASSES, in_channels=in_channels, lstm_hidden=lstm_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manually run all backbone layers up to (and including) attention pool
        # without going through fc.
        x = self.backbone.bn(x)
        x = self.backbone.conv_block1(x)
        x = self.backbone.conv_block2(x)
        x = self.backbone.conv_block3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.backbone.lstm(x)
        emb = self.backbone.attn(x)  # (B, 256)
        return self.backbone.dropout(emb)


# ─── Pretraining dataset: returns 2 augmented views + label + subject ──────
class SICLPretrainDataset(Dataset):
    def __init__(self, X, y, subjects, aug_probs, per_file_norm=False, concat_stats=False, seed=42):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.sub = subjects.astype(np.int64)
        self.aug_probs = {**DEFAULT_AUG_PROBS, **(aug_probs or {})}
        self.per_file_norm = per_file_norm
        self.concat_stats = concat_stats and per_file_norm
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def _maybe_norm(self, x):
        if not self.per_file_norm:
            return x
        if self.concat_stats:
            mean_track = np.linalg.norm(x[:3], axis=0, keepdims=True)
            std_track = np.linalg.norm(x[3:], axis=0, keepdims=True)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / (std + 1e-6)
        if self.concat_stats:
            x = np.concatenate([x, mean_track, std_track], axis=0)
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        v1 = augment_sample(x, self._rng, **self.aug_probs)
        v2 = augment_sample(x, self._rng, **self.aug_probs)
        v1 = self._maybe_norm(v1)
        v2 = self._maybe_norm(v2)
        return torch.from_numpy(v1).float(), torch.from_numpy(v2).float(), int(self.y[idx]), int(self.sub[idx])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sicl_v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--pretrain-epochs", type=int, default=80)
    p.add_argument("--ft-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128, help="Larger batch helps contrastive (more negatives).")
    p.add_argument("--lr-pretrain", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--q-s", type=float, default=0.5, help="Same-subject down-weight (1.0 = standard SupCon, 0.0 = ignore same-subject negatives entirely).")
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--concat-stats", action="store_true")
    return p.parse_args()


def pretrain(args, Xtr, ytr, subjects_idx, device, ckpt_dir):
    encoder_ckpt = ckpt_dir / "encoder.pt"
    if encoder_ckpt.exists():
        print(f"Loading existing pretrained encoder: {encoder_ckpt}")
        in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
        encoder = CNNBiLSTMEncoder(in_channels=in_ch).to(device)
        ck = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        return encoder

    print(f"\n=== SICL pretraining ({args.pretrain_epochs} epochs) ===")
    aug_probs = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    ds = SICLPretrainDataset(Xtr, ytr, subjects_idx, aug_probs,
                              per_file_norm=args.per_file_norm, concat_stats=args.concat_stats,
                              seed=SEED)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                         pin_memory=True, drop_last=True)

    in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
    encoder = CNNBiLSTMEncoder(in_channels=in_ch).to(device)
    head = ContrastiveProjectionHead(in_dim=256, proj_dim=128, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                    lr=args.lr_pretrain, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    for ep in range(args.pretrain_epochs):
        encoder.train(); head.train()
        t0 = time.time()
        total_loss = 0.0
        n_batch = 0
        for v1, v2, lbl, sub in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            sub = sub.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                e1 = encoder(v1); e2 = encoder(v2)
                z1 = head(e1); z2 = head(e2)
                loss = sicl_loss(z1, z2, lbl, sub, temperature=args.temperature, q_s=args.q_s)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()
        print(f"  Pretrain ep {ep+1}/{args.pretrain_epochs}: loss={total_loss/max(1,n_batch):.4f}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "epochs": args.pretrain_epochs,
                "args": vars(args)}, encoder_ckpt)
    print(f"Saved encoder: {encoder_ckpt}")
    return encoder


def finetune_one_fold(fold_k, encoder, Xtr, ytr, Xva, yva, args, class_w, device, ckpt_dir):
    """Linear classifier on top of frozen encoder, trained with class-weighted CE."""
    aug_probs = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    train_ds = SeqDataset(Xtr, ytr, training=True, seed=SEED + fold_k, aug_probs=aug_probs,
                          per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    val_ds = SeqDataset(Xva, yva, training=False,
                        per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(256, N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr_ft, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    for ep in range(args.ft_epochs):
        classifier.train()
        t0 = time.time()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                with torch.no_grad():
                    emb = encoder(xb)
                logits = classifier(emb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * xb.size(0)
        scheduler.step()

        classifier.eval()
        val_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    emb = encoder(xb)
                    logits = classifier(emb)
                val_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_preds = val_probs.argmax(axis=1)
        val_f1 = float(f1_score(yva, val_preds, average="macro"))
        print(f"    FT fold {fold_k} ep {ep+1}/{args.ft_epochs}: loss={total/len(train_ds):.4f}  val_f1={val_f1:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    classifier.load_state_dict(best_state)
    classifier.eval()
    val_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = classifier(encoder(xb))
            val_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    val_probs = np.concatenate(val_probs)
    return val_probs.argmax(axis=1), val_probs, best_val_f1, classifier


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"SICL run: {args.name}  device={device}  q_s={args.q_s}  τ={args.temperature}")

    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta_train["user_id"].values
    le = LabelEncoder()
    subjects_idx = le.fit_transform(groups)

    folds = make_folds(groups, n_splits=5)
    class_w = make_class_weights(ytr)

    ckpt_dir = run_dir(f"sicl_{args.name}")

    # Stage 1: pretrain on ALL train data (subjects + classes provided)
    encoder = pretrain(args, Xtr, ytr, subjects_idx, device, ckpt_dir)

    # Stage 2: per-fold linear classifier
    oof_probs = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
    oof_preds = np.zeros(len(ytr), dtype=np.int64)
    fold_f1s = []

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    classifiers = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        cached = load_fold(f"sicl_{args.name}", k)
        if cached is not None:
            preds_k, probs_k = cached
            f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
            print(f"\nFold {k}: F1-macro = {f1_k:.4f} (resumed)")
            oof_preds[va_idx] = preds_k
            oof_probs[va_idx] = probs_k
            fold_f1s.append(f1_k)
            continue

        print(f"\n=== SICL FT Fold {k}: train={len(tr_idx)}  val={len(va_idx)} ===", flush=True)
        preds_k, probs_k, best_f1, classifier = finetune_one_fold(
            k, encoder, Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx],
            args, class_w, device, ckpt_dir)
        save_fold(f"sicl_{args.name}", k, preds_k, probs_k)
        f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
        print(f"Fold {k}: F1-macro = {f1_k:.4f}")
        oof_preds[va_idx] = preds_k
        oof_probs[va_idx] = probs_k
        fold_f1s.append(f1_k)
        classifiers.append(classifier)

    cv_mean = float(np.mean(fold_f1s))
    cv_std = float(np.std(fold_f1s))
    oof_macro = float(f1_score(ytr, oof_preds, average="macro"))
    per_class_f1 = f1_score(ytr, oof_preds, average=None)
    print(f"\nSICL CV F1: {cv_mean:.4f} ± {cv_std:.4f}  |  OOF: {oof_macro:.4f}")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: {f:.4f}")
    print(classification_report(ytr, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"sicl_{args.name}_oof.npy", oof_probs)

    # Final: train classifier on FULL data + frozen encoder, predict test
    print("\n=== Final SICL classifier (all train + frozen encoder) ===")
    aug_probs = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    full_ds = SeqDataset(Xtr, ytr, training=True, seed=SEED, aug_probs=aug_probs,
                          per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    full_loader = DataLoader(full_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers, pin_memory=True, drop_last=True)
    classifier = nn.Linear(256, N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr_ft, weight_decay=1e-4)
    epoch_budget = max(15, args.ft_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_budget)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    for ep in range(epoch_budget):
        classifier.train()
        t0 = time.time()
        for xb, yb in full_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                with torch.no_grad():
                    emb = encoder(xb)
                logits = classifier(emb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        print(f"  Final FT ep {ep+1}/{epoch_budget}  ({time.time()-t0:.1f}s)", flush=True)

    test_ds = SeqDataset(Xte, training=False, per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    test_loader = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    classifier.eval()
    test_probs = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb = encoder(xb)
                logits = classifier(emb)
            test_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    test_probs = np.concatenate(test_probs).astype(np.float32)
    test_preds = test_probs.argmax(axis=1)
    np.save(ROOT / "oof" / f"sicl_{args.name}_test_probs.npy", test_probs)
    sub_path = ROOT / "submissions" / f"sub_sicl_{args.name}.csv"
    to_submission(test_ids, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_sicl_{args.name} | "
                f"SICL pretrain (q_s={args.q_s}, τ={args.temperature}) + linear FT | "
                f"{cv_mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")
    sidecar = {
        "model": f"sicl_{args.name}",
        "pretrain_epochs": args.pretrain_epochs, "ft_epochs": args.ft_epochs,
        "q_s": args.q_s, "temperature": args.temperature,
        "per_file_norm": args.per_file_norm, "concat_stats": args.concat_stats,
        "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "fold_best_val_f1": fold_f1s, "seed": SEED,
    }
    with open(ROOT / "oof" / f"sicl_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


if __name__ == "__main__":
    main()
