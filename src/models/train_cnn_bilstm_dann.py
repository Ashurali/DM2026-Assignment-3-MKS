"""Train CNN-BiLSTM with DANN — domain head over user_id (Tier B.4).

Same scaffold as train_cnn_bilstm.py: 5-fold GroupKFold, AMP, per-fold
+ per-epoch checkpointing. Adds:

- Domain classifier head (predicts user_id from encoder output, with GRL)
- λ ramp 0 → max_lambda over training (gamma=10 default)
- Combined loss: class CE + λ * domain CE
- Domain labels are encoded user_ids in the TRAINING fold; at val time
  the domain head is ignored (domain shift, by design)

Usage:
    python -m src.models.train_cnn_bilstm_dann --gpu --name dann_v1 \
        [--max-lambda 0.5] [--per-file-norm] [--epochs 40]
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
from src.models.cnn_bilstm import SeqDataset, worker_init_fn
from src.models.cnn_bilstm_dann import CNNBiLSTM_DANN, lambda_schedule

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42
CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


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
        print("Loading sequence cache from data/seq_*.npy ...")
        return np.load(cx), np.load(cy), np.load(cxte), np.load(cti)
    raise SystemExit("Missing seq cache; run train_cnn_bilstm.py first to build it.")


def make_class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return torch.tensor(inv, dtype=torch.float32)


def mixup_batch(x, y_class, y_domain, alpha, rng):
    """Mixup for both class and domain labels."""
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y_class, y_class[perm], y_domain, y_domain[perm], lam


class SeqDatasetWithDomain(Dataset):
    """Wraps SeqDataset and also returns domain (user_id encoded as int)."""

    def __init__(self, X, y, domain_ids, training, seed, aug_probs, per_file_norm, concat_stats):
        self.inner = SeqDataset(X, y, training=training, seed=seed, aug_probs=aug_probs,
                                 per_file_norm=per_file_norm, concat_stats=concat_stats)
        self.domain_ids = domain_ids.astype(np.int64)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x, y_c = self.inner[idx]
        return x, int(y_c), int(self.domain_ids[idx])


def train_one_fold(fold_k, Xtr, ytr, Xva, yva, dom_tr, dom_va, n_domains_train, args, class_w, device, ckpt_dir):
    aug_probs = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    train_ds = SeqDatasetWithDomain(Xtr, ytr, dom_tr, training=True, seed=SEED + fold_k,
                                     aug_probs=aug_probs, per_file_norm=args.per_file_norm,
                                     concat_stats=args.concat_stats)
    val_ds = SeqDataset(Xva, yva, training=False,
                        per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                               worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
    model = CNNBiLSTM_DANN(n_classes=N_CLASSES, n_domains=n_domains_train, in_channels=in_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
    criterion_cls = nn.CrossEntropyLoss(weight=class_w.to(device))
    criterion_dom = nn.CrossEntropyLoss()
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

    total_steps = len(train_loader) * args.epochs
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss_c = 0.0
        train_loss_d = 0.0
        for step, (xb, yb_c, yb_d) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb_c = yb_c.to(device, non_blocking=True)
            yb_d = yb_d.to(device, non_blocking=True)
            # λ ramp: progress goes 0..1 across all training steps
            global_step = epoch * len(train_loader) + step
            progress = global_step / max(1, total_steps)
            lam = lambda_schedule(progress, gamma=args.gamma, max_lambda=args.max_lambda)
            model.set_lambda(lam)

            optimizer.zero_grad(set_to_none=True)
            if args.mixup_alpha > 0:
                xb, ya, yb_perm, dya, dyb_perm, lam_mix = mixup_batch(xb, yb_c, yb_d, args.mixup_alpha, rng)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits_c, logits_d, _ = model(xb)
                    loss_c = lam_mix * criterion_cls(logits_c, ya) + (1 - lam_mix) * criterion_cls(logits_c, yb_perm)
                    loss_d = lam_mix * criterion_dom(logits_d, dya) + (1 - lam_mix) * criterion_dom(logits_d, dyb_perm)
            else:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits_c, logits_d, _ = model(xb)
                    loss_c = criterion_cls(logits_c, yb_c)
                    loss_d = criterion_dom(logits_d, yb_d)
            loss = loss_c + loss_d  # GRL handles the sign on the encoder
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_c += loss_c.item() * xb.size(0)
            train_loss_d += loss_d.item() * xb.size(0)
        train_loss_c /= len(train_ds)
        train_loss_d /= len(train_ds)
        scheduler.step()

        # Validation: only the class head matters
        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits_c, _, _ = model(xb)
                val_probs.append(torch.softmax(logits_c.float(), dim=1).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_preds = val_probs.argmax(axis=1)
        val_f1 = float(f1_score(yva, val_preds, average="macro"))
        elapsed = time.time() - t0
        print(f"  Fold {fold_k} ep {epoch+1:>2d}/{args.epochs}: cls_loss={train_loss_c:.4f}  dom_loss={train_loss_d:.4f}  λ={lam:.3f}  val_f1={val_f1:.4f}  ({elapsed:.1f}s)", flush=True)
        history.append({"epoch": epoch + 1, "loss_c": train_loss_c, "loss_d": train_loss_d, "lam": lam, "val_f1": val_f1})

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

    print(f"  Fold {fold_k}: reloading best (val_f1={best_val_f1:.4f}) for OOF inference")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_c, _, _ = model(xb)
            val_probs.append(torch.softmax(logits_c.float(), dim=1).cpu().numpy())
    val_probs = np.concatenate(val_probs)
    return val_probs.argmax(axis=1), val_probs, {"history": history, "best_val_f1": best_val_f1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="dann_v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--max-lambda", type=float, default=0.5,
                   help="Maximum DANN GRL multiplier λ (typical 0.1-1.0).")
    p.add_argument("--gamma", type=float, default=10.0,
                   help="λ ramp shape parameter (higher = sharper ramp).")
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--concat-stats", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"DANN run: {args.name}  device={device}  max_lambda={args.max_lambda}")

    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta_train["user_id"].values

    le = LabelEncoder()
    domain_full = le.fit_transform(groups)  # 0..n_domains-1
    n_domains_full = int(len(le.classes_))
    print(f"Encoded domains: {n_domains_full}")

    folds = make_folds(groups, n_splits=5)
    class_w = make_class_weights(ytr)

    ckpt_dir = run_dir(f"cnn_bilstm_dann_{args.name}")
    oof_probs = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
    oof_preds = np.zeros(len(ytr), dtype=np.int64)
    fold_f1s = []

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    for k, (tr_idx, va_idx) in enumerate(folds):
        cached = load_fold(f"cnn_bilstm_dann_{args.name}", k)
        if cached is not None:
            preds_k, probs_k = cached
            f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
            print(f"\nFold {k}: F1-macro = {f1_k:.4f}  (resumed)")
            oof_preds[va_idx] = preds_k
            oof_probs[va_idx] = probs_k
            fold_f1s.append(f1_k)
            continue

        # Fold-local domain encoding (only train-fold subjects matter)
        # Each fold's train slice has a subset of all 60 users.
        fold_users = np.unique(groups[tr_idx])
        fold_le = LabelEncoder().fit(fold_users)
        dom_tr = fold_le.transform(groups[tr_idx])
        # Val users are disjoint by design — domain head ignores them
        dom_va = np.zeros(len(va_idx), dtype=np.int64)  # placeholder
        n_domains_train = int(len(fold_users))
        print(f"\n=== DANN Fold {k}: train={len(tr_idx)}  val={len(va_idx)}  n_domains_train={n_domains_train} ===")

        preds_k, probs_k, _ = train_one_fold(k, Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx],
                                             dom_tr, dom_va, n_domains_train,
                                             args, class_w, device, ckpt_dir)
        save_fold(f"cnn_bilstm_dann_{args.name}", k, preds_k, probs_k)
        f1_k = float(f1_score(ytr[va_idx], preds_k, average="macro"))
        print(f"Fold {k}: F1-macro = {f1_k:.4f}")
        oof_preds[va_idx] = preds_k
        oof_probs[va_idx] = probs_k
        fold_f1s.append(f1_k)

    cv_mean = float(np.mean(fold_f1s))
    cv_std = float(np.std(fold_f1s))
    oof_macro = float(f1_score(ytr, oof_preds, average="macro"))
    per_class_f1 = f1_score(ytr, oof_preds, average=None)
    print(f"\nDANN CV F1: {cv_mean:.4f} ± {cv_std:.4f}  |  OOF macro: {oof_macro:.4f}")
    for c, f in enumerate(per_class_f1):
        print(f"  class {c}: {f:.4f}")
    print(classification_report(ytr, oof_preds, digits=4))

    np.save(ROOT / "oof" / f"cnn_bilstm_dann_{args.name}_oof.npy", oof_probs)

    # Final retrain on full data with all 60 domains
    print("\n=== Final DANN model on full train (all 60 users as domains) ===")
    final_ckpt = ckpt_dir / "final.pt"
    if final_ckpt.exists():
        print("Loading existing final.pt")
        in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
        model = CNNBiLSTM_DANN(n_classes=N_CLASSES, n_domains=n_domains_full, in_channels=in_ch).to(device)
        ck = torch.load(final_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    else:
        epoch_budget = max(15, int(np.median([
            torch.load(ckpt_dir / f"fold_{k}_best.pt", map_location="cpu", weights_only=False)["epoch"]
            for k in range(5)
        ])) + 1)
        print(f"Retraining final for {epoch_budget} epochs")
        full_ds = SeqDatasetWithDomain(Xtr, ytr, domain_full, training=True, seed=SEED,
                                        aug_probs={"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp},
                                        per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
        full_loader = DataLoader(full_ds, batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                                  worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
        in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
        model = CNNBiLSTM_DANN(n_classes=N_CLASSES, n_domains=n_domains_full, in_channels=in_ch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_budget)
        scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))
        criterion_cls = nn.CrossEntropyLoss(weight=class_w.to(device))
        criterion_dom = nn.CrossEntropyLoss()
        rng = np.random.default_rng(SEED * 7919)
        total_steps = len(full_loader) * epoch_budget
        for ep in range(epoch_budget):
            model.train()
            t0 = time.time()
            for step, (xb, yc, yd) in enumerate(full_loader):
                xb = xb.to(device, non_blocking=True)
                yc = yc.to(device, non_blocking=True)
                yd = yd.to(device, non_blocking=True)
                progress = (ep * len(full_loader) + step) / max(1, total_steps)
                model.set_lambda(lambda_schedule(progress, args.gamma, args.max_lambda))
                optimizer.zero_grad(set_to_none=True)
                if args.mixup_alpha > 0:
                    xb, ya, yb_perm, dya, dyb_perm, lam_mix = mixup_batch(xb, yc, yd, args.mixup_alpha, rng)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        lc, ld, _ = model(xb)
                        loss_c = lam_mix * criterion_cls(lc, ya) + (1 - lam_mix) * criterion_cls(lc, yb_perm)
                        loss_d = lam_mix * criterion_dom(ld, dya) + (1 - lam_mix) * criterion_dom(ld, dyb_perm)
                else:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        lc, ld, _ = model(xb)
                        loss_c = criterion_cls(lc, yc)
                        loss_d = criterion_dom(ld, yd)
                loss = loss_c + loss_d
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            print(f"  Final ep {ep+1}/{epoch_budget}  ({time.time()-t0:.1f}s)", flush=True)
        torch.save({"model": model.state_dict(), "epochs": epoch_budget, "n_domains": n_domains_full}, final_ckpt)

    test_ds = SeqDataset(Xte, training=False,
                          per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    test_loader = DataLoader(test_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    model.eval()
    test_probs = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                lc, _, _ = model(xb)
            test_probs.append(torch.softmax(lc.float(), dim=1).cpu().numpy())
    test_probs = np.concatenate(test_probs).astype(np.float32)
    test_preds = test_probs.argmax(axis=1)
    np.save(ROOT / "oof" / f"cnn_bilstm_dann_{args.name}_test_probs.npy", test_probs)
    sub_path = ROOT / "submissions" / f"sub_cnn_bilstm_dann_{args.name}.csv"
    to_submission(test_ids, test_preds, str(sub_path))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_cnn_bilstm_dann_{args.name} | "
                f"CNN-BiLSTM + DANN (max_λ={args.max_lambda}) | "
                f"{cv_mean:.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class_f1]} |\n")
    sidecar = {
        "model": f"cnn_bilstm_dann_{args.name}",
        "max_lambda": args.max_lambda, "gamma": args.gamma,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "mixup_alpha": args.mixup_alpha, "patience": args.patience,
        "per_file_norm": args.per_file_norm, "concat_stats": args.concat_stats,
        "cv_f1_mean": cv_mean, "cv_f1_std": cv_std, "oof_f1_macro": oof_macro,
        "per_class_f1": [float(x) for x in per_class_f1],
        "fold_best_val_f1": fold_f1s, "seed": SEED, "n_domains_full": n_domains_full,
    }
    with open(ROOT / "oof" / f"cnn_bilstm_dann_{args.name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)


if __name__ == "__main__":
    main()
