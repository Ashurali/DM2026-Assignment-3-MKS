"""CNN-BiGRU + Evidential Alignment trainer (Ye, Zheng, Zhang, KDD 2025).

Per outer GroupKFold(5)-by-user fold:
  Phase A — ERM-train the CNN-BiGRU backbone (class-weighted CE, augs, mixup,
            AMP, cosine LR, early-stop on a user-disjoint calibration split
            carved from the TRAIN users).
  Phase B — freeze the backbone; Stage-1 Second-order Risk Minimization on the
            (cached) train embeddings: train an evidential last layer with
            L1 = −log(α_y/S) + λ_t·KL.            → θ1
  Phase C — Stage-2 Evidential Calibration on the calibration embeddings:
            class-balanced sampling + reweighted CE (w=1 if right else u(x))
            + β·‖θ2−θ1‖² anchor.                   → θ2
  OOF     — predict E[p] on the held-out outer-val fold (clean — those users
            were never seen by ERM / Stage-1 / Stage-2).

The calibration set comes from held-out TRAIN users (not the val fold) so the
outer-val OOF stays uncontaminated for the downstream stacker — our adaptation
of the paper's "half the validation set" to a stacking pipeline.

Outputs (mirrors train_cnn_bilstm):
    oof/gru_evidential_<name>_oof.npy          (N_train, 6)
    oof/gru_evidential_<name>_test_probs.npy   (N_test, 6)
    oof/gru_evidential_<name>_meta.json
    submissions/sub_gru_evidential_<name>.csv

Usage (SERVER, conda env dm2026-a3):
    python -m src.models.train_gru_evidential --gpu --name v1 \
        --epochs 40 --sorm-epochs 15 --calib-epochs 15 --eta 10 --beta 1.0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch not installed. In env dm2026-a3:", file=sys.stderr)
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)
    raise

from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

from src.utils.cv import make_folds, to_submission
from src.utils.checkpoint import run_dir, save_fold, load_fold
from src.models.cnn_bigru import CNNBiGRU, SeqDataset, worker_init_fn
from src.models.train_cnn_bilstm import (
    build_or_load_seq_cache, make_class_weights, mixup_batch,
)
from src.utils.evidential_align import (
    EvidentialHead, sorm_loss, reweighted_ce, anchor_penalty,
    probs_from_evidence, lambda_anneal,
)

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — ERM backbone
# ─────────────────────────────────────────────────────────────────────────────
def erm_train(
    X_tr, y_tr, X_val, y_val, args, class_w, device, tag: str,
) -> CNNBiGRU:
    """ERM-train the CNN-BiGRU; early-stop on val macro-F1. Returns best model."""
    aug = {"p_rot": args.p_rot, "p_jitter": args.p_jitter,
           "p_scale": args.p_scale, "p_warp": args.p_warp}
    tr_ds = SeqDataset(X_tr, y_tr, training=True, seed=SEED, aug_probs=aug,
                       per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    va_ds = SeqDataset(X_val, y_val, training=False,
                       per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.n_workers, worker_init_fn=worker_init_fn,
                       pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch * 2, shuffle=False,
                       num_workers=args.n_workers, pin_memory=True)

    in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
    model = CNNBiGRU(n_classes=N_CLASSES, in_channels=in_ch,
                     gru_hidden=args.gru_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    rng = np.random.default_rng(SEED)

    best_f1, best_state, no_improve = -1.0, None, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        for xb, yb in tr_ld:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if args.mixup_alpha > 0:
                    xb, ya, yb2, lam = mixup_batch(xb, yb, args.mixup_alpha, rng)
                    logits = model(xb)
                    loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb2)
                else:
                    loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()

        model.eval(); probs = []
        with torch.no_grad():
            for xb, _ in va_ld:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    probs.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
        f1 = float(f1_score(y_val, np.concatenate(probs).argmax(1), average="macro"))
        print(f"  [ERM {tag}] ep {ep+1:>2d}/{args.epochs} val_f1={f1:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [ERM {tag}] early stop @ ep {ep+1} (best {best_f1:.4f})", flush=True)
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def compute_emb(model: CNNBiGRU, X, args, device) -> torch.Tensor:
    """Frozen-backbone embeddings (N, emb_dim) on `device`. No augmentations."""
    ds = SeqDataset(X, training=False, per_file_norm=args.per_file_norm,
                    concat_stats=args.concat_stats)
    ld = DataLoader(ds, batch_size=args.batch * 2, shuffle=False,
                    num_workers=args.n_workers, pin_memory=True)
    model.eval()
    out = []
    for xb in ld:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out.append(model.forward_features(xb).float().cpu())
    return torch.cat(out).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Phases B & C — evidential last layer on cached embeddings
# ─────────────────────────────────────────────────────────────────────────────
def sorm_train(emb_tr, y_tr, emb_dim, args, class_w, device) -> EvidentialHead:
    head = EvidentialHead(emb_dim, n_classes=N_CLASSES, hidden_dim=args.head_hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    y_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    ds = TensorDataset(emb_tr, y_t)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=True)
    cw = class_w.to(device)
    for ep in range(args.sorm_epochs):
        head.train(); lam = lambda_anneal(ep + 1, args.eta); last = {}
        for eb, yb in ld:
            opt.zero_grad(set_to_none=True)
            loss, last = sorm_loss(head(eb), yb, lam, class_weights=cw)
            loss.backward(); opt.step()
        print(f"  [SORM] ep {ep+1:>2d}/{args.sorm_epochs} "
              f"loss={last['loss']:.4f} kl={last['kl']:.3f} "
              f"u={last['u_mean']:.3f} acc={last['acc']:.3f} (λ={lam:.2f})", flush=True)
    return head


def calib_train(head: EvidentialHead, emb_calib, y_calib, args, device) -> EvidentialHead:
    theta1 = {k: v.detach().clone() for k, v in head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    y_t = torch.as_tensor(y_calib, dtype=torch.long, device=device)
    # Class-balanced sampling (paper Stage 2).
    counts = np.bincount(y_calib, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    sample_w = (1.0 / counts)[y_calib]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(y_calib), replacement=True)
    ds = TensorDataset(emb_calib, y_t)
    ld = DataLoader(ds, batch_size=args.batch, sampler=sampler)
    for ep in range(args.calib_epochs):
        head.train(); last = {}
        for eb, yb in ld:
            opt.zero_grad(set_to_none=True)
            ce, last = reweighted_ce(head(eb), yb)
            loss = ce + args.beta * anchor_penalty(head, theta1)
            loss.backward(); opt.step()
        print(f"  [CALIB] ep {ep+1:>2d}/{args.calib_epochs} "
              f"loss={last['loss']:.4f} frac_correct={last['frac_correct']:.3f} "
              f"wrong_u={last['wrong_u_mean']:.3f}", flush=True)
    return head


@torch.no_grad()
def head_probs(head: EvidentialHead, emb, device) -> np.ndarray:
    head.eval()
    return probs_from_evidence(head(emb)).cpu().numpy()


def run_pipeline(X_tr, y_tr, X_eval, y_eval, groups_tr, args, class_w, device, tag):
    """v2: strong FULL-data ERM backbone, then last-layer SORM + calibration.

    v1 starved ERM by training only on (train − calib) ≈ 64% of the data. Here
    ERM trains on (almost) the whole train fold with a small RANDOM in-fold
    holdout for early stopping. The user-disjoint calibration slice is reserved
    for Stage-2 only — held out from the evidential HEAD (the backbone may see
    those users, DFR/EA style: last-layer debiasing on head-unseen data).
    """
    # User-disjoint calibration slice — held out from the head stages.
    gss = GroupShuffleSplit(n_splits=1, test_size=args.calib_frac, random_state=SEED)
    inner_loc, calib_loc = next(gss.split(X_tr, y_tr, groups_tr))
    Xi, yi = X_tr[inner_loc], y_tr[inner_loc]
    Xc, yc = X_tr[calib_loc], y_tr[calib_loc]

    # ERM backbone on the FULL train fold; early-stop on a small random holdout.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y_tr))
    n_val = max(1, int(round(args.erm_val_frac * len(y_tr))))
    es_va, es_tr = perm[:n_val], perm[n_val:]
    print(f"  [{tag}] ERM train={len(es_tr)} (full fold) es_val={len(es_va)} | "
          f"head: inner={len(yi)} calib={len(yc)} (user-disjoint)", flush=True)
    backbone = erm_train(X_tr[es_tr], y_tr[es_tr], X_tr[es_va], y_tr[es_va],
                         args, class_w, device, tag)

    emb_dim = backbone.emb_dim
    emb_inner = compute_emb(backbone, Xi, args, device)
    emb_calib = compute_emb(backbone, Xc, args, device)
    emb_eval = compute_emb(backbone, X_eval, args, device) if X_eval is not None else None

    head = sorm_train(emb_inner, yi, emb_dim, args, class_w, device)
    head = calib_train(head, emb_calib, yc, args, device)

    eval_probs = head_probs(head, emb_eval, device) if emb_eval is not None else None
    return eval_probs, backbone, head


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=40, help="ERM epochs")
    p.add_argument("--sorm-epochs", type=int, default=15)
    p.add_argument("--calib-epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="ERM lr")
    p.add_argument("--head-lr", type=float, default=1e-3, help="evidential head lr")
    p.add_argument("--head-hidden", type=int, default=None,
                   help="None=pure last-layer (paper-faithful); int=small MLP head")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--eta", type=int, default=10, help="KL anneal horizon (epochs)")
    p.add_argument("--beta", type=float, default=1.0, help="Stage-2 anchor strength")
    p.add_argument("--calib-frac", type=float, default=0.2, help="fraction of TRAIN users for calib")
    p.add_argument("--erm-val-frac", type=float, default=0.1,
                   help="random in-fold holdout for ERM early stopping (v2 full-data backbone)")
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
    print(f"Run: gru_evidential_{args.name}  device={device} "
          f"(cuda={torch.cuda.is_available()})", flush=True)

    import pandas as pd
    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    print(f"Xtr {Xtr.shape}  ytr {ytr.shape}  Xte {Xte.shape}", flush=True)
    groups = pd.read_parquet(ROOT / "data" / "meta_train.parquet")["user_id"].values
    folds = make_folds(groups, n_splits=5)
    class_w = make_class_weights(ytr)
    print(f"Class weights: {class_w.numpy().round(3).tolist()}", flush=True)

    name = f"gru_evidential_{args.name}"
    run_dir(name)
    oof_probs = np.zeros((len(ytr), N_CLASSES), dtype=np.float32)
    fold_f1s = []
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    for k, (tr_idx, va_idx) in enumerate(folds):
        cached = load_fold(name, k)
        if cached is not None:
            preds_k, probs_k = cached
            oof_probs[va_idx] = probs_k
            f1k = float(f1_score(ytr[va_idx], probs_k.argmax(1), average="macro"))
            fold_f1s.append(f1k)
            print(f"\nFold {k}: F1={f1k:.4f} (resumed)", flush=True)
            continue
        print(f"\n=== Fold {k}: train={len(tr_idx)} val={len(va_idx)} ===", flush=True)
        va_probs, _, _ = run_pipeline(
            Xtr[tr_idx], ytr[tr_idx], Xtr[va_idx], ytr[va_idx],
            groups[tr_idx], args, class_w, device, tag=f"fold{k}")
        oof_probs[va_idx] = va_probs
        save_fold(name, k, va_probs.argmax(1), va_probs)
        f1k = float(f1_score(ytr[va_idx], va_probs.argmax(1), average="macro"))
        fold_f1s.append(f1k)
        print(f"Fold {k}: F1-macro = {f1k:.4f}", flush=True)

    oof_macro = float(f1_score(ytr, oof_probs.argmax(1), average="macro"))
    per_class = f1_score(ytr, oof_probs.argmax(1), average=None)
    print(f"\nCV F1 (fold mean): {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)
    print(f"OOF F1-macro: {oof_macro:.4f}", flush=True)
    print("Per-class OOF F1: " + " ".join(f"c{c}={f:.4f}" for c, f in enumerate(per_class)), flush=True)
    print(classification_report(ytr, oof_probs.argmax(1), digits=4), flush=True)
    (ROOT / "oof").mkdir(exist_ok=True)
    np.save(ROOT / "oof" / f"{name}_oof.npy", oof_probs)

    # ── Final model on ALL train users → test prediction ──
    print("\n=== Final model: all train users → test ===", flush=True)
    _, backbone, head = run_pipeline(Xtr, ytr, None, None, groups, args, class_w, device, tag="final")
    emb_test = compute_emb(backbone, Xte, args, device)
    test_probs = head_probs(head, emb_test, device).astype(np.float32)
    np.save(ROOT / "oof" / f"{name}_test_probs.npy", test_probs)
    sub_path = ROOT / "submissions" / f"sub_{name}.csv"
    to_submission(test_ids, test_probs.argmax(1), str(sub_path))

    with open(ROOT / "submissions" / "log.md", "a", encoding="utf-8") as f:
        f.write(f"| {date.today().isoformat()} | sub_{name} | "
                f"CNN-BiGRU + Evidential Alignment (KDD25) | "
                f"{np.mean(fold_f1s):.4f} (fold-mean) / {oof_macro:.4f} (OOF) | _pending_ | _pending_ | "
                f"per-class F1 {[round(float(x), 4) for x in per_class]} |\n")
    with open(ROOT / "oof" / f"{name}_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"model": name, "oof_f1_macro": oof_macro,
                   "per_class_f1": [float(x) for x in per_class],
                   "fold_f1s": fold_f1s, "args": vars(args), "seed": SEED}, fh, indent=2)
    print(f"\nDone. OOF F1={oof_macro:.4f}  submission={sub_path}", flush=True)


if __name__ == "__main__":
    main()
