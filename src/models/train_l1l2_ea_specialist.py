"""Binary CNN-BiGRU + Evidential Alignment specialist for the L1<->L2 bottleneck.

Trains ONLY on the walking subset (true label in {1,2}, relabeled 0=L1, 1=L2):
  per outer GroupKFold(5)-by-user fold:
    - carve a user-disjoint calibration slice from train
    - ERM-train CNN-BiGRU(2-class) on the full train fold (class-weighted CE,
      augs, AMP, cosine, early-stop on a random in-fold holdout)
    - freeze backbone; Stage-1 SORM evidential head on train embeddings
    - Stage-2 uncertainty-weighted calibration on the calib slice (class-bal.)
    - predict E[p] on the held-out fold  -> OOF P(L2 | walking)

Reports binary macro-F1 + L2 recall vs existing sources (gru 0.61/0.59,
cnn 0.62/0.63, combo 0.68/0.28, P2 0.58/0.12). Saves the OOF so it can later
feed a rebuilt Fine_walk stage. EVAL caveat: judge integration under FROZEN
thresholds, never re-tuned (the EA-P2 lesson).

Usage (server, dm2026-a3):
    python -m src.models.train_l1l2_ea_specialist --gpu --name v1 --per-file-norm --concat-stats
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch required.", file=sys.stderr)
    raise

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from src.utils.cv import make_folds
from src.models.cnn_bigru import CNNBiGRU, SeqDataset, worker_init_fn
from src.models.train_cnn_bilstm import build_or_load_seq_cache
from src.utils.evidential_align import (
    EvidentialHead, sorm_loss, reweighted_ce, anchor_penalty,
    probs_from_evidence, lambda_anneal,
)

ROOT = Path(__file__).resolve().parents[2]
NC = 2
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--sorm-epochs", type=int, default=15)
    p.add_argument("--calib-epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--eta", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--erm-val-frac", type=float, default=0.12)
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--concat-stats", action="store_true")
    return p.parse_args()


def class_weights(yb):
    cnt = np.bincount(yb, minlength=NC).astype(float); cnt[cnt == 0] = 1
    return torch.tensor(len(yb) / (NC * cnt), dtype=torch.float32)


def erm_train(Xtr, ytr, Xva, yva, args, cw, device, tag):
    aug = {"p_rot": args.p_rot, "p_jitter": args.p_jitter, "p_scale": args.p_scale, "p_warp": args.p_warp}
    trl = DataLoader(SeqDataset(Xtr, ytr, training=True, seed=SEED, aug_probs=aug,
                                per_file_norm=args.per_file_norm, concat_stats=args.concat_stats),
                     batch_size=args.batch, shuffle=True, num_workers=args.n_workers,
                     worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
    val = DataLoader(SeqDataset(Xva, yva, training=False, per_file_norm=args.per_file_norm, concat_stats=args.concat_stats),
                     batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
    m = CNNBiGRU(n_classes=NC, in_channels=in_ch, gru_hidden=args.gru_hidden).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    best, best_state, no_imp = -1.0, None, 0
    for ep in range(args.epochs):
        m.train()
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(m(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        m.eval(); pr = []
        with torch.no_grad():
            for xb, _ in val:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pr.append(torch.softmax(m(xb).float(), 1).cpu().numpy())
        f1 = f1_score(yva, np.concatenate(pr).argmax(1), average="macro")
        if f1 > best:
            best, no_imp, best_state = f1, 0, copy.deepcopy(m.state_dict())
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    m.load_state_dict(best_state)
    print(f"  [ERM {tag}] best binary val_f1={best:.4f}", flush=True)
    return m


@torch.no_grad()
def get_emb(m, X, args, device):
    ld = DataLoader(SeqDataset(X, training=False, per_file_norm=args.per_file_norm, concat_stats=args.concat_stats),
                    batch_size=args.batch * 2, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    m.eval(); out = []
    for xb in ld:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out.append(m.forward_features(xb).float().cpu())
    return torch.cat(out).to(device)


def sorm_then_calib(emb_inner, y_inner, emb_calib, y_calib, emb_dim, args, cw, device):
    head = EvidentialHead(emb_dim, n_classes=NC, hidden_dim=32).to(device)
    yt = torch.as_tensor(y_inner, dtype=torch.long, device=device)
    ld = DataLoader(TensorDataset(emb_inner, yt), batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    for ep in range(args.sorm_epochs):
        lam = lambda_anneal(ep + 1, args.eta)
        for eb, yb in ld:
            opt.zero_grad(); loss, _ = sorm_loss(head(eb), yb, lam, class_weights=cw.to(device)); loss.backward(); opt.step()
    theta1 = {k: v.detach().clone() for k, v in head.named_parameters()}
    yc = torch.as_tensor(y_calib, dtype=torch.long, device=device)
    cc = np.bincount(y_calib, minlength=NC).astype(float); cc[cc == 0] = 1
    sw = (1.0 / cc)[y_calib]
    samp = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(y_calib), replacement=True)
    ldc = DataLoader(TensorDataset(emb_calib, yc), batch_size=args.batch, sampler=samp)
    opt2 = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    for ep in range(args.calib_epochs):
        for eb, yb in ldc:
            opt2.zero_grad(); ce, _ = reweighted_ce(head(eb), yb)
            loss = ce + args.beta * anchor_penalty(head, theta1); loss.backward(); opt2.step()
    return head


def run_fold(Xtr, ytr, gtr, Xeval, args, cw, device, tag):
    gss = GroupShuffleSplit(1, test_size=args.calib_frac, random_state=SEED)
    inner, calib = next(gss.split(Xtr, ytr, gtr))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ytr)); nv = max(1, int(round(args.erm_val_frac * len(ytr))))
    es_va, es_tr = perm[:nv], perm[nv:]
    backbone = erm_train(Xtr[es_tr], ytr[es_tr], Xtr[es_va], ytr[es_va], args, cw, device, tag)
    emb_dim = backbone.emb_dim
    e_inner = get_emb(backbone, Xtr[inner], args, device)
    e_calib = get_emb(backbone, Xtr[calib], args, device)
    e_eval = get_emb(backbone, Xeval, args, device)
    head = sorm_then_calib(e_inner, ytr[inner], e_calib, ytr[calib], emb_dim, args, cw, device)
    head.eval()
    with torch.no_grad():
        return probs_from_evidence(head(e_eval)).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"L1<->L2 EA specialist  device={device}", flush=True)
    X, y, _, _ = build_or_load_seq_cache()
    groups = pd.read_parquet(ROOT / "data" / "meta_train.parquet")["user_id"].values
    walk = np.where((y == 1) | (y == 2))[0]
    Xw, yb, gw = X[walk], (y[walk] == 2).astype(np.int64), groups[walk]
    print(f"walking subset: {len(walk)} (L1={int((yb==0).sum())}, L2={int((yb==1).sum())})", flush=True)
    cw = class_weights(yb)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    oof = np.zeros((len(walk), NC), dtype=np.float32)
    for k, (tr, va) in enumerate(make_folds(gw, n_splits=5)):
        print(f"\n=== Fold {k}: train={len(tr)} val={len(va)} ===", flush=True)
        oof[va] = run_fold(Xw[tr], yb[tr], gw[tr], Xw[va], args, cw, device, f"f{k}")
        pred = oof[va].argmax(1)
        print(f"  fold {k}: L1vL2 macroF1={f1_score(yb[va], pred, average='macro'):.4f}  "
              f"L2recall={float((pred[yb[va]==1]==1).mean()):.3f}", flush=True)

    pred = oof.argmax(1)
    f1m = float(f1_score(yb, pred, average="macro"))
    l2_rec = float((pred[yb == 1] == 1).mean()); l1_rec = float((pred[yb == 0] == 0).mean())
    print(f"\n=== OOF L1<->L2: macroF1={f1m:.4f}  L2recall={l2_rec:.3f}  L1recall={l1_rec:.3f} ===", flush=True)
    print(f"   (refs: gru 0.608/0.589, cnn 0.620/0.626, combo 0.677/0.282, P2 0.577/0.115)", flush=True)
    (ROOT / "oof").mkdir(exist_ok=True)
    np.save(ROOT / "oof" / f"l1l2_ea_{args.name}_oof.npy", oof)
    json.dump({"walk_idx": walk.tolist(), "macro_f1": f1m, "l2_recall": l2_rec,
               "l1_recall": l1_rec, "args": vars(args)},
              open(ROOT / "oof" / f"l1l2_ea_{args.name}_meta.json", "w"), indent=2)
    print(f"Saved oof/l1l2_ea_{args.name}_oof.npy", flush=True)


if __name__ == "__main__":
    main()
