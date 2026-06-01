"""Tilt-trajectory L1<->L2 model — the representation-layer attack on separation.

The wrist-ADL discriminator is the TIME-RESOLVED orientation motion shape (a drink
= one tilt arc; brushing = many oscillations). Production uses orientation only as
STATIC/summary features; the raw-6-channel CNN-BiGRU failed because the gravity
magnitude/offset noise drowns the motion. So: derive a CLEAN, low-dim, per-file-
normalised orientation trajectory and feed it to a small 1D-CNN.

Derived channels (per second, then per-file z-scored -> shape, not offset):
  pitch(t), roll(t), inclination-vs-window-mean(t),  angular-speed(t) [pseudo-gyro],
  intensity(t)=||std||,  gravity-magnitude(t).

6-class (GroupKFold-5 by user, class-weighted). We use its L2 column as an
injectable complementary L2 source (same path as orient_lgbm). Saves OOF/test.

Usage (server, dm2026-a3):
  python -m src.models.train_orient_traj --gpu --name v1
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
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: PyTorch required.", file=sys.stderr)
    raise
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.train_cnn_bilstm import build_or_load_seq_cache

NC = 6
SEED = 42
EPS = 1e-8


def derive_traj(X):
    """(N,6,300) raw [mean_xyz, std_xyz] -> (N,6,300) clean orientation trajectory,
    per-file z-scored (removes mounting offset, keeps motion shape)."""
    N = X.shape[0]
    g = X[:, 0:3, :].astype(np.float64)            # gravity/orientation
    s = X[:, 3:6, :].astype(np.float64)            # within-second intensity
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    gn = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + EPS
    pitch = np.arctan2(gx, np.sqrt(gy ** 2 + gz ** 2) + EPS)
    roll = np.arctan2(gy, gz)
    gu = g / gn[:, None, :]                         # (N,3,300) unit orientation
    gbar = g.mean(axis=2, keepdims=True)
    gbar = gbar / (np.linalg.norm(gbar, axis=1, keepdims=True) + EPS)
    incl = np.arccos(np.clip(np.sum(gu * gbar, axis=1), -1, 1))         # (N,300)
    cosd = np.clip(np.sum(gu[:, :, 1:] * gu[:, :, :-1], axis=1), -1, 1)  # (N,299)
    angspeed = np.concatenate([np.zeros((N, 1)), np.arccos(cosd)], axis=1)
    intensity = np.linalg.norm(s, axis=1)           # (N,300)
    gmag = gn                                       # (N,300)
    T = np.stack([pitch, roll, incl, angspeed, intensity, gmag], axis=1)  # (N,6,300)
    mu = T.mean(axis=2, keepdims=True)
    sd = T.std(axis=2, keepdims=True) + EPS
    return ((T - mu) / sd).astype(np.float32)


class OrientTrajCNN(nn.Module):
    def __init__(self, in_ch=6, n_classes=NC, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch)
        self.c1 = nn.Sequential(nn.Conv1d(in_ch, 64, 7, padding=3), nn.ReLU(inplace=True),
                                nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.c2 = nn.Sequential(nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
                                nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.c3 = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool1d(1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.bn(x); x = self.c1(x); x = self.c2(x); x = self.c3(x).squeeze(-1)
        return self.fc(self.drop(x))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--jitter", type=float, default=0.05, help="train-time gaussian jitter std")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def class_weights(y, device):
    cnt = np.bincount(y, minlength=NC).astype(float); cnt[cnt == 0] = 1
    return torch.tensor((len(y) / (NC * cnt)), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, T, args, device):
    ld = DataLoader(TensorDataset(torch.from_numpy(T)), batch_size=args.batch * 2, shuffle=False)
    model.eval(); out = []
    for (xb,) in ld:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train_fold(Ttr, ytr, args, device, tag):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ytr)); nv = max(args.batch, int(0.12 * len(ytr)))
    va, tr = perm[:nv], perm[nv:]
    Xtr = torch.from_numpy(Ttr[tr]); Ytr = torch.tensor(ytr[tr], dtype=torch.long)
    trl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True, drop_last=True)
    model = OrientTrajCNN(in_ch=Ttr.shape[1], dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 3 if args.smoke else args.epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr[tr], device))
    best, best_state, no_imp = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in trl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            if args.jitter > 0:
                xb = xb + torch.randn_like(xb) * args.jitter
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        vp = predict(model, Ttr[va], args, device).argmax(1)
        f1 = f1_score(ytr[va], vp, average="macro")
        if f1 > best:
            best, no_imp, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    model.load_state_dict(best_state)
    print(f"  [{tag}] best in-fold val macroF1={best:.4f}", flush=True)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    groups = pd.read_parquet(ROOT / "data" / "meta_train.parquet")["user_id"].values
    print(f"deriving trajectory ...  device={device}", flush=True)
    Ttr, Tte = derive_traj(Xtr), derive_traj(Xte)
    print(f"trajectory shape: {Ttr.shape}  channels=[pitch,roll,incl,angspeed,intensity,gmag]", flush=True)

    oof = np.zeros((len(ytr), NC), dtype=np.float32)
    test = np.zeros((len(Tte), NC), dtype=np.float32)
    for k, (tr, va) in enumerate(GroupKFold(5).split(Ttr, ytr, groups)):
        m = train_fold(Ttr[tr], ytr[tr], args, device, f"f{k}")
        oof[va] = predict(m, Ttr[va], args, device)
        test += predict(m, Tte, args, device) / 5
        vp = oof[va].argmax(1)
        pc = f1_score(ytr[va], vp, average=None, labels=list(range(NC)))
        print(f"  fold {k}: macroF1={f1_score(ytr[va], vp, average='macro'):.4f}  "
              f"L2={pc[2]:.3f} L3={pc[3]:.3f} L5={pc[5]:.3f}", flush=True)
        if args.smoke:
            print("SMOKE done"); return

    pred = oof.argmax(1)
    f1m = float(f1_score(ytr, pred, average="macro"))
    P, R, F, S = precision_recall_fscore_support(ytr, pred, labels=list(range(NC)))
    wm = (ytr == 1) | (ytr == 2)
    pb = oof[wm][:, [1, 2]]; pb = pb / np.clip(pb.sum(1, keepdims=True), 1e-12, None)
    l1l2 = f1_score((ytr[wm] == 2).astype(int), pb.argmax(1), average="macro")
    print(f"\n=== OOF macroF1={f1m:.4f}  L1vL2sep={l1l2:.4f} ===", flush=True)
    for i in range(NC):
        print(f"  L{i}: prec={P[i]:.3f} rec={R[i]:.3f} f1={F[i]:.3f}", flush=True)
    print("  (refs: orient_lgbm L1vL2sep~0.62; production L2=0.384; summary-orient injection->0.8200 LB)", flush=True)

    (ROOT / "oof").mkdir(exist_ok=True)
    np.save(ROOT / "oof" / f"orient_traj_{args.name}_oof.npy", oof)
    np.save(ROOT / "oof" / f"orient_traj_{args.name}_test_probs.npy", test)
    json.dump({"oof_macro_f1": f1m, "l1l2_sep": float(l1l2),
               "per_class_f1": [float(x) for x in F], "args": vars(args)},
              open(ROOT / "oof" / f"orient_traj_{args.name}_meta.json", "w"), indent=2)
    print(f"\nSaved oof/orient_traj_{args.name}_oof.npy + _test_probs.npy", flush=True)


if __name__ == "__main__":
    main()
