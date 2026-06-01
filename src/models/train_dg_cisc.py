"""CISC — Conditional Invariant Supervised-Contrastive CNN-BiGRU (DG model).

Goal (per reports/findings_and_solution.md): learn class-discriminative features
for the under-served classes {L2, L3, L5} that are INVARIANT to the user, by
combining standard class-weighted CE with a *conditional* supervised-contrastive
loss that pulls same-class-DIFFERENT-user embeddings together (subject-invariance
within class — the fix for why SICL/DANN failed).

Protocol (matches the project's disjoint-user conventions):
  - backbone: CNNBiGRU (per-file-norm + concat-stats -> 8 input channels), the
    config that rescued L4 and gave the best standalone GRU.
  - GroupKFold(5) by user_id -> clean OOF; full-train model -> test probs.
  - per fold: random in-fold holdout for early-stopping on macro-F1 (OOF stays
    user-disjoint and clean).
  - minority-oversampling WeightedRandomSampler so L2/L3/L5 appear with positives.

This produces an OOF + test-prob source ONLY. Integration into production is done
separately (gated meta-learner under FROZEN top-1 thresholds — never re-tuned).

Usage (server, dm2026-a3):
  python -m src.models.train_dg_cisc --gpu --name v1 --per-file-norm --concat-stats
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
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch required.", file=sys.stderr)
    raise

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.cnn_bigru import CNNBiGRU, SeqDataset, worker_init_fn
from src.models.train_cnn_bilstm import build_or_load_seq_cache
from src.utils.cond_supcon import conditional_supcon_loss

NC = 6
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler-power", type=float, default=1.0,
                   help="oversampling strength: 1.0=inverse-freq, 0.5=sqrt (milder, better precision), 0=uniform")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lambda-supcon", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--gamma-cross", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--es-frac", type=float, default=0.12)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--p-rot", type=float, default=0.5)
    p.add_argument("--p-jitter", type=float, default=0.5)
    p.add_argument("--p-scale", type=float, default=0.3)
    p.add_argument("--p-warp", type=float, default=0.3)
    p.add_argument("--per-file-norm", action="store_true")
    p.add_argument("--concat-stats", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 fold, few epochs")
    return p.parse_args()


class DGModel(nn.Module):
    """CNN-BiGRU backbone + projection head for contrastive learning."""

    def __init__(self, in_ch, gru_hidden, n_classes=NC, proj_dim=128, dropout=0.3):
        super().__init__()
        self.backbone = CNNBiGRU(n_classes=n_classes, in_channels=in_ch,
                                 gru_hidden=gru_hidden, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.emb_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim),
        )

    def forward(self, x):
        emb = self.backbone.forward_features(x)
        logits = self.backbone.fc(self.backbone.dropout(emb))
        return logits, emb


class SeqDatasetGroup(Dataset):
    """Wrap SeqDataset so each item also returns its integer user/group id.

    worker_init_fn (in cnn_bilstm) already re-seeds `inner._rng`, so augmentation
    randomness stays per-worker correct through this wrapper.
    """

    def __init__(self, inner: SeqDataset, groups: np.ndarray):
        self.inner = inner
        self.groups = np.asarray(groups).astype(np.int64)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        out = self.inner[i]
        g = int(self.groups[i])
        if isinstance(out, tuple):
            x, y = out
            return x, y, g
        return out, g


def class_weights(y):
    cnt = np.bincount(y, minlength=NC).astype(float)
    cnt[cnt == 0] = 1.0
    inv = len(y) / (NC * cnt)
    return torch.tensor(inv, dtype=torch.float32)


def sampler_for(y, power=1.0):
    """WeightedRandomSampler so minorities get in-batch positives.

    power=1.0 -> inverse-frequency (aggressive); 0.5 -> sqrt-inverse (milder,
    less minority over-prediction -> better precision); 0 -> uniform.
    """
    cnt = np.bincount(y, minlength=NC).astype(float)
    cnt[cnt == 0] = 1.0
    w = (1.0 / np.power(cnt, power))[y]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y), replacement=True)


def make_loader(X, y, g, args, training):
    aug = {"p_rot": args.p_rot, "p_jitter": args.p_jitter,
           "p_scale": args.p_scale, "p_warp": args.p_warp}
    inner = SeqDataset(X, y, training=training, seed=args.seed, aug_probs=aug,
                       per_file_norm=args.per_file_norm, concat_stats=args.concat_stats)
    ds = SeqDatasetGroup(inner, g)
    if training:
        return DataLoader(ds, batch_size=args.batch, sampler=sampler_for(y, args.sampler_power),
                          num_workers=args.n_workers, worker_init_fn=worker_init_fn,
                          pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=args.batch * 2, shuffle=False,
                      num_workers=args.n_workers, pin_memory=True)


@torch.no_grad()
def predict_probs(model, X, g, args, device):
    ld = make_loader(X, None, g, args, training=False)
    model.eval()
    out = []
    for batch in ld:
        xb = batch[0].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(xb)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def train_model(Xtr, ytr, gtr, args, device, tag):
    # in-fold random holdout for early stopping (does NOT touch the OOF fold)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ytr))
    nv = max(args.batch, int(round(args.es_frac * len(ytr))))
    va_idx, tr_idx = perm[:nv], perm[nv:]
    trl = make_loader(Xtr[tr_idx], ytr[tr_idx], gtr[tr_idx], args, training=True)
    val_X, val_y, val_g = Xtr[va_idx], ytr[va_idx], gtr[va_idx]

    in_ch = 8 if (args.per_file_norm and args.concat_stats) else 6
    model = DGModel(in_ch, args.gru_hidden, NC, args.proj_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 3 if args.smoke else args.epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))

    best, best_state, no_imp = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, gb in trl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            gb = gb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, emb = model(xb)
                ce = crit(logits, yb)
            z = model.proj(emb.float())
            sc = conditional_supcon_loss(z, yb, gb, tau=args.tau, gamma_cross=args.gamma_cross)
            loss = ce + args.lambda_supcon * sc
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sch.step()
        # validation macro-F1
        vp = predict_probs(model, val_X, val_g, args, device).argmax(1)
        f1 = f1_score(val_y, vp, average="macro")
        if f1 > best:
            best, no_imp, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{tag}] best in-fold val macro-F1={best:.4f}", flush=True)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"CISC DG model  device={device}  lambda={args.lambda_supcon} "
          f"gamma_cross={args.gamma_cross} tau={args.tau} batch={args.batch}", flush=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = pd.factorize(meta["user_id"].values)[0].astype(np.int64)
    assert len(groups) == len(ytr), (len(groups), len(ytr))
    print(f"train={len(ytr)}  test={len(Xte)}  users={groups.max() + 1}  "
          f"class_counts={np.bincount(ytr, minlength=NC).tolist()}", flush=True)

    oof = np.zeros((len(ytr), NC), dtype=np.float32)
    n_splits = 5
    for k, (tr, va) in enumerate(GroupKFold(n_splits).split(Xtr, ytr, groups)):
        print(f"\n=== Fold {k}: train={len(tr)} val={len(va)} "
              f"(val users={np.unique(groups[va]).size}) ===", flush=True)
        model = train_model(Xtr[tr], ytr[tr], groups[tr], args, device, f"f{k}")
        oof[va] = predict_probs(model, Xtr[va], groups[va], args, device)
        vp = oof[va].argmax(1)
        pc = f1_score(ytr[va], vp, average=None, labels=list(range(NC)))
        print(f"  fold {k}: macroF1={f1_score(ytr[va], vp, average='macro'):.4f}  "
              f"L2={pc[2]:.3f} L3={pc[3]:.3f} L5={pc[5]:.3f}", flush=True)
        if args.smoke:
            print("SMOKE: stopping after fold 0", flush=True)
            return

    pred = oof.argmax(1)
    f1m = float(f1_score(ytr, pred, average="macro"))
    P, R, F, S = precision_recall_fscore_support(ytr, pred, labels=list(range(NC)))
    print(f"\n=== OOF macro-F1 = {f1m:.4f} ===", flush=True)
    print("  class  prec   rec    f1     support", flush=True)
    for i in range(NC):
        print(f"  L{i}   {P[i]:.3f}  {R[i]:.3f}  {F[i]:.3f}  {int(S[i])}", flush=True)
    print("  (production peak OOF per-class F1: L2=.384 L3=.764 L5=.781; "
          "best GRU L2~.23)", flush=True)

    # full-train model -> test probs
    print("\nTraining full model on all train -> test probs ...", flush=True)
    full = train_model(Xtr, ytr, groups, args, device, "full")
    test_probs = predict_probs(full, Xte, np.zeros(len(Xte), dtype=np.int64), args, device)

    (ROOT / "oof").mkdir(exist_ok=True)
    np.save(ROOT / "oof" / f"dg_cisc_{args.name}_oof.npy", oof)
    np.save(ROOT / "oof" / f"dg_cisc_{args.name}_test_probs.npy", test_probs)
    json.dump(
        {"oof_macro_f1": f1m,
         "per_class_f1": [float(x) for x in F],
         "per_class_recall": [float(x) for x in R],
         "args": vars(args)},
        open(ROOT / "oof" / f"dg_cisc_{args.name}_meta.json", "w"), indent=2)
    print(f"\nSaved oof/dg_cisc_{args.name}_oof.npy + _test_probs.npy + _meta.json", flush=True)


if __name__ == "__main__":
    main()
