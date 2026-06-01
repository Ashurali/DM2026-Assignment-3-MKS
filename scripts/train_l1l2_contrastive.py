"""Train a triplet-loss MLP head on top of CNN-BiLSTM v1's 256-d embedding,
specifically targeting L1↔L2 discrimination.

Why this is genuinely different from past attempts:
  - SICL (failed earlier) used SUBJECT as the negative class. This uses
    LABEL (L1 vs L2) directly. Different objective, different inductive bias.
  - Pair specialists (failed earlier) trained binary classifiers; this trains
    an EMBEDDING that the downstream stacker can use as features.
  - Hard negative mining: each L2 anchor pairs with its NEAREST L1 sample
    (in the current MLP's embedding space) — focuses learning on the actual
    confusion zone, not random easy negatives.

Per-fold protocol (matches OOF protocol of all other base models):
  For fold k:
    Train MLP on tr_k using triplet sampling from {L1 ∪ L2} ∩ tr_k.
    Embed val_k samples (all classes) → OOF embeddings.
    Embed test → contributes 1/5 to averaged test embedding.

Saves:
  oof/l1l2_contrast_emb_train.npy    (N_train, 64)  per-fold OOF embeddings
  oof/l1l2_contrast_emb_test.npy     (N_test, 64)   averaged across 5 fold-MLPs

Usage:
  python scripts/train_l1l2_contrastive.py --gpu --epochs 200 --margin 0.4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

print("=== train_l1l2_contrastive.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.triplet_mlp import TripletMLP
from src.utils.cv import make_folds

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hard-neg-pool", type=int, default=512,
                   help="Pool size from which to pick hard negatives.")
    return p.parse_args()


def hard_negative_indices(model: nn.Module, anchors: torch.Tensor, neg_pool: torch.Tensor,
                          device: torch.device) -> torch.Tensor:
    """For each anchor, find index of nearest pool entry (hardest negative)."""
    model.eval()
    with torch.no_grad():
        a_emb = model(anchors.to(device))
        n_emb = model(neg_pool.to(device))
        # Use angular distance on L2-normalised emb (1 - cos_sim)
        dist = 1.0 - a_emb @ n_emb.T   # (N_a, N_n)
        hard_idx = dist.argmin(dim=1)
    return hard_idx


def train_fold(emb_train_fold: np.ndarray, y_train_fold: np.ndarray,
               args, device: torch.device) -> nn.Module:
    """Train MLP for one fold using triplet loss with hard-negative mining.

    Triplet sampling per batch:
      - Anchors are L2 samples (with replacement so we get B anchors per batch).
      - Positives are different-from-anchor L2 samples (with replacement).
      - Negatives: pool of B×4 random L1 samples; hard-mined for each anchor.
    """
    in_dim = emb_train_fold.shape[1]
    model = TripletMLP(in_dim=in_dim, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    l1_idx = np.where(y_train_fold == 1)[0]
    l2_idx = np.where(y_train_fold == 2)[0]
    print(f"    fold sizes: L1={len(l1_idx)}  L2={len(l2_idx)}", flush=True)

    if len(l2_idx) < 2:
        print("    !! too few L2 samples in this fold to do triplet sampling", flush=True)
        return model

    rng = np.random.default_rng(0)
    emb_t = torch.from_numpy(emb_train_fold.astype(np.float32))

    n_batches_per_epoch = max(1, len(l2_idx) // args.batch)

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for _ in range(n_batches_per_epoch):
            B = min(args.batch, len(l2_idx))
            anchor_choices = rng.choice(l2_idx, size=B, replace=False if B <= len(l2_idx) else True)
            positive_choices = np.zeros(B, dtype=np.int64)
            for i, a in enumerate(anchor_choices):
                pool = l2_idx[l2_idx != a]
                positive_choices[i] = rng.choice(pool) if len(pool) > 0 else a

            # Hard negative pool: random sample of L1
            pool_size = min(args.hard_neg_pool, len(l1_idx))
            neg_pool_idx = rng.choice(l1_idx, size=pool_size, replace=False)

            anchors = emb_t[anchor_choices]
            positives = emb_t[positive_choices]
            neg_pool = emb_t[neg_pool_idx]

            # Pick the hardest negative for each anchor
            hard_idx = hard_negative_indices(model, anchors, neg_pool, device)
            negatives = neg_pool[hard_idx.cpu()]

            # Forward + loss
            model.train()
            a_emb = model(anchors.to(device))
            p_emb = model(positives.to(device))
            n_emb = model(negatives.to(device))

            d_ap = (a_emb - p_emb).pow(2).sum(dim=1)
            d_an = (a_emb - n_emb).pow(2).sum(dim=1)
            loss = F.relu(d_ap - d_an + args.margin).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched.step()
        if (ep + 1) % 25 == 0 or ep == args.epochs - 1:
            print(f"      ep {ep + 1}/{args.epochs}  triplet loss: {ep_loss / max(n_batches, 1):.4f}",
                  flush=True)

    return model


def embed_all(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Forward all rows of X through model.eval() → return (N, emb_dim) np array."""
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
            z = model(batch).cpu().numpy()
            out_chunks.append(z)
    return np.concatenate(out_chunks, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  epochs: {args.epochs}  margin: {args.margin}  emb_dim: {args.emb_dim}",
          flush=True)

    emb_train = np.load(ROOT / "oof" / "cnn_bilstm_v1_emb_train.npy").astype(np.float32)
    emb_test = np.load(ROOT / "oof" / "cnn_bilstm_v1_emb_test.npy").astype(np.float32)
    print(f"Source CNN-BiLSTM emb: train {emb_train.shape}  test {emb_test.shape}", flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)

    out_train = np.zeros((len(emb_train), args.emb_dim), dtype=np.float32)
    out_test_sum = np.zeros((len(emb_test), args.emb_dim), dtype=np.float32)

    for k, (tr, va) in enumerate(folds):
        print(f"\n  Fold {k}: tr={len(tr)}  va={len(va)}", flush=True)
        t0 = time.time()
        model = train_fold(emb_train[tr], y[tr], args, device)
        # Embed val (all classes) and test
        out_train[va] = embed_all(model, emb_train[va], device)
        out_test_sum += embed_all(model, emb_test, device)
        print(f"    fold {k} done in {(time.time() - t0):.1f}s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_test = out_test_sum / len(folds)
    # Re-normalise the averaged test embeddings (mean of L2-normalised vectors isn't unit-norm)
    norms = np.linalg.norm(out_test, axis=1, keepdims=True)
    out_test = out_test / np.clip(norms, 1e-8, None)

    np.save(ROOT / "oof" / "l1l2_contrast_emb_train.npy", out_train.astype(np.float32))
    np.save(ROOT / "oof" / "l1l2_contrast_emb_test.npy", out_test.astype(np.float32))
    print(f"\nSaved oof/l1l2_contrast_emb_train.npy {out_train.shape}", flush=True)
    print(f"Saved oof/l1l2_contrast_emb_test.npy {out_test.shape}", flush=True)

    # Sanity diagnostic: L1↔L2 separation in the new embedding (cosine distance)
    print("\n=== L1↔L2 separation diagnostic in new embedding ===", flush=True)
    from sklearn.neighbors import NearestNeighbors
    l2_emb = out_train[y == 2]
    l1_emb = out_train[y == 1]
    nn_l1 = NearestNeighbors(n_neighbors=1).fit(l1_emb)
    nn_l2 = NearestNeighbors(n_neighbors=2).fit(l2_emb)  # 2 because nearest = self
    d_l2_to_l1 = nn_l1.kneighbors(l2_emb)[0][:, 0]
    d_l2_to_l2 = nn_l2.kneighbors(l2_emb)[0][:, 1]
    closer_to_l1 = (d_l2_to_l1 < d_l2_to_l2).mean() * 100
    print(f"  L2 samples closer to nearest L1 than nearest L2: {closer_to_l1:.1f}%")
    print(f"  median d(L2 → L1) = {np.median(d_l2_to_l1):.3f}")
    print(f"  median d(L2 → L2) = {np.median(d_l2_to_l2):.3f}")
    print(f"  Reference (CNN-BiLSTM emb): 62% of L2 neighbours were L1.")
    print(f"  Goal: bring this percentage way down → contrastive learning worked.")
    print("\n=== train_l1l2_contrastive.py done ===", flush=True)


if __name__ == "__main__":
    main()
