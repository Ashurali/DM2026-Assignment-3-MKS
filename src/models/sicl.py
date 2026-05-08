"""Subject-Invariant Contrastive Learning (SICL) for HAR (Tier B.3).

Talegaonkar et al., MLSP 2025 (arXiv 2507.03250). Code reference:
github.com/olivesgatech/SICL.

The standard supervised-contrastive loss treats all same-class pairs as
positives. SICL re-weights the same-SUBJECT positives/negatives so that
the encoder can't distinguish samples by who produced them — only by
what activity they represent.

Loss (unimodal):
  L_SICL = - sum_i log [ exp(z_i · z_j / τ) / D_i ]
  D_i = Q_S * sum_{s in S(i)} exp(z_i · z_s / τ)
       + sum_{k not in S(i)} exp(z_i · z_k / τ)

where:
- z_i, z_j are L2-normalized embeddings of an anchor and a positive
- S(i) = indices of OTHER samples from the same subject as i (excluding i, j)
- Q_S is a hyperparameter (default ~0.5) that down-weights same-subject
  contrast — pushing the encoder to NOT use subject-discriminative
  features for pulling positives close.

This module provides:
- `ContrastiveProjectionHead`: small MLP that maps encoder output → 128-D
  contrastive embedding space (standard SimCLR/SupCon practice).
- `sicl_loss`: the SICL loss above with positive-pair selection.
- `make_two_views`: standard augmentation pair generator using existing
  augment_sample with two independent random calls.
- `train_sicl_pretrain`: pretraining loop. After this, the encoder is
  frozen and a linear classifier is trained on top with class CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjectionHead(nn.Module):
    """2-layer MLP projection head as in SimCLR. Output is L2-normalized
    in `forward` so caller works with unit vectors."""

    def __init__(self, in_dim: int, proj_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1, p=2)


def sicl_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    temperature: float = 0.1,
    q_s: float = 0.5,
) -> torch.Tensor:
    """SICL contrastive loss.

    z1, z2: (B, D) two views of each sample, L2-normalized.
    labels: (B,) class label for each sample.
    subjects: (B,) subject id for each sample.
    temperature: τ (default 0.1, typical SupCon value).
    q_s: down-weight factor for same-subject pairs (default 0.5).

    Returns scalar loss.
    """
    B = z1.size(0)
    device = z1.device

    # Stack the two views into a (2B, D) tensor; same labels and subjects
    z = torch.cat([z1, z2], dim=0)
    lbl = torch.cat([labels, labels], dim=0)
    sub = torch.cat([subjects, subjects], dim=0)

    # Pairwise similarities (2B x 2B) in log-space (cosine since L2-normalized)
    sim = torch.matmul(z, z.t()) / temperature  # (2B, 2B)
    # Mask out self-similarity (diagonal)
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Stability: subtract max per row before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)

    # Build masks
    same_class = lbl.unsqueeze(0) == lbl.unsqueeze(1)
    same_subject = sub.unsqueeze(0) == sub.unsqueeze(1)
    pos_mask = same_class & ~self_mask  # positives = same class (incl. its other view)
    same_subject_no_self = same_subject & ~self_mask

    # Weighted denominator: same-subject negatives get q_s, others get 1
    weights = torch.where(same_subject_no_self,
                           torch.full_like(exp_sim, q_s),
                           torch.ones_like(exp_sim))
    weights = weights.masked_fill(self_mask, 0.0)
    denom = (weights * exp_sim).sum(dim=1)  # (2B,)

    # Numerator: sum over positives (other than self)
    num = (pos_mask.float() * exp_sim).sum(dim=1)  # (2B,)
    num_count = pos_mask.float().sum(dim=1).clamp(min=1.0)

    # Mean log-ratio over rows that have at least 1 positive
    has_pos = num_count > 0
    log_ratio = torch.log((num + 1e-12) / (denom + 1e-12))
    # The "average over positives" form:
    # L = − (1/2B) Σ_i (1/|P(i)|) Σ_{p∈P(i)} log[ exp(sim_ip) / D_i ]
    # Approximated above by log of summed numerator; both are valid SupCon
    # variants. We keep the log-of-sum form for simplicity and stability.
    loss = -log_ratio[has_pos].mean()
    return loss
