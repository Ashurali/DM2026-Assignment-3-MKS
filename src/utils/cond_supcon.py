"""Conditional supervised-contrastive loss for subject-invariant HAR.

Standard SupCon (Khosla et al., 2020) pulls together same-class embeddings and
pushes apart different-class ones. The *conditional* variant here additionally
**up-weights positive pairs drawn from DIFFERENT users (subjects)**. That makes
the representation invariant to the subject *within each class* — exactly the
thing this project needs: learn the L2/L3/L5 activity characteristics while
removing the user-spurious attribute.

Why this is the right fix (and not a repeat of the failed attempts):
  - SICL used *subject as the negative class* → it pushed apart same-class-
    different-user pairs and destroyed anatomy-correlated activity signal.
  - DANN marginally suppressed all user signal → same collateral damage.
  - Here we do the OPPOSITE: we PULL same-class-different-user pairs TOGETHER
    (conditional invariance), so class-discriminative structure is preserved
    while the *within-class* user variation is collapsed.

Pairing logic for anchor i (embeddings L2-normalised, temperature tau):
    positives P(i) = {p : y_p = y_i, p != i}
    weight   w_ip  = gamma_cross  if user_p != user_i   (cross-user → invariance)
                   = 1.0          if user_p == user_i
    loss_i        = -(1 / sum_p w_ip) * sum_{p in P(i)} w_ip * log_prob(i, p)
where log_prob(i,p) = sim_ip - log sum_{a != i} exp(sim_ia).

Anchors with no in-batch positive are masked out — use a class-balanced sampler
so the minority classes (L2/L3/L5) still get positives in each batch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def conditional_supcon_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    tau: float = 0.1,
    gamma_cross: float = 2.0,
) -> torch.Tensor:
    """Conditional supervised-contrastive loss.

    Args:
        z: (B, D) embeddings (will be L2-normalised here).
        y: (B,) long class labels.
        g: (B,) long user/subject ids (integer-encoded).
        tau: softmax temperature.
        gamma_cross: weight multiplier for cross-user positive pairs (>1 pulls
            same-class-different-user pairs together harder → user-invariance).

    Returns:
        Scalar loss. Zero (with grad) if the batch has no valid positive pair.
    """
    device = z.device
    B = z.shape[0]
    if B < 2:
        return z.sum() * 0.0

    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / tau                                       # (B, B)
    # Row-wise max subtraction for numerical stability (detach so it is a const)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)          # exclude self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    same_y = y.unsqueeze(0) == y.unsqueeze(1)                     # (B, B)
    pos = same_y & (~self_mask)                                  # positives
    diff_user = (g.unsqueeze(0) != g.unsqueeze(1))               # cross-user

    w = torch.where(
        diff_user,
        torch.full((B, B), float(gamma_cross), device=device),
        torch.ones((B, B), device=device),
    ) * pos.float()                                              # zero non-pos

    Z = w.sum(dim=1)                                             # (B,)
    valid = Z > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    loss_i = -(w * log_prob).sum(dim=1) / Z.clamp_min(1e-12)
    return loss_i[valid].mean()
