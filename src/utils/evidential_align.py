"""Evidential Alignment — paper-faithful losses + head (Ye, Zheng, Zhang, KDD 2025).

"Improving Group Robustness on Spurious Correlation via Evidential Alignment."
Self-contained (does not import the older experiments_archive evidential code)
so it can be scp'd to the server without depending on / clobbering existing
files.

Evidential basics. The last layer outputs per-class *evidence* e_k ≥ 0 (softplus).
Dirichlet concentration α_k = e_k + 1; strength S = Σα_k; expected prob
E[p_k] = α_k / S; epistemic uncertainty u(x) = K / S  (K = n_classes).

Stage 1 — Second-order Risk Minimization (Eq. 12):
    L1 = −log(α_y / S)  +  λ_t · KL(Dir(α̃) ‖ Dir(1))
  classification term is the expected-probability NLL (the paper's form — NOT
  the Bayes-risk MSE used by the earlier failed attempt). α̃ clamps the correct
  class to 1 so KL only penalises evidence on WRONG classes. λ_t = min(t/η, 1).

Stage 2 — Evidential Calibration (Eq. 14–15):
    w(x, y) = 1 if argmax E[p] == y  else  u(x)
    L2 = E[w · CE(E[p], y)]  +  β · ‖θ2 − θ1‖²
  drawn class-balanced. Upweights high-uncertainty (minority-group) errors,
  anchors to the Stage-1 last layer. Both stages train ONLY the last layer on a
  frozen backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Evidential head (the "last layer")
# ─────────────────────────────────────────────────────────────────────────────
class EvidentialHead(nn.Module):
    """Maps a frozen backbone embedding → per-class evidence e_k ≥ 0.

    Default is a single Linear (the paper's "retrain only the last layer").
    Pass `hidden_dim` for a small MLP head if more capacity is wanted.
    """

    def __init__(self, in_dim: int, n_classes: int = 6,
                 hidden_dim: int | None = None, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        if hidden_dim is None:
            self.net: nn.Module = nn.Linear(in_dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(emb))          # evidence ≥ 0


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet helpers
# ─────────────────────────────────────────────────────────────────────────────
def alpha_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def probs_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    alpha = evidence + 1.0
    return alpha / alpha.sum(dim=1, keepdim=True)


def uncertainty_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    """u(x) = K / S ∈ (0, 1]."""
    alpha = evidence + 1.0
    return alpha.shape[1] / alpha.sum(dim=1)


def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(α) ‖ Dir(1,…,1)) per row → (B,). Closed form (Sensoy 2018)."""
    K = float(alpha.shape[1])
    S = alpha.sum(dim=1)
    log_term = (
        torch.lgamma(S)
        - torch.lgamma(torch.tensor(K, device=alpha.device, dtype=alpha.dtype))
        - torch.lgamma(alpha).sum(dim=1)
    )
    digamma_term = (
        (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S).unsqueeze(1))
    ).sum(dim=1)
    return log_term + digamma_term


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Second-order Risk Minimization
# ─────────────────────────────────────────────────────────────────────────────
def sorm_loss(
    evidence: torch.Tensor,
    y: torch.Tensor,
    lambda_t: float,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """L1 = −log(α_y/S) + λ_t · KL(Dir(α̃)‖Dir(1)).  (Ye et al. Eq. 12)"""
    alpha = evidence + 1.0
    S = alpha.sum(dim=1)
    K = alpha.shape[1]
    y_oh = F.one_hot(y, num_classes=K).float()

    # Classification: expected-probability NLL = log S − log α_y (stable; α_y ≥ 1).
    alpha_y = (alpha * y_oh).sum(dim=1)
    cls = torch.log(S) - torch.log(alpha_y)

    # Evidence regulariser with correct-class clamp: α̃_y = 1.
    alpha_tilde = (1.0 - y_oh) * alpha + y_oh
    kl = kl_dirichlet_to_uniform(alpha_tilde)

    per_sample = cls + lambda_t * kl
    if class_weights is not None:
        per_sample = per_sample * class_weights[y]
    loss = per_sample.mean()

    with torch.no_grad():
        u = (K / S).mean()
        acc = ((alpha / S.unsqueeze(1)).argmax(1) == y).float().mean()
    return loss, {
        "loss": float(loss.item()), "cls": float(cls.mean().item()),
        "kl": float(kl.mean().item()), "lambda_t": float(lambda_t),
        "u_mean": float(u.item()), "acc": float(acc.item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Evidential Calibration
# ─────────────────────────────────────────────────────────────────────────────
def reweighted_ce(evidence: torch.Tensor, y: torch.Tensor,
                  eps: float = 1e-8) -> tuple[torch.Tensor, dict]:
    """E[w · CE(E[p], y)] with w = 1 if correct else u(x).  (Eq. 14–15, 1st term)

    The weight is detached — gradient flows through the CE only — so the
    uncertainty acts as a fixed sample importance, not a path for the model to
    game by inflating S.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    K = alpha.shape[1]

    pred = p.argmax(dim=1)
    correct = (pred == y).float()
    u = (K / S.squeeze(1))
    w = correct + (1.0 - correct) * u                 # 1 if right, u if wrong

    log_p = torch.log(p.clamp_min(eps))
    ce = -log_p.gather(1, y.unsqueeze(1)).squeeze(1)
    loss = (w.detach() * ce).mean()

    with torch.no_grad():
        wrong = (1.0 - correct)
        wrong_u = (u * wrong).sum() / wrong.sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.item()), "frac_correct": float(correct.mean().item()),
        "wrong_u_mean": float(wrong_u.item()), "u_mean": float(u.mean().item()),
    }


def anchor_penalty(head: nn.Module,
                   anchor_state: dict[str, torch.Tensor]) -> torch.Tensor:
    """β-less ‖θ2 − θ1‖² over the head's parameters (caller multiplies by β)."""
    device = next(head.parameters()).device
    total = torch.zeros((), device=device)
    cur = dict(head.named_parameters())
    for name, p in cur.items():
        if name in anchor_state:
            total = total + (p - anchor_state[name].to(device)).pow(2).sum()
    return total


def lambda_anneal(epoch: int, eta: int) -> float:
    """λ_t = min(t/η, 1) — t is 1-indexed epoch, η the annealing horizon."""
    if eta <= 0:
        return 1.0
    return min(float(epoch) / float(eta), 1.0)
