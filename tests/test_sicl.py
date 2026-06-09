"""Smoke tests for SICL: contrastive head + sicl_loss."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.models.sicl import ContrastiveProjectionHead, sicl_loss


def test_projection_head_output_normalized():
    h = ContrastiveProjectionHead(in_dim=256, proj_dim=128)
    x = torch.randn(4, 256)
    z = h(x)
    norms = z.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)


def test_sicl_loss_runs_and_is_finite():
    B, D = 8, 64
    z1 = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    subjects = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    loss = sicl_loss(z1, z2, labels, subjects, temperature=0.1, q_s=0.5)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_sicl_loss_lower_when_positives_close():
    """If two views are nearly identical and same-class, loss should be lower
    than when views are random."""
    B, D = 6, 32
    # Same-class pairs, near-identical views
    same = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z1 = same
    z2 = same + 0.01 * torch.randn_like(same)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    subjects = torch.tensor([0, 1, 2, 3, 4, 5])
    loss_aligned = sicl_loss(z1, z2, labels, subjects, temperature=0.1, q_s=0.5).item()

    # Random views
    z1r = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z2r = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    loss_random = sicl_loss(z1r, z2r, labels, subjects, temperature=0.1, q_s=0.5).item()

    assert loss_aligned < loss_random


def test_sicl_loss_qs_zero_ignores_same_subject():
    """With q_s=0, same-subject negatives don't contribute → loss should
    differ from q_s=1.0 (standard SupCon)."""
    B, D = 6, 32
    z1 = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    z2 = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    labels = torch.tensor([0, 1, 0, 1, 2, 2])
    subjects = torch.tensor([0, 0, 1, 1, 2, 2])
    loss_qs0 = sicl_loss(z1, z2, labels, subjects, temperature=0.1, q_s=0.0).item()
    loss_qs1 = sicl_loss(z1, z2, labels, subjects, temperature=0.1, q_s=1.0).item()
    assert abs(loss_qs0 - loss_qs1) > 1e-4
