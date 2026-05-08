"""Smoke tests for DANN: gradient reversal + model forward + λ schedule."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.models.cnn_bilstm_dann import CNNBiLSTM_DANN, grad_reverse, lambda_schedule


def test_grl_forward_is_identity():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, lambda_=1.0)
    torch.testing.assert_close(y, x)


def test_grl_backward_negates_gradient():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, lambda_=2.0)
    y.sum().backward()
    # The gradient of sum is 1s; with λ=2, we expect −2s in x.grad
    expected = -2.0 * torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected)


def test_dann_model_forward_shapes():
    m = CNNBiLSTM_DANN(n_classes=6, n_domains=60, in_channels=6)
    m.set_lambda(0.5)
    x = torch.randn(4, 6, 300)
    cls, dom, emb = m(x)
    assert cls.shape == (4, 6)
    assert dom.shape == (4, 60)
    assert emb.shape == (4, 256)


def test_dann_backward_runs():
    m = CNNBiLSTM_DANN(n_classes=6, n_domains=10).train()
    m.set_lambda(0.3)
    x = torch.randn(2, 6, 300)
    yc = torch.tensor([0, 4], dtype=torch.long)
    yd = torch.tensor([3, 7], dtype=torch.long)
    cls, dom, _ = m(x)
    loss = torch.nn.functional.cross_entropy(cls, yc) + torch.nn.functional.cross_entropy(dom, yd)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad


def test_lambda_schedule_monotone():
    progress = np.linspace(0, 1, 11)
    lams = [lambda_schedule(p, gamma=10, max_lambda=1.0) for p in progress]
    # Should monotonically increase from ~0 to ~1
    assert lams[0] < 0.05
    assert lams[-1] > 0.95
    assert all(b >= a - 1e-6 for a, b in zip(lams, lams[1:]))


def test_dann_input_8_channels_with_concat_stats():
    """If per_file_norm + concat_stats is used, input is 8 channels."""
    m = CNNBiLSTM_DANN(n_classes=6, n_domains=60, in_channels=8)
    x = torch.randn(2, 8, 300)
    cls, dom, _ = m(x)
    assert cls.shape == (2, 6)
    assert dom.shape == (2, 60)
