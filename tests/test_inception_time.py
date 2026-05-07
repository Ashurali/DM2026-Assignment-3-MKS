"""Smoke tests for InceptionTime."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.models.inception_time import InceptionTime, InceptionModule


def test_inception_module_shape():
    m = InceptionModule(in_channels=6, n_filters=32)
    x = torch.randn(2, 6, 300)
    out = m(x)
    # Output channels = (3 conv branches + 1 pool branch) × n_filters = 4 × 32 = 128
    assert out.shape == (2, 128, 300)


def test_inception_time_forward():
    m = InceptionTime(in_channels=6, n_classes=6, depth=6)
    x = torch.randn(4, 6, 300)
    out = m(x)
    assert out.shape == (4, 6)
    assert out.requires_grad


def test_inception_time_backward():
    m = InceptionTime(in_channels=6, n_classes=6, depth=6).train()
    x = torch.randn(2, 6, 300)
    y = torch.tensor([0, 4], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad


def test_inception_time_residual_path():
    """Forward should run cleanly with depth=6 (2 residual blocks)."""
    m = InceptionTime(in_channels=6, n_classes=6, depth=6, residual_every=3).eval()
    x = torch.randn(1, 6, 300)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()


def test_inception_time_param_count_reasonable():
    """Sanity check: the model should be ~100K-1M parameters."""
    m = InceptionTime(in_channels=6, n_classes=6, depth=6, n_filters=32)
    n = sum(p.numel() for p in m.parameters())
    assert 50_000 < n < 2_000_000, f"Got {n:,} params"
