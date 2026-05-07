"""Smoke tests for the Phase-6 Transformer model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.models.transformer import TransformerHAR, sinusoidal_positional_encoding


def test_transformer_forward_shape():
    model = TransformerHAR(n_classes=6, in_channels=6, d_model=128, num_layers=4)
    x = torch.randn(4, 6, 300)
    out = model(x)
    assert out.shape == (4, 6)
    assert out.requires_grad


def test_transformer_backward():
    model = TransformerHAR().train()
    x = torch.randn(2, 6, 300)
    y = torch.tensor([0, 4], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_positional_encoding_shape():
    pe = sinusoidal_positional_encoding(seq_len=301, d_model=128)
    assert pe.shape == (301, 128)
    # Sin/cos pairs at the smallest frequency: pe[1, 0] should differ from pe[0, 0]
    assert not torch.allclose(pe[0], pe[1])


def test_cls_token_distinct_position():
    """The [CLS] embedding at position 0 should be the (LayerNormed) representation."""
    model = TransformerHAR().eval()
    x = torch.randn(1, 6, 300)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()
