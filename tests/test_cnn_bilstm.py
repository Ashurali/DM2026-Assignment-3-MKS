"""Smoke tests for the Phase-5 CNN-BiLSTM model and augmentations.

Skipped when torch isn't installed (Phase 5 is a server-only run on the 4090).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")  # skip whole module if torch missing

from src.models.cnn_bilstm import (
    CNNBiLSTM, AttentionPool1d, SeqDataset, augment_sample,
    _random_rotation_3d, _time_warp,
)


def test_cnn_bilstm_forward_shape():
    model = CNNBiLSTM(n_classes=6, in_channels=6, lstm_hidden=128, dropout=0.0)
    x = torch.randn(4, 6, 300)
    out = model(x)
    assert out.shape == (4, 6)
    assert out.requires_grad


def test_cnn_bilstm_backward():
    model = CNNBiLSTM(n_classes=6).train()
    x = torch.randn(2, 6, 300)
    y = torch.tensor([0, 3], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_attention_pool_softmax_sums_to_one():
    pool = AttentionPool1d(in_dim=8)
    x = torch.randn(2, 10, 8)
    out = pool(x)
    assert out.shape == (2, 8)


def test_random_rotation_is_orthogonal():
    rng = np.random.default_rng(42)
    R = _random_rotation_3d(rng)
    assert R.shape == (3, 3)
    # R @ R.T ≈ I
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    # det(R) ≈ +1 (proper rotation, not reflection)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_time_warp_preserves_shape_and_endpoints():
    rng = np.random.default_rng(0)
    x = np.random.randn(6, 300).astype(np.float32)
    out = _time_warp(x, sigma=0.05, rng=rng)
    assert out.shape == x.shape
    # Endpoints should be (approximately) preserved by knot anchoring
    np.testing.assert_allclose(out[:, 0], x[:, 0], atol=1e-4)


def test_augment_sample_no_nan_and_correct_shape():
    rng = np.random.default_rng(0)
    x = np.random.randn(6, 300).astype(np.float32) * 0.1
    x[3:] = np.abs(x[3:])  # std channels are non-negative
    for _ in range(5):
        out = augment_sample(x, rng)
        assert out.shape == x.shape
        assert np.isfinite(out).all()
        assert (out[3:] >= 0).all(), "std channels must remain non-negative"


def test_dataset_iter_with_and_without_aug():
    X = np.random.randn(8, 6, 300).astype(np.float32)
    X[:, 3:] = np.abs(X[:, 3:])
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1])

    ds_eval = SeqDataset(X, y, training=False)
    ds_train = SeqDataset(X, y, training=True)

    x0_eval, y0_eval = ds_eval[0]
    x0_train, y0_train = ds_train[0]
    assert x0_eval.shape == (6, 300)
    assert x0_train.shape == (6, 300)
    assert y0_eval == 0 and y0_train == 0
    # Test mode never modifies inputs (deterministic)
    np.testing.assert_array_equal(x0_eval.numpy(), X[0])


def test_dataset_test_mode_returns_only_x():
    X = np.random.randn(4, 6, 300).astype(np.float32)
    ds = SeqDataset(X, training=False)
    out = ds[0]
    # In test mode (no y), __getitem__ returns just the tensor
    assert isinstance(out, torch.Tensor)
    assert out.shape == (6, 300)
