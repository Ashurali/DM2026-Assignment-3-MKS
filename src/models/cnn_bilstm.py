"""Phase-5 CNN-BiLSTM model + sequence dataset + augmentations.

Architecture (per PROJECT_PLAN.md §Phase 5, Xia et al. 2020-style):
    Input (B, 6, 300)
    → BatchNorm1d
    → 2× [Conv1d(k=5) + ReLU] + MaxPool1d
    → 2× [Conv1d(k=5) + ReLU] + MaxPool1d
    → Conv1d(k=3) + ReLU
    → BiLSTM (1 layer, hidden=128)
    → Attention pooling over time
    → Dropout(0.3) → Linear(256→6)

The motivation from Phase-4 ablation: 5 of 6 L2-critical groups encode
temporal structure (jerk, subwindow, fft, autocorr, zerocross). A 1D-CNN's
first conv layer learns derivative-like kernels for free; deeper layers
compose multi-scale temporal patterns. So the LGBM L2 ceiling at F1 ≈ 0.27
should give way to direct sequence modelling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class AttentionPool1d(nn.Module):
    """Soft-attention pooling over the time dimension."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        a = self.score(x).squeeze(-1)        # (B, T)
        a = torch.softmax(a, dim=1)
        return torch.sum(x * a.unsqueeze(-1), dim=1)  # (B, D)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_classes: int = 6,
        in_channels: int = 6,
        lstm_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 300 -> 150
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 150 -> 75
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = AttentionPool1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 300)
        x = self.bn(x)
        x = self.conv_block1(x)            # (B, 64, 150)
        x = self.conv_block2(x)            # (B, 128, 75)
        x = self.conv_block3(x)            # (B, 128, 75)
        x = x.permute(0, 2, 1)             # (B, 75, 128)
        x, _ = self.lstm(x)                # (B, 75, 256)
        x = self.attn(x)                   # (B, 256)
        x = self.dropout(x)
        return self.fc(x)                  # (B, n_classes)


# -----------------------------------------------------------------------------
# Augmentations (numpy-level, applied per-sample inside Dataset)
# -----------------------------------------------------------------------------
def _random_rotation_3d(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix in SO(3) via Arvo's method."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    # quaternion → rotation matrix
    R = np.array([
        [1 - 2*(q3*q3 + q4*q4), 2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [2*(q2*q3 + q1*q4),     1 - 2*(q2*q2 + q4*q4), 2*(q3*q4 - q1*q2)],
        [2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2),     1 - 2*(q2*q2 + q3*q3)],
    ])
    return R


def _time_warp(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth cubic-spline distortion of the time axis. x: (C, T)."""
    from scipy.interpolate import CubicSpline
    C, T = x.shape
    # 5 control points, smoothly interpolated
    n_knots = 5
    knot_x = np.linspace(0, T - 1, n_knots)
    knot_y = knot_x + rng.normal(0, sigma * T, size=n_knots)
    knot_y[0] = 0.0
    knot_y[-1] = T - 1
    # Monotonic projection (sort in case knots cross)
    knot_y = np.maximum.accumulate(knot_y)
    cs = CubicSpline(knot_x, knot_y)
    new_t = np.clip(cs(np.arange(T)), 0, T - 1)
    out = np.empty_like(x)
    for c in range(C):
        out[c] = np.interp(new_t, np.arange(T), x[c])
    return out


DEFAULT_AUG_PROBS = {
    "p_rot": 0.5,
    "p_jitter": 0.5,
    "p_scale": 0.3,
    "p_warp": 0.3,
}


def per_file_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-file z-score normalisation: subtract each channel's mean and divide
    by its std, computed over the time axis of THIS file only.

    Tier A.2 fix per the literature synthesis: cross-subject HAR benchmarks
    consistently recommend per-subject standardisation. Since each file in our
    dataset comes from one subject doing one activity, per-file == per-subject.

    Loses absolute-magnitude information. To compensate, callers may
    concatenate the original mean/std as extra channels (we expose this via
    `concat_stats=True` in SeqDataset).
    """
    arr = np.asarray(x, dtype=np.float32)  # (C, T)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / (std + eps)


def augment_sample(
    x: np.ndarray,
    rng: np.random.Generator,
    p_rot: float = 0.5,
    p_jitter: float = 0.5,
    p_scale: float = 0.3,
    p_warp: float = 0.3,
    jitter_mean_sigma: float = 0.02,
    jitter_std_sigma: float = 0.01,
    scale_range: float = 0.1,
    warp_sigma: float = 0.02,
) -> np.ndarray:
    """Apply training-time augmentations to one (6, 300) sample."""
    out = x.copy()

    # 1. Random 3D rotation of the (mean_x, mean_y, mean_z) channels.
    #    Most important per PROJECT_PLAN — handles wrist orientation drift.
    if rng.random() < p_rot:
        R = _random_rotation_3d(rng)
        out[:3] = R @ out[:3]
        # Std channels are rotation-invariant magnitudes per axis; we
        # leave them untouched to preserve their physical interpretation.

    # 2. Gaussian jitter — additive noise.
    if rng.random() < p_jitter:
        out[:3] += rng.normal(0, jitter_mean_sigma, out[:3].shape).astype(out.dtype)
        out[3:] += rng.normal(0, jitter_std_sigma, out[3:].shape).astype(out.dtype)
        out[3:] = np.clip(out[3:], a_min=0.0, a_max=None)  # std must stay >= 0

    # 3. Magnitude scaling — multiplicative.
    if rng.random() < p_scale:
        s = 1.0 + rng.uniform(-scale_range, scale_range)
        out *= s

    # 4. Time warping — smooth temporal distortion.
    if rng.random() < p_warp:
        out = _time_warp(out, warp_sigma, rng)

    return out


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SeqDataset(torch.utils.data.Dataset):
    """In-memory (N, 6, 300) sequence dataset with optional augmentations.

    Storing all data in RAM avoids re-reading 11k CSVs each epoch — the cache
    pre-build is in `data/seq_train.npy` / `data/seq_test.npy`.

    `aug_probs` overrides any of {p_rot, p_jitter, p_scale, p_warp}; missing
    keys fall back to DEFAULT_AUG_PROBS.

    `per_file_norm`: apply per-file z-score normalisation BEFORE augmentations.
    Cross-subject HAR literature consistently recommends per-subject
    standardisation. Each file == one subject in our dataset.
    `concat_stats`: if True (and per_file_norm is True), concatenate the per-
    channel mean and std as 2 extra channels so the model still sees absolute
    magnitude. Output becomes (8, T) instead of (6, T) — caller must use a
    model with in_channels=8.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        training: bool = False,
        seed: int = 42,
        aug_probs: Optional[dict] = None,
        per_file_norm: bool = False,
        concat_stats: bool = False,
    ):
        self.X = X.astype(np.float32)  # (N, 6, 300)
        self.y = None if y is None else y.astype(np.int64)
        self.training = training
        self.aug_probs = {**DEFAULT_AUG_PROBS, **(aug_probs or {})}
        self.per_file_norm = per_file_norm
        self.concat_stats = concat_stats and per_file_norm
        # Per-worker RNG (re-seeded in worker_init_fn)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (6, T) — raw

        # Order: augmentations first (operate on raw 6 channels), THEN per-file
        # z-score, THEN optionally concat the original magnitude tracks as
        # extra channels. This keeps rotation/jitter physically meaningful on
        # raw data while still removing subject offsets at the model input.
        if self.training:
            x = augment_sample(x, self._rng, **self.aug_probs)

        if self.per_file_norm:
            # Capture absolute-magnitude tracks BEFORE z-score for optional concat
            if self.concat_stats:
                mean_mag_track = np.linalg.norm(x[:3], axis=0, keepdims=True)  # (1, T)
                std_mag_track = np.linalg.norm(x[3:], axis=0, keepdims=True)   # (1, T)
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-6)
            if self.concat_stats:
                x = np.concatenate([x, mean_mag_track, std_mag_track], axis=0)  # (8, T)

        x_t = torch.from_numpy(np.ascontiguousarray(x))
        if self.y is None:
            return x_t
        return x_t, int(self.y[idx])


def worker_init_fn(worker_id: int):
    """Re-seed each DataLoader worker so augmentations are independent."""
    info = torch.utils.data.get_worker_info()
    base = info.dataset._rng.bit_generator.state["state"]["state"]
    info.dataset._rng = np.random.default_rng((base + worker_id) % (2 ** 63 - 1))
