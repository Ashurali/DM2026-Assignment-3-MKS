"""Basic window-level features for the Day-1 LightGBM submission.

~50 features per file. The Phase-3 expansion ([PROJECT_PLAN.md §Phase 3])
adds FFT bands, autocorrelation, sub-window pooling, jerk, and cross-axis
correlations on top of this.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
STATS = ["mean", "std", "min", "max", "median", "q25", "q75"]


def _agg(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }


def build_features(file_path: str) -> dict[str, float]:
    """Compute ~50 scalar features summarising one window CSV."""
    df = pd.read_csv(file_path).sort_values("index").reset_index(drop=True)
    feat: dict[str, float] = {}

    # 6 channels × 7 stats = 42 features
    for ch in CHANNELS:
        for k, v in _agg(df[ch]).items():
            feat[f"{ch}__{k}"] = v

    # Window-mean magnitude on the 3 mean axes (gravity vector magnitude proxy)
    mag = np.sqrt(df["mean_x"] ** 2 + df["mean_y"] ** 2 + df["mean_z"] ** 2)
    for k, v in _agg(mag).items():
        feat[f"mag_mean__{k}"] = v

    # Std-channel magnitude (intensity proxy)
    smag = np.sqrt(df["std_x"] ** 2 + df["std_y"] ** 2 + df["std_z"] ** 2)
    for k, v in _agg(smag).items():
        feat[f"mag_std__{k}"] = v

    # Explicit gravity-orientation features (also covered by mean_*__mean above,
    # exposed separately for clarity and to make ablation easier).
    feat["gravity_x"] = float(df["mean_x"].mean())
    feat["gravity_y"] = float(df["mean_y"].mean())
    feat["gravity_z"] = float(df["mean_z"].mean())
    feat["gravity_norm"] = float(
        np.sqrt(feat["gravity_x"] ** 2 + feat["gravity_y"] ** 2 + feat["gravity_z"] ** 2)
    )

    return feat


def build_dataset(file_paths: Iterable[str], file_ids: Iterable[int]) -> pd.DataFrame:
    """Build a feature DataFrame from a list of CSV paths.

    Returns a DataFrame with `file_id` as the first column followed by all
    feature columns. Order is preserved from `file_paths` / `file_ids`.
    """
    rows = [build_features(p) for p in file_paths]
    df = pd.DataFrame(rows)
    df.insert(0, "file_id", list(file_ids))
    return df
