"""Smoke tests for src/features/build.py.

These run on real data (a 10-file sample) so we catch numerical issues
(skew on near-constant series, FFT-on-empty, etc.) early.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build import (
    build_features, build_dataset, FEATURE_GROUPS, ALL_GROUPS,
    feat_basic_stats, feat_magnitude, feat_gravity, feat_jerk, feat_fft,
    feat_autocorr, feat_subwindow, feat_crossaxis, feat_zerocross,
    feat_quality, feat_per_file_norm,
)


META_TRAIN = ROOT / "data" / "meta_train.parquet"


def _sample_paths(n: int = 10):
    if not META_TRAIN.exists():
        pytest.skip("meta_train.parquet not present — run notebooks/01_eda.py first")
    meta = pd.read_parquet(META_TRAIN).head(n)
    return meta["path"].tolist(), meta["file_id"].tolist()


def test_build_features_shape_and_no_nan():
    paths, ids = _sample_paths(5)
    df = build_dataset(paths, ids, show_progress=False)
    assert df.shape[0] == 5
    # >150 features for the default catalog (excluding file_id)
    assert df.shape[1] - 1 > 150, f"got {df.shape[1] - 1} features"
    # Every cell finite
    feats = df.iloc[:, 1:].values
    assert np.isfinite(feats).all(), "non-finite feature value(s)"


def test_each_group_runs_independently():
    """Every feature group must run on a real CSV without raising or producing NaN."""
    paths, _ = _sample_paths(3)
    for path in paths:
        df_raw = pd.read_csv(path)
        for name, fn in FEATURE_GROUPS.items():
            out = fn(df_raw)
            assert isinstance(out, dict), f"{name} did not return a dict"
            assert len(out) > 0, f"{name} returned empty dict"
            for k, v in out.items():
                assert np.isfinite(v), f"{name}.{k} = {v} is not finite (path={path})"


def test_exclude_reduces_feature_count():
    paths, ids = _sample_paths(3)
    full = build_dataset(paths, ids, show_progress=False)
    no_fft = build_dataset(paths, ids, exclude=["fft"], show_progress=False)
    assert full.shape[1] > no_fft.shape[1], \
        "excluding 'fft' should reduce feature count"


def test_all_groups_known():
    assert set(FEATURE_GROUPS.keys()) == set(ALL_GROUPS)
    # Sanity: 11 groups defined
    assert len(ALL_GROUPS) == 11
