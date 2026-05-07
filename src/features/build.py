"""Phase-3 feature catalog for the LightGBM track.

Each feature group is a separate function returning a `dict[str, float]`,
so they can be ablated cleanly (Phase 4). The full catalog is composed by
`build_features(path, exclude=None)`.

Feature counts (approximate):
- basic_stats:  84  (14 stats × 6 channels)
- magnitude:    28  (14 stats × 2 magnitudes)
- gravity:      11  (3 means + norm + theta + phi + per-chunk variance)
- jerk:         24  (4 stats × 6 diff-channels)
- fft:          36  (3 bands × 6 + dominant freq + dom power + entropy, on each mean channel)
- autocorr:      6  (lag-1, 5, 10, 30 + first-peak loc + height)
- subwindow:    60  (5 chunks × 6 channels × 2 stats)
- crossaxis:     6  (3 mean-axis pairs + 3 std-axis pairs)
- zerocross:    12  (zero-cross rate × 6 + n_peaks-on-mag + amplitude)
- quality:       2  (n_rows, nan_total)
- per_file_norm: 18 (z-scored channel stats × 6, kept lean)
Total: ~287 features.

Each function reads the same `df` (pre-loaded CSV) and returns scalar features.
The top-level `build_features(path)` does one CSV read and dispatches.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import signal as scisig
from scipy import stats as scistats

CHANNELS_MEAN = ["mean_x", "mean_y", "mean_z"]
CHANNELS_STD = ["std_x", "std_y", "std_z"]
CHANNELS = CHANNELS_MEAN + CHANNELS_STD


# -----------------------------------------------------------------------------
# Per-channel basic statistics  (14 × 6 = 84)
# -----------------------------------------------------------------------------
def _stats_dict(arr: np.ndarray) -> dict[str, float]:
    """14 summary stats for a 1-D array."""
    arr = np.asarray(arr, dtype=np.float64)
    q10, q25, q50, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    mad = float(np.median(np.abs(arr - q50)))
    # scipy.stats handles edge cases; bias=False matches pandas default
    sk = float(scistats.skew(arr, bias=False)) if arr.std() > 1e-12 else 0.0
    ku = float(scistats.kurtosis(arr, bias=False)) if arr.std() > 1e-12 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(q50),
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "iqr": float(q75 - q25),
        "range": float(arr.max() - arr.min()),
        "mad": mad,
        "skew": sk,
        "kurt": ku,
    }


def feat_basic_stats(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in CHANNELS:
        for k, v in _stats_dict(df[ch].values).items():
            out[f"{ch}__{k}"] = v
    return out


# -----------------------------------------------------------------------------
# Magnitude features  (14 × 2 = 28)
# -----------------------------------------------------------------------------
def feat_magnitude(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    mag = np.sqrt(df["mean_x"].values ** 2 + df["mean_y"].values ** 2 + df["mean_z"].values ** 2)
    smag = np.sqrt(df["std_x"].values ** 2 + df["std_y"].values ** 2 + df["std_z"].values ** 2)
    for k, v in _stats_dict(mag).items():
        out[f"mag_mean__{k}"] = v
    for k, v in _stats_dict(smag).items():
        out[f"mag_std__{k}"] = v
    return out


# -----------------------------------------------------------------------------
# Gravity orientation  (~11)
# -----------------------------------------------------------------------------
def feat_gravity(df: pd.DataFrame, n_chunks: int = 5) -> dict[str, float]:
    gx = float(df["mean_x"].mean())
    gy = float(df["mean_y"].mean())
    gz = float(df["mean_z"].mean())
    gnorm = float(np.sqrt(gx * gx + gy * gy + gz * gz))
    # Spherical angles (theta = polar from +z, phi = azimuth in xy)
    theta = float(np.arccos(np.clip(gz / (gnorm + 1e-12), -1.0, 1.0)))
    phi = float(np.arctan2(gy, gx))
    # Variance of gravity vector across n_chunks sub-windows
    chunks = np.array_split(df[CHANNELS_MEAN].values, n_chunks)
    chunk_means = np.vstack([c.mean(axis=0) for c in chunks])  # (n_chunks, 3)
    chunk_g_norms = np.linalg.norm(chunk_means, axis=1)
    drift = float(chunk_g_norms.std(ddof=1))
    # Cosine angle between first and last chunk's gravity vector
    a, b = chunk_means[0], chunk_means[-1]
    cos_ab = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    # Per-axis chunk-variance summed
    chunk_axis_var = float(chunk_means.var(axis=0).sum())
    return {
        "grav_x": gx,
        "grav_y": gy,
        "grav_z": gz,
        "grav_norm": gnorm,
        "grav_theta": theta,
        "grav_phi": phi,
        "grav_drift_norm_std": drift,
        "grav_first_last_cos": cos_ab,
        "grav_chunk_axis_var": chunk_axis_var,
        "grav_chunk_norm_min": float(chunk_g_norms.min()),
        "grav_chunk_norm_max": float(chunk_g_norms.max()),
    }


# -----------------------------------------------------------------------------
# Jerk (temporal first-derivative)  (4 × 6 = 24)
# -----------------------------------------------------------------------------
def feat_jerk(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in CHANNELS:
        d = np.diff(df[ch].values)
        out[f"jerk_{ch}__mean_abs"] = float(np.mean(np.abs(d)))
        out[f"jerk_{ch}__std"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        out[f"jerk_{ch}__max_abs"] = float(np.max(np.abs(d))) if len(d) else 0.0
        out[f"jerk_{ch}__rms"] = float(np.sqrt(np.mean(d * d))) if len(d) else 0.0
    return out


# -----------------------------------------------------------------------------
# Frequency-domain  (~36)
# -----------------------------------------------------------------------------
def _fft_features(arr: np.ndarray, prefix: str, fs: float = 1.0) -> dict[str, float]:
    """FFT band-energies, dominant freq, dominant power, spectral entropy."""
    arr = arr - arr.mean()
    F = np.fft.rfft(arr)
    P = (np.abs(F) ** 2).astype(np.float64)
    f_axis = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    P_total = float(P.sum() + 1e-12)
    # Skip DC for "dominant"
    if len(P) > 1:
        dom_idx = int(np.argmax(P[1:]) + 1)
        dom_hz = float(f_axis[dom_idx])
        dom_pow = float(P[dom_idx])
    else:
        dom_hz = 0.0
        dom_pow = 0.0
    # Band energies (Hz cutoffs are conservative given Nyquist = 0.5 Hz here)
    band_lo = float(P[(f_axis >= 0.0) & (f_axis < 0.05)].sum())
    band_mid = float(P[(f_axis >= 0.05) & (f_axis < 0.15)].sum())
    band_hi = float(P[(f_axis >= 0.15) & (f_axis <= 0.5)].sum())
    # Spectral entropy
    p = P / P_total
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    return {
        f"{prefix}__dom_hz": dom_hz,
        f"{prefix}__dom_pow": dom_pow,
        f"{prefix}__band_lo": band_lo,
        f"{prefix}__band_mid": band_mid,
        f"{prefix}__band_hi": band_hi,
        f"{prefix}__spec_ent": ent,
    }


def feat_fft(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in CHANNELS_MEAN:
        out.update(_fft_features(df[ch].values, prefix=f"fft_{ch}"))
    # Magnitude (mean) FFT
    mag = np.sqrt(df["mean_x"].values ** 2 + df["mean_y"].values ** 2 + df["mean_z"].values ** 2)
    out.update(_fft_features(mag, prefix="fft_mag_mean"))
    return out


# -----------------------------------------------------------------------------
# Autocorrelation  (~6)
# -----------------------------------------------------------------------------
def _autocorr_lag(arr: np.ndarray, lag: int) -> float:
    """Pearson autocorr at the given lag."""
    if lag >= len(arr):
        return 0.0
    a = arr[: len(arr) - lag]
    b = arr[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
    return float((a * b).sum() / denom)


def feat_autocorr(df: pd.DataFrame) -> dict[str, float]:
    mag = np.sqrt(df["mean_x"].values ** 2 + df["mean_y"].values ** 2 + df["mean_z"].values ** 2)
    out = {
        "ac_mag__lag1": _autocorr_lag(mag, 1),
        "ac_mag__lag5": _autocorr_lag(mag, 5),
        "ac_mag__lag10": _autocorr_lag(mag, 10),
        "ac_mag__lag30": _autocorr_lag(mag, 30),
    }
    # First major peak in autocorr beyond lag-2 (rhythmicity proxy)
    max_lag = min(len(mag) // 2, 100)
    if max_lag > 3:
        ac = np.array([_autocorr_lag(mag, lg) for lg in range(2, max_lag)])
        peaks, _ = scisig.find_peaks(ac, height=0.05)
        if len(peaks) > 0:
            first_peak = int(peaks[0]) + 2
            out["ac_mag__first_peak_lag"] = float(first_peak)
            out["ac_mag__first_peak_height"] = float(ac[peaks[0]])
        else:
            out["ac_mag__first_peak_lag"] = 0.0
            out["ac_mag__first_peak_height"] = 0.0
    else:
        out["ac_mag__first_peak_lag"] = 0.0
        out["ac_mag__first_peak_height"] = 0.0
    return out


# -----------------------------------------------------------------------------
# Sub-window pooling  (5 × 6 × 2 = 60)
# -----------------------------------------------------------------------------
def feat_subwindow(df: pd.DataFrame, n_chunks: int = 5) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in CHANNELS:
        chunks = np.array_split(df[ch].values, n_chunks)
        for i, c in enumerate(chunks):
            out[f"sw{i}_{ch}__mean"] = float(c.mean())
            out[f"sw{i}_{ch}__std"] = float(c.std(ddof=1)) if len(c) > 1 else 0.0
    return out


# -----------------------------------------------------------------------------
# Cross-axis correlations  (6)
# -----------------------------------------------------------------------------
def feat_crossaxis(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    pairs = [
        ("mean_x", "mean_y"),
        ("mean_x", "mean_z"),
        ("mean_y", "mean_z"),
        ("std_x", "std_y"),
        ("std_x", "std_z"),
        ("std_y", "std_z"),
    ]
    for a, b in pairs:
        out[f"corr__{a}__{b}"] = safe_corr(df[a].values, df[b].values)
    return out


# -----------------------------------------------------------------------------
# Zero-crossings & peaks  (~12)
# -----------------------------------------------------------------------------
def feat_zerocross(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for ch in CHANNELS_MEAN:
        x = df[ch].values - df[ch].values.mean()
        zc = int(np.sum(np.sign(x[1:]) != np.sign(x[:-1])))
        out[f"zcr_{ch}"] = float(zc) / max(1, len(x) - 1)
    for ch in CHANNELS_STD:
        x = df[ch].values - df[ch].values.mean()
        zc = int(np.sum(np.sign(x[1:]) != np.sign(x[:-1])))
        out[f"zcr_{ch}"] = float(zc) / max(1, len(x) - 1)
    # Peaks on smoothed magnitude
    mag = np.sqrt(df["mean_x"].values ** 2 + df["mean_y"].values ** 2 + df["mean_z"].values ** 2)
    if len(mag) >= 11:
        smooth = pd.Series(mag).rolling(window=11, center=True).mean().bfill().ffill().values
    else:
        smooth = mag
    peaks, props = scisig.find_peaks(smooth, prominence=smooth.std() * 0.3 if smooth.std() > 0 else 0.0)
    out["mag_n_peaks"] = float(len(peaks))
    out["mag_peak_mean_height"] = float(np.mean(props["prominences"])) if len(peaks) > 0 else 0.0
    return out


# -----------------------------------------------------------------------------
# Quality features  (2 — essentially constant on this dataset, kept for safety)
# -----------------------------------------------------------------------------
def feat_quality(df: pd.DataFrame) -> dict[str, float]:
    return {
        "qual_n_rows": float(len(df)),
        "qual_nan_total": float(df.isna().sum().sum()),
    }


# -----------------------------------------------------------------------------
# Per-file (z-scored) channel stats  (3 × 6 = 18)
# -----------------------------------------------------------------------------
def feat_per_file_norm(df: pd.DataFrame) -> dict[str, float]:
    """After per-file z-score, what's the *shape* of the residual? Robust to
    inter-user gravity-orientation drift identified in EDA §5."""
    out: dict[str, float] = {}
    for ch in CHANNELS:
        x = df[ch].values
        mu = x.mean()
        sd = x.std(ddof=1) if len(x) > 1 else 0.0
        if sd < 1e-12:
            z = np.zeros_like(x)
        else:
            z = (x - mu) / sd
        # Lean: only mean-abs, max, skew  (the central "shape" stats)
        out[f"znorm_{ch}__mean_abs"] = float(np.mean(np.abs(z)))
        out[f"znorm_{ch}__max"] = float(np.max(np.abs(z)))
        out[f"znorm_{ch}__skew"] = float(scistats.skew(z, bias=False)) if sd >= 1e-12 else 0.0
    return out


# -----------------------------------------------------------------------------
# Group registry  (used for ablation)
# -----------------------------------------------------------------------------
FEATURE_GROUPS = {
    "basic_stats": feat_basic_stats,
    "magnitude": feat_magnitude,
    "gravity": feat_gravity,
    "jerk": feat_jerk,
    "fft": feat_fft,
    "autocorr": feat_autocorr,
    "subwindow": feat_subwindow,
    "crossaxis": feat_crossaxis,
    "zerocross": feat_zerocross,
    "quality": feat_quality,
    "per_file_norm": feat_per_file_norm,
}
ALL_GROUPS: tuple[str, ...] = tuple(FEATURE_GROUPS.keys())


def build_features(file_path: str, exclude: Iterable[str] = ()) -> dict[str, float]:
    """Read one CSV and compute features from all groups except `exclude`."""
    df = pd.read_csv(file_path).sort_values("index").reset_index(drop=True)
    skip = set(exclude)
    feat: dict[str, float] = {}
    for name, fn in FEATURE_GROUPS.items():
        if name in skip:
            continue
        feat.update(fn(df))
    return feat


def build_dataset(
    file_paths: Sequence[str],
    file_ids: Sequence[int],
    exclude: Iterable[str] = (),
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build the full feature DataFrame from a list of CSV paths.

    `exclude` is the set of group names to skip (for ablation). Default = none.
    """
    if show_progress:
        from tqdm.auto import tqdm
        rows = [build_features(p, exclude=exclude) for p in tqdm(file_paths, desc="features")]
    else:
        rows = [build_features(p, exclude=exclude) for p in file_paths]
    df = pd.DataFrame(rows)
    df.insert(0, "file_id", list(file_ids))
    return df
