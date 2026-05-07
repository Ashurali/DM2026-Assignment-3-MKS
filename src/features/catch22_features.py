"""catch22 features for the 6 channels (132 features total).

The catch22 library (Lubba et al. 2019) provides 22 features chosen via
principled selection from 7,700 candidate time-series features. They cover
distributional shape, autocorrelation, periodicity, fluctuation analysis,
and entropy — complementary to our hand-crafted catalog.

We compute catch22 on each of the 6 channels (mean_x/y/z, std_x/y/z) →
22 × 6 = 132 features per file. Install via `pip install pycatch22`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CHANNELS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def _catch22_one(arr: np.ndarray) -> dict[str, float]:
    """Run catch22 on a 1-D array; returns dict of 22 features."""
    import pycatch22
    arr = np.asarray(arr, dtype=np.float64).tolist()
    out = pycatch22.catch22_all(arr, catch24=False)
    # 'names' and 'values' are parallel lists
    return {name: float(v) for name, v in zip(out["names"], out["values"])}


def feat_catch22(df: pd.DataFrame) -> dict[str, float]:
    feat: dict[str, float] = {}
    for ch in CHANNELS:
        d = _catch22_one(df[ch].values)
        for name, val in d.items():
            feat[f"c22_{ch}__{name}"] = float(val) if np.isfinite(val) else 0.0
    return feat


def build_catch22_dataset(file_paths: Iterable[str], file_ids: Iterable[int], show_progress: bool = True) -> pd.DataFrame:
    """Build a catch22 feature DataFrame. ~132 features per file."""
    if show_progress:
        from tqdm.auto import tqdm
        iterable = tqdm(file_paths, desc="catch22")
    else:
        iterable = file_paths

    rows = []
    for p in iterable:
        df = pd.read_csv(p).sort_values("index").reset_index(drop=True)
        rows.append(feat_catch22(df))
    out = pd.DataFrame(rows)
    out.insert(0, "file_id", list(file_ids))
    return out


# CLI: build cached parquet
def main() -> None:
    import sys
    from pathlib import Path, PureWindowsPath
    ROOT = Path(__file__).resolve().parents[2]

    def fix(p):
        win = PureWindowsPath(str(p))
        parts = win.parts
        if "data" in parts:
            return ROOT / Path(*parts[parts.index("data"):])
        return Path(p)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    meta_train["path"] = meta_train["path"].apply(fix)
    meta_test["path"] = meta_test["path"].apply(fix)

    cache_train = ROOT / "data" / "feat_catch22_train.parquet"
    cache_test = ROOT / "data" / "feat_catch22_test.parquet"

    if cache_train.exists() and cache_test.exists():
        print(f"Cache already present at {cache_train.name} / {cache_test.name} — delete to rebuild.")
        return

    print("Building catch22 features (~3-5 min)...")
    Xtr = build_catch22_dataset(meta_train["path"].tolist(), meta_train["file_id"].tolist())
    Xte = build_catch22_dataset(meta_test["path"].tolist(), meta_test["file_id"].tolist())
    Xtr.to_parquet(cache_train, index=False)
    Xte.to_parquet(cache_test, index=False)
    print(f"Wrote {cache_train.name} ({Xtr.shape}) and {cache_test.name} ({Xte.shape})")


if __name__ == "__main__":
    main()
