"""Day-1 submission #1: predict the majority class for every test file.

Pipeline-validation submission. Confirms that:
- meta_test.parquet → submission CSV path works end-to-end
- the Kaggle scorer accepts our format
- public LB score for "all majority" matches what we'd expect from the
  train-set class fraction (~0.103 micro-F1 with label 1 dominant; macro
  is the floor we computed in EDA §7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.cv import to_submission

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")

    majority = int(meta_train["label"].mode().iloc[0])
    print(f"Majority class on train: {majority} "
          f"(count {(meta_train['label'] == majority).sum()} / {len(meta_train)})")

    preds = np.full(len(meta_test), majority, dtype=int)
    out = ROOT / "submissions" / "sub01_majority.csv"
    to_submission(meta_test["file_id"].values, preds, str(out))


if __name__ == "__main__":
    main()
