"""Group-aware CV harness used by every model from now on.

Every Phase-3+ model imports `make_folds`, `cv_score`, and `to_submission`
from here so the CV scheme is identical across tracks (LGBM, CNN-BiLSTM,
Transformer) and the OOF probabilities can be blended at the end.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score


def make_folds(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of `(train_idx, val_idx)` tuples for GroupKFold.

    GroupKFold doesn't take a random seed — its split is deterministic given
    the input `groups` array. Results are stable across runs.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def cv_score(
    fit_predict: FitPredict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_classes: int = 6,
    verbose: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run group-aware CV and return mean/std F1-macro plus OOF preds and probs.

    Parameters
    ----------
    fit_predict : callable(X_tr, y_tr, X_va) -> (preds_va, probs_va)
        The training/inference function. `preds_va` is a 1-D array of predicted
        class indices; `probs_va` is a 2-D array of shape (n_val, n_classes).
        Sklearn-style or pytorch-style models can both be wrapped to fit.
    X, y, groups : array-like
        Feature matrix, labels, and group ids (e.g. user_id) of length N.
    n_splits : int
        Number of folds (default 5).
    n_classes : int
        Number of target classes (default 6 for this competition).
    verbose : bool
        If True, print per-fold and aggregate F1.

    Returns
    -------
    mean_f1, std_f1, oof_preds (N,), oof_probs (N, n_classes)
    """
    folds = make_folds(groups, n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=np.int64)
    oof_probs = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_f1: list[float] = []
    for k, (tr, va) in enumerate(folds):
        preds, probs = fit_predict(X[tr], y[tr], X[va])
        oof_preds[va] = preds
        oof_probs[va] = probs
        f = float(f1_score(y[va], preds, average="macro"))
        fold_f1.append(f)
        if verbose:
            print(f"  Fold {k}: F1-macro = {f:.4f}")
    mean = float(np.mean(fold_f1))
    std = float(np.std(fold_f1))
    if verbose:
        print(f"  CV F1-macro = {mean:.4f} ± {std:.4f}")
    return mean, std, oof_preds, oof_probs


def to_submission(file_ids: np.ndarray, preds: np.ndarray, path: str) -> pd.DataFrame:
    """Write a Kaggle submission CSV with columns `Id,Label`.

    `file_ids` and `preds` must be aligned 1-D arrays of the same length.
    """
    df = pd.DataFrame({"Id": np.asarray(file_ids).astype(int), "Label": np.asarray(preds).astype(int)})
    df.to_csv(path, index=False)
    print(f"Wrote submission: {path}  ({len(df)} rows)")
    return df
