"""Tests for the GroupKFold CV harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where pytest is invoked from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.cv import make_folds, cv_score, to_submission  # noqa: E402


def test_make_folds_deterministic():
    """Same groups + n_splits → identical fold indices on every call."""
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] * 3)
    folds_a = make_folds(groups, n_splits=5)
    folds_b = make_folds(groups, n_splits=5)
    assert len(folds_a) == 5
    for (tra, vaa), (trb, vab) in zip(folds_a, folds_b):
        assert np.array_equal(tra, trb)
        assert np.array_equal(vaa, vab)


def test_groupkfold_no_leak():
    """No group should appear in both train and val of the same fold."""
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    for tr, va in make_folds(groups, n_splits=5):
        assert set(groups[tr]).isdisjoint(set(groups[va])), "user leak!"


def test_cv_score_shapes_and_value(tmp_path):
    """Smoke test: cv_score returns correct shapes; mean F1 is reasonable on a separable toy problem."""
    rng = np.random.default_rng(42)
    n_per_group = 30
    n_groups = 10
    n_classes = 3
    X_parts, y_parts, g_parts = [], [], []
    for g in range(n_groups):
        # 3 well-separated clusters in 4-D so a tiny LR can solve it
        for c in range(n_classes):
            mu = np.array([c * 5.0, -c * 5.0, c * 2.0, 0.0])
            X_parts.append(rng.normal(loc=mu, scale=0.3, size=(n_per_group // n_classes, 4)))
            y_parts.append(np.full(n_per_group // n_classes, c))
            g_parts.append(np.full(n_per_group // n_classes, g))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    from sklearn.linear_model import LogisticRegression

    def fit_predict(Xtr, ytr, Xva):
        clf = LogisticRegression(max_iter=500, random_state=42).fit(Xtr, ytr)
        return clf.predict(Xva), clf.predict_proba(Xva)

    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=n_classes, verbose=False
    )
    assert oof_preds.shape == y.shape
    assert oof_probs.shape == (len(y), n_classes)
    assert mean > 0.95, f"toy problem should be near-perfect; got {mean}"


def test_to_submission_writes_expected_columns(tmp_path):
    file_ids = np.array([11021, 11022, 11023])
    preds = np.array([0, 1, 2])
    out = tmp_path / "sub.csv"
    df = to_submission(file_ids, preds, str(out))
    assert list(df.columns) == ["Id", "Label"]
    import pandas as pd
    df2 = pd.read_csv(out)
    assert df2.shape == (3, 2)
    assert df2["Id"].tolist() == [11021, 11022, 11023]
    assert df2["Label"].tolist() == [0, 1, 2]
