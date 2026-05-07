"""Tests for checkpoint helpers + cv_score resumability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.checkpoint import save_fold, load_fold, clear_run, fold_cache_path
from src.utils.cv import cv_score


def test_save_and_load_fold(tmp_path, monkeypatch):
    """Roundtrip through the npz cache."""
    # Redirect CKPT_DIR so we don't pollute the repo
    import src.utils.checkpoint as cp
    monkeypatch.setattr(cp, "CKPT_DIR", tmp_path / "checkpoints")

    preds = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    probs = np.random.rand(6, 6)
    save_fold("toy_run", k=2, preds=preds, probs=probs)

    loaded = load_fold("toy_run", k=2)
    assert loaded is not None
    p_loaded, q_loaded = loaded
    np.testing.assert_array_equal(preds, p_loaded)
    np.testing.assert_allclose(probs, q_loaded)

    # Missing fold returns None
    assert load_fold("toy_run", k=99) is None


def test_clear_run(tmp_path, monkeypatch):
    import src.utils.checkpoint as cp
    monkeypatch.setattr(cp, "CKPT_DIR", tmp_path / "checkpoints")
    save_fold("ephemeral", k=0, preds=np.array([0], dtype=np.int64), probs=np.array([[1.0]]))
    save_fold("ephemeral", k=1, preds=np.array([0], dtype=np.int64), probs=np.array([[1.0]]))
    n_removed = clear_run("ephemeral")
    assert n_removed == 2
    assert load_fold("ephemeral", k=0) is None


def test_cv_score_resumes_from_checkpoint(tmp_path, monkeypatch):
    """Simulate an interrupted run: pre-populate fold 0 cache, then verify
    that cv_score loads it instead of calling fit_predict for that fold."""
    import src.utils.checkpoint as cp
    monkeypatch.setattr(cp, "CKPT_DIR", tmp_path / "checkpoints")

    rng = np.random.default_rng(42)
    n_per_group = 30
    n_groups = 10
    n_classes = 3
    X_parts, y_parts, g_parts = [], [], []
    for g in range(n_groups):
        for c in range(n_classes):
            mu = np.array([c * 5.0, -c * 5.0, c * 2.0, 0.0])
            X_parts.append(rng.normal(loc=mu, scale=0.3, size=(n_per_group // n_classes, 4)))
            y_parts.append(np.full(n_per_group // n_classes, c))
            g_parts.append(np.full(n_per_group // n_classes, g))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(g_parts)

    from sklearn.linear_model import LogisticRegression
    call_count = {"n": 0}

    def fit_predict(Xtr, ytr, Xva):
        call_count["n"] += 1
        clf = LogisticRegression(max_iter=500, random_state=42).fit(Xtr, ytr)
        return clf.predict(Xva), clf.predict_proba(Xva)

    # First pass — populates the cache for all 5 folds
    mean1, _, _, _ = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=n_classes,
        verbose=False, checkpoint_name="toy_resume",
    )
    assert call_count["n"] == 5  # all 5 folds called fit_predict

    # Second pass — should hit the cache; fit_predict NOT called
    call_count["n"] = 0
    mean2, _, _, _ = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=n_classes,
        verbose=False, checkpoint_name="toy_resume",
    )
    assert call_count["n"] == 0, f"expected 0 fresh fits on resume, got {call_count['n']}"
    assert abs(mean1 - mean2) < 1e-9


def test_cv_score_without_checkpoint_unchanged(tmp_path):
    """Sanity: when checkpoint_name=None, behaviour is unchanged from before."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (rng.normal(size=60) > 0).astype(int) + (rng.normal(size=60) > 0.5).astype(int)
    groups = np.repeat(np.arange(10), 6)

    from sklearn.linear_model import LogisticRegression

    def fit_predict(Xtr, ytr, Xva):
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        # Make sure probs has 3 columns (or pad if a class missing)
        probs = clf.predict_proba(Xva)
        if probs.shape[1] < 3:
            pad = np.zeros((probs.shape[0], 3 - probs.shape[1]))
            probs = np.concatenate([probs, pad], axis=1)
        return clf.predict(Xva), probs

    mean, std, oof_p, oof_pr = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=3, verbose=False,
    )
    assert oof_p.shape == y.shape
    assert oof_pr.shape == (60, 3)
