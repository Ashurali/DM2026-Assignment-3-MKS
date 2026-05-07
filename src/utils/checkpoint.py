"""Checkpoint helpers for resumable training.

Used by `cv.cv_score` (per-fold caching) and the LGBM/DL training scripts.
Files live under `checkpoints/<run_name>/...` and are gitignored.

Pattern for callers:
    from src.utils.checkpoint import fold_cache_path, save_fold, load_fold

    cached = load_fold("lgbm_full_v1", k=0)
    if cached is not None:
        preds, probs = cached
    else:
        preds, probs = my_train_fn(X_tr, y_tr, X_va)
        save_fold("lgbm_full_v1", k=0, preds=preds, probs=probs)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"


def run_dir(name: str) -> Path:
    d = CKPT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_cache_path(name: str, k: int) -> Path:
    return run_dir(name) / f"fold_{k}.npz"


def save_fold(name: str, k: int, preds: np.ndarray, probs: np.ndarray) -> None:
    """Persist a single fold's predictions and probabilities atomically.

    Writes to a `.tmp.npz` then atomically renames into place, so an interrupt
    during the write can never leave a half-written file at the cache path.
    """
    final = fold_cache_path(name, k)
    # numpy.savez auto-appends ".npz" if the path doesn't end in it, so we
    # construct the tmp path with the extension already attached.
    tmp = final.with_name(final.stem + ".tmp.npz")
    np.savez(tmp, preds=preds.astype(np.int64), probs=probs.astype(np.float64))
    tmp.replace(final)


def load_fold(name: str, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (preds, probs) for a cached fold, or None if not present."""
    p = fold_cache_path(name, k)
    if not p.exists():
        return None
    d = np.load(p)
    return d["preds"], d["probs"]


def clear_run(name: str) -> int:
    """Delete all checkpoints under `<name>`. Returns count of files removed."""
    d = CKPT_DIR / name
    if not d.exists():
        return 0
    n = 0
    for f in d.iterdir():
        if f.is_file():
            f.unlink()
            n += 1
    return n


def list_runs() -> list[str]:
    if not CKPT_DIR.exists():
        return []
    return sorted([p.name for p in CKPT_DIR.iterdir() if p.is_dir()])


def write_run_status(name: str, payload: dict) -> None:
    (run_dir(name) / "status.json").write_text(json.dumps(payload, indent=2))


def read_run_status(name: str) -> dict | None:
    p = run_dir(name) / "status.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
