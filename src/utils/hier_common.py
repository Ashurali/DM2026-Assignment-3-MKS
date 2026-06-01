"""Shared helpers for the hierarchical pipeline (Stage 1 coarse + Stage 2 fine).

Why this lives here: the four hierarchical scripts (coarse 3-way, fine
L1-vs-L2, fine L3-v-L4-v-L5, compose-and-blend) all need the same 805-feature
stack and the same fold splits as combo_full_v2. Centralising the build
prevents drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6


def build_feature_blocks(
    include_combo_oof: bool = True,
    include_gmm: bool = False,
    include_gaf_oof: bool = False,
    include_gaf_emb: bool = False,
    include_spec_oof: bool = False,
    include_spec_emb: bool = False,
    include_l1l2_contrast_emb: bool = False,
    include_covariance: bool = False,
    eo_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Replicate combo_full_v2's 805-feature stack, optionally extended with
    the v3 phase A/B/C blocks:

      include_gmm           — 13-col GMM cluster posteriors (12 components + log-likelihood)
      include_gaf_oof       — 6-col GAF CNN OOF probs (stacked)
      include_gaf_emb       — 128-col GAF CNN penultimate embedding
      include_spec_oof      — 6-col Spec CNN OOF probs
      include_spec_emb      — 128-col Spec CNN penultimate embedding
      include_covariance    — 22-col cross-channel covariance features
                              (15 corrs + 6 eigvals + log condition number)

    All extension blocks are optional and OFF by default — v2 callers see the
    same 805 features they always did. v3 callers turn the relevant flags on.

    Returns (X_train, X_test, block_names).
    """
    Xtr_df = pd.read_parquet(ROOT / "data" / "feat_train_none.parquet")
    Xte_df = pd.read_parquet(ROOT / "data" / "feat_test_none.parquet")
    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    eng_train = Xtr_df[feat_cols].values.astype(np.float64)
    eng_test = Xte_df[feat_cols].values.astype(np.float64)

    c22_tr_path = ROOT / "data" / "feat_catch22_train.parquet"
    c22_te_path = ROOT / "data" / "feat_catch22_test.parquet"
    have_c22 = c22_tr_path.exists() and c22_te_path.exists()
    if have_c22:
        c22_tr = pd.read_parquet(c22_tr_path)
        c22_te = pd.read_parquet(c22_te_path)
        c22_cols = [c for c in c22_tr.columns if c != "file_id"]
        c22_train = c22_tr[c22_cols].values.astype(np.float64)
        c22_test = c22_te[c22_cols].values.astype(np.float64)

    # Apply EO mask if provided. The mask covers exactly engineered + catch22
    # in that order, with length len(feat_cols) + len(c22_cols).
    if eo_mask is not None:
        expected_len = len(feat_cols) + (len(c22_cols) if have_c22 else 0)
        if len(eo_mask) != expected_len:
            raise ValueError(
                f"eo_mask length {len(eo_mask)} != engineered+catch22 width {expected_len}"
            )
        eng_mask = eo_mask[: len(feat_cols)]
        eng_train = eng_train[:, eng_mask]
        eng_test = eng_test[:, eng_mask]
        block_names_eng = f"engineered({int(eng_mask.sum())}/{len(feat_cols)} kept)"
        if have_c22:
            c22_mask = eo_mask[len(feat_cols):]
            c22_train = c22_train[:, c22_mask]
            c22_test = c22_test[:, c22_mask]
            block_names_c22 = f"catch22({int(c22_mask.sum())}/{len(c22_cols)} kept)"
    else:
        block_names_eng = f"engineered({len(feat_cols)})"
        if have_c22:
            block_names_c22 = f"catch22({len(c22_cols)})"

    blocks_train = [eng_train]
    blocks_test = [eng_test]
    block_names = [block_names_eng]

    if have_c22:
        blocks_train.append(c22_train)
        blocks_test.append(c22_test)
        block_names.append(block_names_c22)
    else:
        block_names.append("catch22(MISSING_skipped)")

    blocks_train.append(np.load(ROOT / "oof" / "cnn_bilstm_v1_emb_train.npy").astype(np.float64))
    blocks_test.append(np.load(ROOT / "oof" / "cnn_bilstm_v1_emb_test.npy").astype(np.float64))
    block_names.append(f"cnn_emb({blocks_train[-1].shape[1]})")

    blocks_train.append(np.load(ROOT / "oof" / "transformer_v1_emb_train.npy").astype(np.float64))
    blocks_test.append(np.load(ROOT / "oof" / "transformer_v1_emb_test.npy").astype(np.float64))
    block_names.append(f"transformer_emb({blocks_train[-1].shape[1]})")

    for run in ("xgb_v1", "cat_v1", "minirocket_v1"):
        blocks_train.append(np.load(ROOT / "oof" / f"{run}_oof.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / f"{run}_test_probs.npy").astype(np.float64))
        block_names.append(f"oof_{run}(6)")

    if include_combo_oof:
        blocks_train.append(np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_oof.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_test_probs.npy").astype(np.float64))
        block_names.append("oof_combo_full_v2(6)")

    if include_gmm:
        blocks_train.append(np.load(ROOT / "oof" / "gmm_features_train.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "gmm_features_test.npy").astype(np.float64))
        block_names.append(f"gmm({blocks_train[-1].shape[1]})")

    if include_gaf_oof:
        blocks_train.append(np.load(ROOT / "oof" / "gaf_cnn_v1_oof.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "gaf_cnn_v1_test_probs.npy").astype(np.float64))
        block_names.append("oof_gaf_cnn_v1(6)")
    if include_gaf_emb:
        blocks_train.append(np.load(ROOT / "oof" / "gaf_cnn_v1_emb_train.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "gaf_cnn_v1_emb_test.npy").astype(np.float64))
        block_names.append(f"gaf_emb({blocks_train[-1].shape[1]})")

    if include_spec_oof:
        blocks_train.append(np.load(ROOT / "oof" / "spec_cnn_v1_oof.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "spec_cnn_v1_test_probs.npy").astype(np.float64))
        block_names.append("oof_spec_cnn_v1(6)")
    if include_spec_emb:
        blocks_train.append(np.load(ROOT / "oof" / "spec_cnn_v1_emb_train.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "spec_cnn_v1_emb_test.npy").astype(np.float64))
        block_names.append(f"spec_emb({blocks_train[-1].shape[1]})")

    if include_l1l2_contrast_emb:
        blocks_train.append(np.load(ROOT / "oof" / "l1l2_contrast_emb_train.npy").astype(np.float64))
        blocks_test.append(np.load(ROOT / "oof" / "l1l2_contrast_emb_test.npy").astype(np.float64))
        block_names.append(f"l1l2_contrast_emb({blocks_train[-1].shape[1]})")

    if include_covariance:
        cov_tr = pd.read_parquet(ROOT / "data" / "feat_train_covariance.parquet")
        cov_te = pd.read_parquet(ROOT / "data" / "feat_test_covariance.parquet")
        cov_cols = [c for c in cov_tr.columns if c != "file_id"]
        blocks_train.append(cov_tr[cov_cols].values.astype(np.float64))
        blocks_test.append(cov_te[cov_cols].values.astype(np.float64))
        block_names.append(f"covariance({len(cov_cols)})")

    X = np.concatenate(blocks_train, axis=1)
    Xte = np.concatenate(blocks_test, axis=1)
    return X, Xte, block_names


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency per-row weights (sum to len(y))."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (n_classes * counts)
    return inv[y]


def label_to_super(y: np.ndarray) -> np.ndarray:
    """Map 6-class labels to 3-class super-class index.
    L0→0 (C_static), L1∪L2→1 (C_walking), L3∪L4∪L5→2 (C_other).
    """
    s = np.zeros_like(y)
    s[(y == 1) | (y == 2)] = 1
    s[(y == 3) | (y == 4) | (y == 5)] = 2
    return s
