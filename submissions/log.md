# Submission Log

## Headline submissions

| Date | Version | Model | OOF F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | −0.031 | Pipeline validator; gap is structural (test L1 fraction differs), not a CV leak. |
| 2026-05-07 | sub02_lgbm_basic | LGBM (60 basic features) | 0.6906 | 0.7473 | +0.057 | Floor for the engineered-feature track. |
| 2026-05-08 | **sub_lgbm_full_v1_tuned** | LGBM full (271) + post-hoc threshold | 0.7253 | **0.7808** | +0.056 | Threshold tuning validated end-to-end. |
| 2026-05-08 | sub_lgbm_full_tuned_v1 | LGBM full + Optuna 50 trials | 0.7282 | 0.7735 | +0.045 | Optuna OOF-overfit; gap shrank from +0.056 to +0.045. |
| 2026-05-08 | **sub_lgbm_full_tuned_v1_tuned** | LGBM full + Optuna + post-hoc threshold | 0.7350 | **0.7816** | +0.047 | **Best LGBM artefact.** Threshold tuning's +0.008 transferred fully; Optuna's +0.007 OOF mostly didn't transfer. |

## Phase 5 — CNN-BiLSTM track

All three OOF measured on the same 5-fold GroupKFold by user_id.

| Date | Version | Config | OOF F1 | Per-class F1 (L0/L1/L2/L3/L4/L5) | Notes |
|---|---|---|---|---|---|
| 2026-05-07 | **sub_cnn_bilstm_v1** | default augs (p_rot=0.5, mixup α=0.2, lr 1e-3) | **0.6712** | 0.936 / 0.792 / 0.233 / 0.666 / 0.748 / 0.651 | **CNN endpoint.** L4 collapse (vs LGBM 0.90) is the headline. |
| 2026-05-07 | sub_cnn_bilstm_v2 | re-run, slight aug variation | 0.6666 | 0.946 / 0.824 / 0.252 / 0.646 / 0.644 / 0.689 | Different per-class profile; useful for averaging if blended with v1. |
| 2026-05-07 | sub_cnn_bilstm_v3 | p_rot=0, mixup=0, lr 5e-4, batch 128 | 0.6333 | 0.936 / 0.807 / 0.219 / 0.676 / 0.637 / 0.523 | Hypothesis (rotation hurts L4) **disproved** — v3 is worse on L4 *and* L5 collapses. CNN ceiling is architectural. |

## Reference / not submitted

| Version | Model | OOF F1 | Why not submitted |
|---|---|---|---|
| _ref_ LR baseline | LR balanced on 6 column-means | 0.5315 | EDA §7 anchor, never submitted. |
| sub_lgbm_full_v1 | LGBM full 271 (no tune) | 0.7211 | Skipped in favor of v1_tuned (+0.004 OOF for free). |
| sub_lgbm_full_smote_v1 | LGBM full + SMOTE | 0.7204 | Net wash (L2 +0.026 / L4 −0.030). Dropped. |

## Phase 4 ablations (CV-only, not submitted)

11 feature-group ablations. Full table in `reports/ablation_features.md`.
Top-3 most-important groups by |ΔCV F1|: **jerk** (−0.0088), **subwindow** (−0.0068), **zerocross** (−0.0060). All others below the +0.005 noise floor (calibrated by `quality`-group ablation showing −0.0036 despite being constant features).

## Phase 8 — Blend (Phase-8 step 1: scipy-simplex / LR-meta + post-hoc threshold)

All blends use OOF-fitted weights; simplex method won every comparison (LR-meta consistently 0.01–0.04 lower).

| Blend | Inputs | Simplex weights (top 3) | Blend OOF | + threshold | Notes |
|---|---|---|---|---|---|
| v1 | tuned_v1 + cnn v1 | 92% / 8% | 0.7315 | 0.7357 | minimal CNN weight |
| v2 | tuned_v1 + v1 + cnn v1/v2 | 70% / 17% / 10% | 0.7301 | 0.7419 | LGBM-v1 untuned earned 17% — its non-Optuna profile decorrelates |
| **v3** | + cnn v3 | 70% / 17% / 9% | 0.7294 | **0.7426** | **best.** v3's degraded-but-different errors add a smidge more decorrelation |
| v4 | tuned_v1 + v1 + smote_v1 + cnn v1/v2 | 67% / 9% / 10% | 0.7311 | 0.7420 | SMOTE variant adds nothing meaningful |

**Saturation around OOF 0.742.** Adding more base models past 5 doesn't help — simplex weights dilute below ~3% per CNN, no extra signal.

**Predicted LB for blend v3:** 0.788–0.792 (using +0.045–0.050 gap from Optuna-heavy weighting). Should beat the current 0.7816 LB endpoint.

## Calibration notes

- **CV→LB gap for non-tuned models:** +0.056 (sub02, v1_tuned).
- **CV→LB gap for Optuna-tuned models:** +0.045 (tuned_v1, tuned_v1_tuned). Optuna's HP search slightly OOF-overfits.
- **Predicted blend OOF (achieved):** 0.7426. Predicted LB: 0.788–0.792.
