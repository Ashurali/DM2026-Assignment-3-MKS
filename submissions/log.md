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

## Round-1 expansion blends (Tier-1 complete, May 8)

After adding XGBoost, CatBoost, MiniRocket, Transformer, catch22 features, and stacked DL embeddings, multiple blend variants were tried:

| Variant | Method | Inputs | Blend OOF | + threshold | Predicted LB |
|---|---|---|---|---|---|
| `blend_top4` | **LR-meta** | combo + xgb + cat + lgbm_tuned | 0.7585 | **0.7760** | **0.821–0.832** ← winner |
| `lgbm_combo_combo_full_tuned` | single | 271 + catch22 + cnn_emb + tx_emb → LGBM | — | 0.7725 | 0.818–0.828 |
| `blend_top4plusdl` | simplex | top4 + 2 CNNs + Transformer | 0.7642 | 0.7678 | 0.813–0.824 |
| `blend_big_v1` | simplex | 9 models, all bases | 0.7591 | 0.7676 | 0.813–0.824 |
| `blend_top5` | simplex | top4 + cnn_v1 | 0.7635 | 0.7675 | 0.812–0.823 |
| `blend_top3` | simplex | combo + xgb + cat | 0.7621 | 0.7656 | 0.811–0.822 |
| `blend_gbdts_only` | simplex | 6 GBDT-family models | 0.7551 | 0.7658 | 0.811–0.822 |

**Key finding:** Adding DL models (CNN/Transformer) to the blend HURTS — they're useful *only* via stacked DL embeddings into LGBM (the `combo` model), not as direct blend inputs. The combo model alone is now so strong that further blending only helps with the right small set of decorrelated GBDTs.

Per-class for winner (`blend_top4_tuned`):
- L0=0.965, L1=0.901, **L2=0.331** (+99% from sub02's 0.166), L3=0.762, L4=0.919, L5=0.779

## Round-2 — combo_full_v2 with stacked base-model OOFs (May 8)

Added 18 columns of stacked OOF probs (XGB + CatBoost + MiniRocket × 6 classes)
to combo_full's existing features. New combo trained on **805 features** total.

Pair specialists (binary L1-vs-L2 / L1-vs-L5) were tried first but pair-AUC
came out 0.4–0.6 (random) — the same 271 features fail in binary mode that
fail in multiclass, so specialists added nothing. Dropped from the plan.

| Variant | OOF | + cal + thresh | Predicted LB |
|---|---|---|---|
| `combo_full_v2` (271 + catch22 + DL emb + 18 OOFs) | 0.7733 | **0.7910** | **0.836–0.847** ← 🥇 NEW BEST |
| `blend_both_combos` (combo_v2 + combo_full) | 0.7812 (LR-meta) | 0.7847 | 0.830–0.841 |
| `combo_full` (without OOFs) | 0.7687 | 0.7839 | 0.829–0.840 |
| `blend_top4` (cal off, threshold) | 0.7585 | 0.7760 | 0.821–0.832 |

Per-class for the new winner (`combo_full_v2` + cal + thresh):
- L0=0.966, L1=0.908, **L2=0.393** (+19 pts from previous 0.275!), L3=0.768, L4=0.928, L5=0.783

L2 trajectory: 0.166 (sub02) → 0.275 (combo) → 0.331 (blend_top4) → 0.374 (combo+thresh) → **0.393 (combo_v2+cal+thresh)**.

**Key finding:** Once the combo model includes stacked OOF probs from
XGB/CAT/MiniRocket, blending it with anything else (including with itself
in different forms) HURTS. The GBDT in combo extracts non-linear interactions
between per-class scores that scalar simplex/LR-meta blending can't match.
combo_full_v2 alone IS the ensemble.

## Calibration notes

- **CV→LB gap for non-tuned models:** +0.056 (sub02, v1_tuned).
- **CV→LB gap for Optuna-tuned models:** +0.045 (tuned_v1, tuned_v1_tuned). Optuna's HP search slightly OOF-overfits.
- **Predicted blend OOF (achieved):** 0.7426. Predicted LB: 0.788–0.792.
| 2026-05-07 | sub_xgb_v1 | XGB (271 features, class-weighted) | 0.7165 (fold-mean) / 0.7290 (OOF) | _pending_ | _pending_ | per-class F1 [0.9659, 0.9045, 0.2416, 0.7093, 0.8826, 0.6701] |
| 2026-05-07 | sub_cat_v1 | CAT (271 features, class-weighted) | 0.7128 (fold-mean) / 0.7220 (OOF) | _pending_ | _pending_ | per-class F1 [0.9609, 0.8834, 0.2715, 0.6902, 0.8905, 0.6352] |
| 2026-05-08 | sub_minirocket_v1 | MiniRocket + LightGBM (~9996 random conv features) | 0.6693 (fold-mean) / 0.6785 (OOF) | _pending_ | _pending_ | per-class F1 [0.9677, 0.8927, 0.0515, 0.6929, 0.8864, 0.5795] |
| 2026-05-08 | sub_transformer_v1 | Transformer encoder (4 layers, d_model=128) | 0.5933 (fold-mean) / 0.6028 (OOF) | _pending_ | _pending_ | per-class F1 [0.8881, 0.6943, 0.2132, 0.6471, 0.7492, 0.4248] |
| 2026-05-08 | sub_lgbm_combo_combo_full | LGBM combo (engineered(271) + catch22(132) + cnn_emb(256) + transformer_emb(128)) | 0.7620 (fold-mean) / 0.7687 (OOF) | _pending_ | _pending_ | per-class F1 [0.9659, 0.9101, 0.2552, 0.7691, 0.9286, 0.7835] |
| 2026-05-08 | sub_lgbm_combo_combo_v1 | LGBM combo (engineered(271)) | 0.7094 (fold-mean) / 0.7199 (OOF) | _pending_ | _pending_ | per-class F1 [0.9648, 0.9031, 0.1747, 0.7141, 0.8889, 0.674] |
| 2026-05-08 | sub_lgbm_combo_combo_full_v2 | LGBM combo (engineered(271) + catch22(132) + cnn_emb(256) + transformer_emb(128) + oof_xgb_v1(6) + oof_cat_v1(6) + oof_minirocket_v1(6)) | 0.7666 (fold-mean) / 0.7733 (OOF) | _pending_ | _pending_ | per-class F1 [0.9682, 0.9112, 0.2811, 0.7745, 0.9286, 0.7763] |
