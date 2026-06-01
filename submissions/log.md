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

## LB results — May 8 round-2 confirmations

Three combo-track LB results revealed a structural issue:

| Submission | OOF | LB | Gap | Status |
|---|---|---|---|---|
| `combo_full` RAW (no post-hoc) | 0.7687 | **0.7568** | **−0.012** | broken transfer |
| `combo_full_v2` RAW (no post-hoc) | 0.7733 | **0.7577** | **−0.016** | broken transfer |
| `combo_full + cal + thresh` | 0.7839 | **0.7924** | +0.008 | **best validated** ✅ |

**Critical finding:** RAW combo models have NEGATIVE LB gaps. cal+thresh
isn't OOF-overfitting — it's correcting a real **train-test feature
distribution shift in the DL embeddings.**

The combo model uses CNN-BiLSTM and Transformer penultimate-layer
activations as features. Those embeddings come from a final-model
retrained on the FULL train set, applied to both train and test:
- Train embedding for sample i: from a model that MEMORIZED i during
  training (DL final layers overfit)
- Test embedding for sample i: pure generalization (sample never seen)

So combo LGBM learns "memorized-pattern → label" but sees "clean-pattern"
at test. cal+thresh recalibrates predictions against OOF (which represents
unseen-data behavior), aligning test predictions with test-time reality.

**Decision rule: combo-style models with DL embeddings MUST use cal+thresh.**
Without it, the gap is structurally negative.

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
| 2026-05-08 | sub_cnn_bilstm_v1_pfn | CNN-BiLSTM (raw 6×300 + augs + mixup) | 0.5802 (fold-mean) / 0.5853 (OOF) | _pending_ | _pending_ | per-class F1 [0.9344, 0.7327, 0.1633, 0.5931, 0.7419, 0.3465] |
| 2026-05-08 | sub_cnn_bilstm_dann_dann_v1 | CNN-BiLSTM + DANN (max_λ=0.5) | 0.5846 (fold-mean) / 0.5946 (OOF) | _pending_ | _pending_ | per-class F1 [0.9391, 0.7295, 0.1843, 0.5896, 0.7273, 0.3979] |
| 2026-05-08 | sub_sicl_sicl_v1 | SICL pretrain (q_s=0.5, τ=0.1) + linear FT | 0.4657 (fold-mean) / 0.4651 (OOF) | _pending_ | _pending_ | per-class F1 [0.9353, 0.7245, 0.141, 0.4737, 0.2778, 0.2385] |
| 2026-05-08 | sub_lgbm_combo_combo_full_robust | LGBM combo (engineered(271) + catch22(132) + cnn_emb(256) + transformer_emb(128) + oof_cnn_bilstm_dann_dann_v1(6) + oof_sicl_sicl_v1(6)) | 0.6615 (fold-mean) / 0.6673 (OOF) | _pending_ | _pending_ | per-class F1 [0.966, 0.8925, 0.0055, 0.6895, 0.8797, 0.5706] |
| 2026-05-09 | sub_lgbm_combo_combo_full_v2_ms | Multi-seed combo_full_v2 (seeds=[7, 17, 23, 41, 99]) | 0.6900 (OOF) | _pending_ | _pending_ | per-class F1 [0.9689, 0.9026, 0.0525, 0.693, 0.8772, 0.646] |
| 2026-05-09 | sub_lgbm_combo_combo_full_v3 | LGBM combo (engineered(271) + catch22(132) + cnn_emb(256) + transformer_emb(128) + oof_xgb_v1(6) + oof_cat_v1(6) + oof_minirocket_v1(6) + oof_cnn_bilstm_v1_tent(6)) | 0.6795 (fold-mean) / 0.6898 (OOF) | _pending_ | _pending_ | per-class F1 [0.9688, 0.9031, 0.0628, 0.696, 0.8641, 0.6441] |
| 2026-05-09 | sub_lgbm_combo_combo_full_v2_pl | Pseudo-label combo (conf=0.85, n=5430) | 0.6922 (fold-mean) / 0.7013 (OOF) | _pending_ | _pending_ | per-class F1 [0.9688, 0.9061, 0.0918, 0.7044, 0.8676, 0.6687] |
| 2026-05-09 | sub_lgbm_combo_combo_full_v2_pl | Pseudo-label combo (conf=0.75, n=5787) | 0.7004 (fold-mean) / 0.7077 (OOF) | _pending_ | _pending_ | per-class F1 [0.9679, 0.9058, 0.1089, 0.7047, 0.8741, 0.6845] |
| 2026-05-10 | sub_hier_blend_cal_thresh | Hierarchical (coarse 3-way + L1vL2 + L3L4L5) blended α=0.88 with combo_full_v2, cal+thresh | 0.7844 (OOF) | _pending_ | _pending_ | per-class F1 [0.9656, 0.9064, 0.3851, 0.7565, 0.9203, 0.7728] |
| 2026-05-10 | sub_hier_p2_cal_thresh | hier α-sweep blend (α=0.00) + cal+thresh | 0.7645 (OOF) | _pending_ | _pending_ | per-class F1 [0.9627, 0.9003, 0.3147, 0.7411, 0.9129, 0.7552] |
| 2026-05-10 | sub_hier_a020_cal_thresh | hier α-sweep blend (α=0.20) + cal+thresh | 0.7740 (OOF) | _pending_ | _pending_ | per-class F1 [0.9671, 0.9066, 0.3373, 0.7491, 0.9225, 0.7612] |
| 2026-05-10 | sub_hier_a040_cal_thresh | hier α-sweep blend (α=0.40) + cal+thresh | 0.7794 (OOF) | _pending_ | _pending_ | per-class F1 [0.967, 0.9086, 0.3564, 0.7589, 0.9225, 0.7632] |
| 2026-05-10 | sub_hier_a060_cal_thresh | hier α-sweep blend (α=0.60) + cal+thresh | 0.7802 (OOF) | _pending_ | _pending_ | per-class F1 [0.9662, 0.9064, 0.3596, 0.7634, 0.9191, 0.7665] |
| 2026-05-10 | sub_hier_v2_p2_only_cal_thresh | hier multi-seed v2 + XGB partner (α=0.000) + cal+thresh | 0.7659 (OOF) | _pending_ | _pending_ | per-class F1 [0.9653, 0.9024, 0.3024, 0.7445, 0.9225, 0.7585] |
| 2026-05-10 | sub_hier_v2_blend_cal_thresh | hier multi-seed v2 + XGB partner (α=0.894) + cal+thresh | 0.7808 (OOF) | _pending_ | _pending_ | per-class F1 [0.9669, 0.9069, 0.354, 0.7671, 0.9191, 0.7709] |
| 2026-05-10 | sub_hier_v2_a088_cal_thresh | hier multi-seed v2 + XGB partner (α=0.880) + cal+thresh | 0.7806 (OOF) | _pending_ | _pending_ | per-class F1 [0.9666, 0.9071, 0.3498, 0.7693, 0.9191, 0.7717] |
| 2026-05-10 | sub_hier_v3_p2_only_cal_thresh | hier v3 (extended features: GMM + GAF + Spec) α=0.000 cal+thresh | 0.7670 (OOF) | _pending_ | _pending_ | per-class F1 [0.9674, 0.9025, 0.313, 0.747, 0.9111, 0.7608] |
| 2026-05-10 | sub_hier_v3_blend_cal_thresh | hier v3 (extended features: GMM + GAF + Spec) α=0.764 cal+thresh | 0.7799 (OOF) | _pending_ | _pending_ | per-class F1 [0.9673, 0.9068, 0.3563, 0.7651, 0.9197, 0.7644] |
| 2026-05-10 | sub_hier_v3_a088_cal_thresh | hier v3 (extended features: GMM + GAF + Spec) α=0.880 cal+thresh | 0.7795 (OOF) | _pending_ | _pending_ | per-class F1 [0.967, 0.9062, 0.3478, 0.7681, 0.9236, 0.7641] |
| 2026-05-10 | sub_hier_clean_strict_a088_cal_thresh | hier with cleanlab cleanup (strict) α=0.88 cal+thresh | 0.7852 (OOF) | _pending_ | _pending_ | per-class F1 [0.9676, 0.9088, 0.3717, 0.7713, 0.9236, 0.7683] |
| 2026-05-10 | sub_hier_clean_med_a088_cal_thresh | hier with cleanlab cleanup (med) α=0.88 cal+thresh | 0.7848 (OOF) | _pending_ | _pending_ | per-class F1 [0.9676, 0.9087, 0.3763, 0.7707, 0.9164, 0.7691] |
| 2026-05-10 | sub_hier_clean_aggressive_a088_cal_thresh | hier with cleanlab cleanup (aggressive) α=0.88 cal+thresh | 0.7846 (OOF) | _pending_ | _pending_ | per-class F1 [0.9671, 0.9077, 0.3714, 0.769, 0.9203, 0.7721] |
| 2026-05-10 | sub_hier_v4_p2_only_cal_thresh | hier v4 (L1↔L2 contrastive emb, MS+XGB) α=0.000 cal+thresh | 0.7678 (OOF) | _pending_ | _pending_ | per-class F1 [0.9666, 0.9041, 0.3204, 0.7436, 0.9151, 0.7567] |
| 2026-05-10 | sub_hier_v4_blend_cal_thresh | hier v4 (L1↔L2 contrastive emb, MS+XGB) α=0.742 cal+thresh | 0.7830 (OOF) | _pending_ | _pending_ | per-class F1 [0.9669, 0.9061, 0.3727, 0.7561, 0.9203, 0.7758] |
| 2026-05-10 | sub_hier_v4_a088_cal_thresh | hier v4 (L1↔L2 contrastive emb, MS+XGB) α=0.880 cal+thresh | 0.7856 (OOF) | _pending_ | _pending_ | per-class F1 [0.9675, 0.9079, 0.3769, 0.7664, 0.9191, 0.7758] |
| 2026-05-30 | sub_gru_evidential_v1 | CNN-BiGRU + Evidential Alignment (KDD25) | 0.6214 (fold-mean) / 0.6254 (OOF) | _pending_ | _pending_ | per-class F1 [0.9437, 0.7797, 0.2176, 0.5918, 0.6006, 0.6192] |
| 2026-05-30 | sub_gru_evidential_v2 | CNN-BiGRU + Evidential Alignment (KDD25) | 0.6438 (fold-mean) / 0.6527 (OOF) | _pending_ | _pending_ | per-class F1 [0.9461, 0.7865, 0.2407, 0.6183, 0.6806, 0.6441] |
| 2026-05-30 | sub_gru_v1 | CNN-BiGRU end-to-end (no EA) | 0.6521 (fold-mean) / 0.6493 (OOF) | _pending_ | _pending_ | per-class F1 [0.9345, 0.805, 0.2211, 0.6623, 0.6742, 0.599] |
| 2026-05-30 | sub_gru_pn | CNN-BiGRU end-to-end (no EA) | 0.6797 (fold-mean) / 0.6851 (OOF) | _pending_ | _pending_ | per-class F1 [0.945, 0.816, 0.2264, 0.6672, 0.8333, 0.6225] |
