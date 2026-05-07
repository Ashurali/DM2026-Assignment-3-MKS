# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | -0.0307 | Pipeline validator; gap is structural (test label-1 fraction ≈ 44.3% vs train 42.6%), not a CV leak |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |
| 2026-05-07 | sub02_lgbm_basic | LGBM (60 basic features, class-weighted, GPU) | 0.6795 (fold-mean) / 0.6906 (OOF) | 0.7473 | +0.068 (vs fold-mean) / +0.057 (vs OOF) | Per-class F1: [0.958, 0.8924, **0.1656**, 0.6759, 0.8253, 0.6266]. Label-2 collapse is the headline. Public > CV — GKF is pessimistic, not leaking. |
| 2026-05-07 | sub_lgbm_full_v1 | LGBM full (271 features) | 0.7091 / 0.7211 (OOF) | _not submitted_ | — | per-class F1 [0.9644, 0.9024, 0.1731, 0.7127, 0.8961, 0.6777]. Skipped in favor of v1_tuned. |
| 2026-05-07 | sub_lgbm_full_smote_v1 | LGBM full (271 features) + SMOTE (target=1500/minority) | 0.7087 / 0.7204 (OOF) | _not submitted_ | — | per-class F1 [0.9652, 0.9046, 0.1992, 0.7103, 0.8664, 0.6768]. L2 +0.026 / L4 −0.030 → net wash. Dropped. |
| 2026-05-08 | **sub_lgbm_full_v1_tuned** | LGBM full + post-hoc threshold tuning (multipliers=[1.745, 0.543, 0.936, 0.402, 0.501, 1.469]) | **0.7253 (OOF)** | **0.7808** | **+0.056** | per-class F1 [0.9643, 0.9022, 0.1974, 0.7169, 0.8961, 0.6750]. Threshold tuning validated end-to-end (+0.0335 LB over sub02; CV→LB gap exactly matches sub02 → GKF pessimism is a stable +0.056 calibration). |
