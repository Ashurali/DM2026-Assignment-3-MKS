# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | -0.0307 | Pipeline validator; gap is structural (test label-1 fraction ≈ 44.3% vs train 42.6%), not a CV leak |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |
| 2026-05-07 | sub02_lgbm_basic | LGBM (60 basic features, class-weighted, GPU) | 0.6795 (fold-mean) / 0.6906 (OOF) | 0.7473 | +0.068 (vs fold-mean) / +0.057 (vs OOF) | Per-class F1: [0.958, 0.8924, **0.1656**, 0.6759, 0.8253, 0.6266]. Label-2 collapse is the headline. Public > CV — GKF is pessimistic, not leaking. |
| 2026-05-07 | sub_lgbm_full_v1 | LGBM full (271 features) | 0.7091 | _pending_ | _pending_ | per-class F1 [0.9644, 0.9024, 0.1731, 0.7127, 0.8961, 0.6777] |
| 2026-05-07 | sub_lgbm_full_smote_v1 | LGBM full (271 features) | 0.7087 | _pending_ | _pending_ | per-class F1 [0.9652, 0.9046, 0.1992, 0.7103, 0.8664, 0.6768]; SMOTE |
| 2026-05-08 | **sub_lgbm_full_tuned_v1** | LGBM full + Optuna 50 trials | 0.7167 / 0.7282 (OOF) | **0.7735** | **+0.045** | per-class F1 [0.9641, 0.9006, **0.2393**, 0.701, 0.8842, 0.6802]. Optuna lifted L2 by +0.066 vs v1; gap shrank from 0.056 → 0.045 (Optuna OOF-overfitting). |
| 2026-05-08 | **sub_lgbm_full_tuned_v1_tuned** | LGBM full + Optuna + post-hoc threshold (multipliers=[1.495, 0.712, 1.901, 1.804, 0.786, 0.629]) | **0.7350 (OOF)** | **0.7816** | **+0.047** | per-class F1 [0.9627, 0.8892, **0.2749**, 0.7022, **0.8993**, 0.682]. **Best LGBM artefact.** Threshold tuning's +0.008 LB lift transferred fully (robust); Optuna's +0.007 OOF mostly didn't transfer to LB. |
| 2026-05-07 | sub_lgbm_full_abl_no_fft | LGBM full (247 features) | 0.7049 | _pending_ | _pending_ | per-class F1 [0.9651, 0.9024, 0.1395, 0.7129, 0.8944, 0.6835]; excluded ['fft'] |
| 2026-05-07 | sub_lgbm_full_abl_no_autocorr | LGBM full (265 features) | 0.7060 | _pending_ | _pending_ | per-class F1 [0.9643, 0.9022, 0.1485, 0.7123, 0.8913, 0.679]; excluded ['autocorr'] |
| 2026-05-07 | sub_lgbm_full_abl_no_subwindow | LGBM full (211 features) | 0.7022 | _pending_ | _pending_ | per-class F1 [0.964, 0.9013, 0.1386, 0.7024, 0.8945, 0.6768]; excluded ['subwindow'] |
| 2026-05-07 | sub_lgbm_full_abl_no_gravity | LGBM full (260 features) | 0.7064 | _pending_ | _pending_ | per-class F1 [0.9649, 0.9029, 0.1542, 0.7094, 0.8953, 0.6806]; excluded ['gravity'] |
| 2026-05-07 | sub_lgbm_full_abl_no_jerk | LGBM full (247 features) | 0.7002 | _pending_ | _pending_ | per-class F1 [0.9652, 0.9023, 0.1218, 0.7153, 0.8889, 0.6746]; excluded ['jerk'] |
| 2026-05-07 | sub_lgbm_full_abl_no_crossaxis | LGBM full (265 features) | 0.7054 | _pending_ | _pending_ | per-class F1 [0.9654, 0.9016, 0.1701, 0.7064, 0.8817, 0.6807]; excluded ['crossaxis'] |
| 2026-05-07 | sub_lgbm_full_abl_no_zerocross | LGBM full (263 features) | 0.7030 | _pending_ | _pending_ | per-class F1 [0.9643, 0.9003, 0.1471, 0.713, 0.8961, 0.6659]; excluded ['zerocross'] |
| 2026-05-07 | sub_lgbm_full_abl_no_per_file_norm | LGBM full (253 features) | 0.7081 | _pending_ | _pending_ | per-class F1 [0.9648, 0.9023, 0.1701, 0.7078, 0.8929, 0.6762]; excluded ['per_file_norm'] |
| 2026-05-07 | sub_lgbm_full_abl_no_magnitude | LGBM full (243 features) | 0.7050 | _pending_ | _pending_ | per-class F1 [0.9654, 0.9008, 0.1636, 0.7067, 0.8832, 0.6799]; excluded ['magnitude'] |
| 2026-05-07 | sub_lgbm_full_abl_no_basic_stats | LGBM full (187 features) | 0.7041 | _pending_ | _pending_ | per-class F1 [0.9645, 0.9003, 0.1157, 0.7123, 0.911, 0.6756]; excluded ['basic_stats'] |
| 2026-05-07 | sub_lgbm_full_abl_no_quality | LGBM full (269 features) | 0.7054 | _pending_ | _pending_ | per-class F1 [0.9642, 0.902, 0.1563, 0.7115, 0.8889, 0.6798]; excluded ['quality'] |
