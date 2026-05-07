# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | -0.0307 | Pipeline validator; gap is structural (test label-1 fraction ≈ 44.3% vs train 42.6%), not a CV leak |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |
| 2026-05-07 | sub02_lgbm_basic | LGBM (60 basic features, class-weighted, GPU) | 0.6795 (fold-mean) / 0.6906 (OOF) | 0.7473 | +0.068 (vs fold-mean) / +0.057 (vs OOF) | Per-class F1: [0.958, 0.8924, **0.1656**, 0.6759, 0.8253, 0.6266]. Label-2 collapse is the headline. Public > CV — GKF is pessimistic, not leaking. |
| 2026-05-07 | sub_lgbm_full_v1 | LGBM full (271 features) | 0.7091 | _pending_ | _pending_ | per-class F1 [0.9644, 0.9024, 0.1731, 0.7127, 0.8961, 0.6777] |
| 2026-05-07 | sub_lgbm_full_smote_v1 | LGBM full (271 features) | 0.7087 | _pending_ | _pending_ | per-class F1 [0.9652, 0.9046, 0.1992, 0.7103, 0.8664, 0.6768]; SMOTE |
| 2026-05-07 | sub_lgbm_full_tuned_v1 _(initial run)_ | LGBM full + Optuna 50 trials | 0.7163 | _superseded_ | — | per-class F1 [0.9647, 0.902, 0.2431, 0.7008, 0.8912, 0.6653]. Replaced by the resumed-Optuna run after checkpointing fix. |
| 2026-05-08 | **sub_lgbm_full_tuned_v1** | LGBM full + Optuna (resumed; SQLite study) | 0.7167 / 0.7282 (OOF) | _pending_ (predicted ~0.7842) | _pending_ | per-class F1 [0.9641, 0.9006, **0.2393**, 0.701, 0.8842, 0.6802]. L2 lifted +0.066 vs v1 by HP bias (smaller min_data_in_leaf, more leaves). |
| 2026-05-08 | **sub_lgbm_full_tuned_v1_tuned** | LGBM full + Optuna + post-hoc threshold (multipliers=[1.495, 0.712, 1.901, 1.804, 0.786, 0.629]) | **0.7350 (OOF)** | _pending_ (predicted ~0.7910) | _pending_ | per-class F1 [0.9627, 0.8892, **0.2749**, 0.7022, **0.8993**, 0.682]. Best LGBM artefact. L2 climbed from 0.166 (sub02) → 0.275 (+66%); total LGBM lift from baseline = +0.044 OOF. |
