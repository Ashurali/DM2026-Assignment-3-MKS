# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | -0.0307 | Pipeline validator; gap is structural (test label-1 fraction ≈ 44.3% vs train 42.6%), not a CV leak |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |

LGBM-basic submission row (sub02) will be appended automatically by `src/models/train_lgbm_basic.py` when it runs.
