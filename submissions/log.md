# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | _pending_ | _pending_ | Pipeline validator; matches EDA §7 floor |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |

LGBM-basic submission row (sub02) will be appended automatically by `src/models/train_lgbm_basic.py` when it runs.
