# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 2026-05-07 | sub01_majority | Majority class (label 1) | 0.1331 | 0.1024 | -0.0307 | Pipeline validator; gap is structural (test label-1 fraction ≈ 44.3% vs train 42.6%), not a CV leak |
| _ref_ | _ | LR balanced on 6 column-means (CV-only, not submitted) | 0.5315 | — | — | EDA §7 trivial-linear anchor |
| 2026-05-07 | sub02_lgbm_basic | LGBM (60 basic features, class-weighted, GPU) | 0.6795 (fold-mean) / 0.6906 (OOF) | 0.7473 | +0.068 (vs fold-mean) / +0.057 (vs OOF) | Per-class F1: [0.958, 0.8924, **0.1656**, 0.6759, 0.8253, 0.6266]. Label-2 collapse is the headline. Public > CV — GKF is pessimistic, not leaking. |
