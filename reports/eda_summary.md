# EDA Summary — Phase 1

> Source: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).
> Numerical findings cached in [`reports/eda_summary.json`](eda_summary.json).
> Figures in [`reports/figures/`](figures/).

## Dataset at a glance

| | Files | Users | User IDs | n_rows | NaNs |
|---|---|---|---|---|---|
| Train | 11,020 | 60 | User_001…User_060 | uniformly 300 | 0 |
| Test | 6,849 | 40 | User_061…User_100 | uniformly 300 | 0 |

CSV columns (train): `index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id`. Each train file carries one constant `label` for all 300 rows.

## The five findings that lock in Phase 2+

### 1. Data is pristine — no imputation work
Every file is exactly 300 rows. Zero NaN cells across both splits. No duplicate `file_id`s. Phase 7 = **do nothing**.

### 2. Severe class imbalance (33×)
Counts: `{0: 4643, 1: 4695, 2: 358, 3: 656, 4: 142, 5: 526}`. Labels 0+1 alone are 84.7% of train. **Imbalance ratio = 33** (label-1 / label-4). For macro-F1 this is the dominant difficulty: a few label-4 mispredictions weigh as much as hundreds of label-1 mispredictions. Only 6/60 users cover all 6 labels (median 5/6) — labels 2/3/4/5 are sparse per user.

**Decision:** class-weighted loss everywhere; log per-class F1; consider focal loss for DL models.

### 3. Train/test users are completely disjoint
Train: User_001…User_060. Test: User_061…User_100. Overlap = 0.

**Decision:** **Strict `GroupKFold(5, groups=user_id)` is mandatory project-wide.** `user_id` is forbidden as a feature. Per-user normalization can only use **per-file** statistics.

### 4. Activity-intensity gradient is obvious; everything else overlaps
Per-class mean of `feat_std_*` ranges from ≈0.009 (label 0, near-static) to ≈0.36 (label 4, high-intensity) — a clean monotonic ordering. The gravity-orientation features (`feat_mean_x/y/z`) also separate label 0 from the rest. **t-SNE on the 6 simple features:** label 4 forms a small distinct cluster; labels 0/1/2/3/5 overlap heavily.

**Decision:** Hardest pairs to disambiguate are among the lower-intensity activities {0, 1, 2, 3, 5}. The Phase-3 feature catalog (FFT bands, autocorr, sub-window pooling, gravity orientation) is justified — the 6-mean representation hits a ceiling here.

### 5. Moderate inter-user drift
For label 1 (most common), between-user variance / mean within-user variance of `feat_mean_x` = **2.14** — not extreme, not negligible. Different users do "the same activity" with different wrist orientations.

**Decision:** Add per-file z-score normalization as a Phase-3 feature variant, ablate.

## Naive baselines (GroupKFold-5)

| Model | Features | CV F1-macro | Per-fold |
|---|---|---|---|
| Majority class | n/a | **0.1331** | [0.1034, 0.0979, 0.0936, 0.0973, 0.0947] |
| Logistic Regression (balanced) | 6 column means | **0.5315** | [0.5157, 0.5353, 0.5417, 0.5366, 0.4901] |

Per-fold std of LR ≈ 0.018 → CV harness is stable. **Phase-3 LGBM target = ≥0.65 CV F1-macro.** Below 0.60 = pipeline bug.

## Open observations

- Test set has 6,849 files but `sample_submission.csv` has 6,850 data rows (1 extra). Likely a header/duplicate-row anomaly in the sample CSV — verify when first submission is built.
- Label 4 (142 files) splits to ~28 per fold on average, but distribution across users is skewed; some folds may have fragile minority-class statistics. Worth checking per-fold per-class counts during Phase 3.
- 1 Hz sampling means true gait frequencies (~2 Hz) alias, but the std channels preserve high-frequency energy implicitly. FFT band-energy features still appear class-discriminative in the spot check.
