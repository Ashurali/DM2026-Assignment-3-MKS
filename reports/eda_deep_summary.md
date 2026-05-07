# Deep EDA — Phase-2 analysis

> Source: [`notebooks/02_deep_eda.py`](../notebooks/02_deep_eda.py)
> Numerical findings: [`reports/eda_deep_summary.json`](eda_deep_summary.json)
> Figures: [`reports/figures_deep/`](figures_deep/)

This is the second-pass EDA, going beyond the initial 7 sections in `eda_summary.md`. Driven by the question: "What's actually limiting our model, and why?"

## TL;DR — five findings that change the model strategy

1. **L2-vs-L1 is the structural confusion bottleneck.** 64.2% of true-L2 files (230 of 358) get predicted as L1 by our best model. Centroid distance L1↔L2 = 0.225 — well below the 0.32 we get to L0. Class **L1 acts as the model's "default" answer** when the signal is ambiguous.
2. **The model is *confidently wrong* on minorities.** When wrong on L2, mean prediction confidence is **0.931**. This is why threshold tuning recovers so much — wrong predictions sit on the wrong side of confident decision boundaries, easy to push back with class multipliers.
3. **L1↔L5 are the closest pair** (centroid distance 0.110). 22% of L5 → L1. This is the second-tier confusion the model hasn't fully addressed.
4. **The 271-feature catalog has heavy redundancy** — 144 feature pairs with |correlation| > 0.95. Several are *literal duplicates* (e.g. `mean_x__mean` = `grav_x`). A pruned ~200-feature catalog should perform identically with less overfitting risk.
5. **Std-channel features are heavy-tailed** (skewness 2.4–3.7, kurtosis up to 21.6) — extreme high-intensity outliers dominate. A log or Yeo-Johnson transform on std features could help linear and tree models capture small-intensity differences.

## §1 — Per-channel distribution shape

| channel | mean | std | skew | kurt | p1 | p99 | n_outliers (>3σ) |
|---|---|---|---|---|---|---|---|
| mean_x | -0.145 | 0.573 | 0.37 | -1.21 | -0.97 | 0.95 | 0 |
| mean_y | 0.011 | 0.381 | -0.13 | -0.38 | -0.87 | 0.80 | 6 |
| mean_z | 0.196 | 0.510 | -0.52 | -0.60 | -0.97 | 0.96 | 0 |
| **std_x** | 0.051 | 0.067 | **2.78** | **13.25** | 0.001 | 0.310 | **211** |
| **std_y** | 0.044 | 0.066 | **3.73** | **21.57** | 0.001 | 0.321 | **220** |
| **std_z** | 0.047 | 0.058 | **2.37** | **8.68** | 0.002 | 0.266 | **206** |

- **Mean channels** (gravity orientation): roughly symmetric, no extreme outliers, kurtosis < 0 (light tails). The negative kurtosis is consistent with a near-uniform `[-1, +1]` accelerometer-magnitude range.
- **Std channels** (motion intensity): heavy right tails. The "high-intensity activity" L4 dominates the tail. ~200 files per std channel are 3+ sigma outliers — these are mostly L4 (142 files) plus high-intensity instances of L3.

**Decision:** Add log/sqrt transformations of std channels as a Phase-3 feature variant. Tree-based models handle skew well, but linear models and DL embeddings benefit from this.

## §2 — Per-class violin shape

See `figures_deep/s2_class_violin.png`. Per-channel intensity gradient (clean monotonic):
- L0: std≈0.008 (most static)
- L1: std≈0.045
- L2/L5: std≈0.09 (overlap!)
- L3: std≈0.16
- L4: std≈0.27 (most intense, distinct)

The intensity gradient explains why L4 is easy and L2/L5 are confusable.

## §3 — Inter-class centroid distance (Euclidean, 6-feature simple representation)

```
       L0     L1     L2     L3     L4     L5
L0  0.000  0.271  0.321  0.450  0.641  0.348
L1  0.271  0.000  0.225  0.372  0.610  0.110  ← L1↔L5 closest
L2  0.321  0.225  0.000  0.190  0.638  0.208  ← L2↔L3 also close
L3  0.450  0.372  0.190  0.000  0.575  0.344
L4  0.641  0.610  0.638  0.575  0.000  0.609
L5  0.348  0.110  0.208  0.344  0.609  0.000
```

**Closest pairs:**
1. **L1↔L5: 0.110** — these clusters are nearly co-located in feature space
2. L2↔L3: 0.190
3. L2↔L5: 0.208
4. **L1↔L2: 0.225** — the actual L2 confusion target
5. L0↔L1: 0.271

L4 sits 0.6+ from everyone — the high-intensity "outlier" class. Easy.

**Implication:** A binary {L1, L5} classifier and a binary {L1, L2} classifier are the two micro-tasks the ensemble must solve. We should train **specialist binary models** for these pairs and stack their probabilities.

## §4 — Confusion matrix from current best (combo OOF, F1=0.7687)

Top off-diagonal cells (where the model is most-wrong):

| true → pred | count | fraction |
|---|---|---|
| L2 → L1 | **230** | **64.2% of L2** ← the L2 collapse |
| L1 → L0 | 165 | 3.5% of L1 |
| L0 → L1 | 129 | 2.8% of L0 |
| L5 → L1 | 114 | 21.7% of L5 |
| L3 → L1 | 88 | 13.4% of L3 |
| L1 → L3 | 71 | 1.5% of L1 |
| L2 → L3 | 56 | 15.6% of L2 |

L1 is the model's "default answer" — when uncertain, predict L1. This is rational given L1 is 42.6% of train, but it's where macro-F1 bleeds.

**The L2 row tells the whole story:** of 358 true-L2 files, only 61 (17%) get L2 prediction; 230 (64%) → L1; 56 (16%) → L3; 8 (2%) → L5. The model rarely even *considers* L2.

## §5 — Feature correlation analysis (271 engineered features)

- **144 feature pairs** with |corr| > 0.95
- **Median pairwise |corr|** ≈ 0.18 (most features are reasonably independent)
- **95th percentile pairwise |corr|** ≈ 0.65

Top **literal duplicates** (correlation = 1.000, exactly redundant):
- `mean_x__mean` = `grav_x` (we computed gravity twice)
- `mean_y__mean` = `grav_y`
- `mean_z__mean` = `grav_z`
- `mean_*__skew` = `znorm_mean_*__skew` (z-score doesn't change skewness)
- `std_*__skew` = `znorm_std_*__skew`
- `jerk_std_y__std` = `jerk_std_y__rms` (zero-mean diff has identity)

**Decision:** Phase-3-v2 should drop the literal-redundant features. Quick gain: smaller catalog → less overfitting → tighter Optuna search. Estimate ~25-50 dropped features.

## §6 — Hard-example analysis (per-class confidence + wrong rate)

| class | n_total | n_correct | n_wrong | mean_conf_correct | **mean_conf_wrong** | wrong_rate |
|---|---|---|---|---|---|---|
| 0 | 4643 | 4495 | 148 | 0.998 | 0.919 | 3.2% |
| 1 | 4695 | 4389 | 306 | 0.989 | 0.901 | 6.5% |
| **2** | **358** | **61** | **297** | 0.828 | **0.931** | **83.0%** |
| 3 | 656 | 523 | 133 | 0.977 | 0.861 | 20.3% |
| 4 | 142 | 130 | 12 | 1.000 | 0.955 | 8.5% |
| 5 | 526 | 380 | 146 | 0.980 | 0.896 | 27.8% |

**The L2 confidence anomaly:**
- mean_conf when *correct* = **0.83** — even when the model gets L2 right, it's not very sure
- mean_conf when *wrong* = **0.93** — the model is more confident on its wrong predictions than its correct ones for L2

This reverses the normal pattern. It means the model has learned that *certain feature signatures imply L1 with high confidence*, and L2 happens to share those signatures. This is a representational problem, not a calibration problem alone.

**793 confidently-wrong files** (max prob > 0.85, prediction ≠ truth):
- L0: 115
- L1: 227
- **L2: 243** ← most of L2's wrong predictions are with high confidence
- L3: 87
- L4: 11
- L5: 110

The 243 confidently-wrong L2 files are where the L2 problem lives. They're the targets for any minority-class technique.

## §7 — t-SNE in the full 271-feature space

See `figures_deep/s7_tsne_full_features.png`. Two views:
- (a) colored by class: L4 forms a clean cluster; L0/L1/L2/L3/L5 are mixed in a complex web
- (b) colored by user: per-user clusters are visible — same user's files tend to cluster regardless of class. **This is the user-disjoint generalization problem made visible.**

The user-clustering effect explains why **CV→LB gap is structural at +0.045–0.056**: the model partly learns user-specific signatures that don't transfer.

## §8 — Per-user outlier detection

Top 5 outlier users (highest mean |z| of their per-feature mean profile):
| user | mean &#124;z&#124; |
|---|---|
| User_030 | 2.033 |
| User_043 | 1.277 |
| User_045 | 1.274 |
| User_021 | 1.258 |
| User_052 | 1.150 |

Most-typical:
| user | mean &#124;z&#124; |
|---|---|
| User_011 | 0.390 |
| User_054 | 0.395 |
| User_053 | 0.427 |

User_030 stands out — mean |z| of 2.03 means their profile is consistently >2σ from the population mean across many features. Could be a high-intensity user, a calibration-different sensor, or label noise. Worth a manual look.

**Decision:** Worth experimenting with **outlier-user-weighted CV**: down-weight User_030's contribution to the loss, or hold it out entirely from one fold to verify it's not destabilizing CV scores.

## §9 — L2 deep dive

- **358 L2 files** spread across 52 users (87% of users have at least one L2 file)
- Top L2-frequent users: User_014 (21), User_047 (21), User_026 (19), User_029 (18), User_035 (17)
- Mean L2 OOF prob distribution (where the model puts mass on true-L2 files):
  - L0: 0.6%, **L1: 64.2%**, L2: 17%, L3: 15.6%, L4: 0.3%, L5: 2.2%

The model puts only **17% probability mass on the correct L2 class** for true-L2 files on average — i.e., calibrated probability is barely above chance for L2 detection. This is why threshold tuning works so dramatically: scaling L2 probs by 2× barely changes raw predictions, scaling by 3-5× starts flipping the argmax.

## Actionable insights — what to add to the plan

| Insight | Action | Tier |
|---|---|---|
| Std channels are heavy-tailed | Add Yeo-Johnson / log transforms as Phase-3 feature variant | 1 |
| 144 redundant features | Drop literal duplicates, retrain combo (faster, less overfit) | 1 |
| L2 confused as L1 | **Train binary L1-vs-L2 specialist** classifier; stack its prob into combo | 2 |
| L1↔L5 confusion | Train binary L1-vs-L5 specialist; stack | 2 |
| Confidently wrong on L2 | Already addressed by threshold tuning + calibration | done |
| User_030 outlier | Hold-out CV experiment + sanity check | 2 |
| t-SNE shows user clustering | This is structural; addressed by GroupKFold; could try DANN domain adaptation | 3 |
| L2 has 358/52 spread | Prevents simple subgroup-balancing — must rely on weighting | done |

The biggest two targeted wins from this analysis are likely:
1. **L1-vs-L2 binary specialist + stacked features** — could lift L2 from 0.37 to 0.45+
2. **De-duplicating the feature catalog** — small lift, but easy
