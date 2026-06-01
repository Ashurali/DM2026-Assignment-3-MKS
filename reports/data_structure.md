# Structural explanation of the dataset

## 1. Competition framing

| Aspect | Value |
|---|---|
| Competition | DM2026 Assignment 3 — Human Activity Recognition |
| Task | Multi-class classification (6 activity classes) |
| Eval metric | F1-macro (each class weighted equally regardless of support) |
| Submission format | `Id,Label` CSV — one row per test `file_id`, integer label 0-5 |
| Baseline thresholds | Baseline-1: 0.1024, Baseline-3: 0.7088 (full points), leader: 0.81+ |

## 2. File-system layout

```
data/raw/
├── train/train/User_001/<file_id>.csv  ... User_060/<file_id>.csv
├── test/test/User_061/<file_id>.csv    ... User_100/<file_id>.csv
└── sample_submission.csv
```

**One CSV = one sample.** Each file's name is its `file_id` (an integer, globally unique across train and test). Files are grouped into per-user directories.

| Split | Files | Users | User IDs | n_rows per file | NaNs |
|---|---|---|---|---|---|
| Train | 11,020 | 60 | `User_001`–`User_060` | exactly 300 | 0 |
| Test | 6,849 | 40 | `User_061`–`User_100` | exactly 300 | 0 |

Pristine — zero missing cells, no duplicate file IDs, every file is 300 rows.

## 3. Per-file sequence structure

Each CSV has 300 sequential rows representing **one continuous 5-minute window** of a user's activity (1 row per second, 300 seconds = 5 min).

**Train file columns:**
```
index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id
```

**Test file columns:** same minus `label`.

**Key fact about the data:** each 1-second row contains the **mean and standard deviation** of the 3-axis accelerometer measurement *within that second*. So:
- The **original sensor stream** was likely 50-200 Hz (we don't know the exact rate)
- It was **aggregated to 1 Hz** by computing per-second mean and std
- We never see the raw IMU samples — only these 6-channel summaries
- `mean_*` channels encode gravity orientation (which direction the device is tilted)
- `std_*` channels encode within-second motion intensity (how much the sensor was vibrating during that second)

The `label` column is **constant across all 300 rows of a file** — the entire 5-minute window has one activity label.

## 4. Label structure

Six classes, anonymous integer labels 0-5. **We don't know what each label represents** (Kaggle didn't disclose the activity name for each class). What we know empirically:

| Label | Train rows | % | Mean-of-`std` channel | Inferred character |
|---|---|---|---|---|
| L0 | 4643 | 42.1% | ≈0.008 | **Most static** (probably "sitting", "lying down", "stationary") |
| L1 | 4695 | 42.6% | ≈0.045 | **Default activity** — bulk of data |
| L2 | 358 | 3.2% | ≈0.090 | Low intensity; **kinematically near-identical to L1** |
| L3 | 656 | 6.0% | ≈0.160 | Medium intensity |
| L4 | 142 | 1.3% | ≈0.270 | **Most intense** (probably "running" — clean cluster) |
| L5 | 526 | 4.8% | ≈0.090 | Low-medium intensity, similar to L2 |

**Severe class imbalance: 33×** between L1 (largest) and L4 (smallest).

The **monotonic intensity gradient** in the std channels gives a partial natural ordering: L0 < L1 < L2/L5 < L3 < L4. L4 is the only class clearly separable by raw intensity.

## 5. The critical experimental-design property: train/test users are disjoint

This is the single most important structural fact about this dataset.

```
Train users:  User_001 ... User_060
Test users:   User_061 ... User_100
Overlap:      ZERO
```

**Implications:**

- **Cross-validation must be group-aware.** Random K-fold leaks user identity into the validation set (because each user has multiple files). We must use `GroupKFold(splits=5, groups=user_id)` so a user's files are entirely in either the train or val side of any fold.

- **`user_id` is forbidden as a feature** — at inference time we'd see new users not in train.

- **Inter-user variability shows up as distribution shift.** Different users perform "the same activity" with different wrist orientations, intensity levels, gait patterns. Models that latch onto user-specific signatures look great on internal CV but transfer poorly to the disjoint test users. This is the source of the OOF→LB gap throughout the project.

- **Per-user normalization can only use file-internal statistics.** We can subtract a file's own mean and divide by its own std (per-file z-score), but we cannot use a global per-user mean across multiple files of the same user, because we'd have nothing equivalent at test time.

## 6. Per-user composition

Each user contributes multiple files (~110-200 each on average). However, per-user *class* coverage is highly uneven:

- **Only 6/60 train users** have files of all 6 activity labels.
- **Median user covers 5/6 labels.**
- L2/L3/L4/L5 are sparse per-user — many users have zero files of a given minority class.
- L2 specifically is spread across 52 of 60 users (87% have ≥1 L2 file), but the count is small (1-21 per user).

The 358 L2 train files are not a clean L2 "subset of users" — they're scattered, mostly 1-3 files per user across many users. Same for L4.

## 7. Aggregation properties (the silent ceiling)

The 1-Hz `mean+std` aggregation is the **single biggest information bottleneck** in this dataset:

- True human gait frequency is ≈2 Hz (two steps per second). At 1-Hz Nyquist, this **aliases**. We can't see step-by-step motion in `mean_*`.
- The `std_*` channel preserves *intensity* of within-second motion but not its *temporal pattern* (when in the second the peak was, what frequency content it had).
- Spectrograms / FFT of the 6 channels can find sub-1-Hz patterns (multi-second cadence, transition events), but anything faster than 0.5 Hz is gone.
- This is why GAF and spectrogram representations didn't help — they can't recover information that was already aggregated away.

The original raw IMU stream presumably contained much more discriminative high-frequency content (step impulses, vibration signatures, gait phase) — but Kaggle's preprocessing erased it.

## 8. Confusion structure (empirically discovered)

Best-model OOF confusion shows a single dominant failure mode:

| Confusion | Count | % of source class |
|---|---|---|
| **L2 → L1** | **167** | **47%** of all L2 |
| L1 → L2 (FP) | 119 | symmetric mistake |
| L5 → L1 | 97 | 18% of L5 |
| L1 → L0 | 191 | 4% of L1 |
| L3 → L1 | 67 | 10% of L3 |

In feature space (CNN-BiLSTM 256-d embedding):
- **62% of L2 samples have an L1 sample as their nearest neighbor.** Only 13% have another L2 sample.
- L1 ↔ L5 centroid distance: 0.110 (closest pair)
- L1 ↔ L2 centroid distance: 0.225 (close)
- L4 sits 0.6+ from everyone (the easy outlier)

L2 effectively **does not have its own region** in any embedding we've tried. L2 samples are scattered among L1 samples.

## 9. Known unknowns

- **What activity each label represents** — we never get the mapping. Inferences from intensity patterns are educated guesses.
- **Original sensor sampling rate** — likely 50-100 Hz before aggregation, but unspecified.
- **Whether windows overlap or are contiguous** — files within a user could be sequential 5-min chunks of one session, or independent recordings.
- **Whether the `index` column has meaning** — appears to be just 0..299 row counter.
- **Sensor placement** — wrist? hip? bag? Affects what activities are easy/hard to discriminate.
- **Whether L1/L2 are intrinsically similar (e.g., "walking" vs "walking-with-load") or whether the aggregation is what made them similar** — we can't tell from the data alone.

## 10. Modeling implications (the design constraints these create)

| Property | Constraint on modeling |
|---|---|
| Disjoint train/test users | GroupKFold mandatory; per-fold reproducibility |
| Severe class imbalance | Class-weighted loss everywhere; macro-F1 primary metric |
| 1-Hz aggregation | High-freq features unusable; `std_*` channels are the proxy |
| Constant label per file | Per-file (not per-row) classification; can use whole-window features |
| Per-user variability | Per-file normalization OK; per-subject normalization NOT OK |
| 300-step sequences | DL models see (6, 300) inputs; 2D image transforms (GAF/STFT) are valid |
| L2 ≈ L1 in feature space | The structural ceiling — confirmed by GMM, cleanlab, and contrastive failure |
| L4 is small but easy | Imbalance alone doesn't predict difficulty — feature-space separability does |

## 11. Why we landed at LB 0.8114

- We **cannot make L2 separable** in this feature space (data is what it is).
- We **can stop letting L2 ambiguity contaminate the easy-class boundaries** — that's what hierarchical decomposition does.
- Adding a contrastive emb gave a small additional signal that the LGBM stacker could exploit.
- Post-hoc isotonic calibration + Nelder-Mead threshold tuning corrects the residual class-prior bias from majority-class weighting during training.

The structural ceiling for this dataset, given the 1-Hz aggregation and the L1↔L2 overlap, looks to be in the **0.81-0.83** range. That's why nobody in the public LB has broken 0.82.
