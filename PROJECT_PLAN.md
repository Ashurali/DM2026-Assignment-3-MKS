# DM2026 Assignment 3 — Human Activity Recognition Project Plan

> **Goal:** Rank #1 above Baseline 3 on the Kaggle private leaderboard for max Kaggle competition points (60/60), with a thorough report (40/40).
> **Deadline:** June 10, 2026, 23:55
> **Evaluation:** F1-score (macro)
> **Submission limit:** 3 per day

---

## 1. North Star

The realistic ceiling on this dataset (1Hz aggregated mean/std only — no raw signal) is probably **~0.80–0.90 macro-F1**, not 0.95. The grading rubric rewards *rank above Baseline 3*, not absolute score, so the explicit target is:

- **Above Baseline 3 on private leaderboard, ranked as high as possible** → 42–60 points
- **Strong report with ablation table and clear methodology** → 30–40 points
- **Reproducible public GitHub repo** → required (else 0)

---

## 2. Key Insights Driving Every Decision

### 2.1 GroupKFold by user is non-negotiable
Train/test are split by user folders. Random k-fold leaks the same user across folds and inflates CV by 10+ F1 points. **Every model uses `GroupKFold(n_splits=5, groups=user_id)`** until proven otherwise by EDA.

### 2.2 Gravity is a feature, not noise
The spec explicitly notes "data does not remove gravity components" and "the current three baselines did not use this information." The window-mean of (mean_x, mean_y, mean_z) encodes average wrist orientation — strongly discriminates sedentary vs. ambulatory activities. Most students will skip this.

### 2.3 The std channel is a free intensity proxy
mean_std over the window ≈ activity intensity. var_of_std over time ≈ activity rhythmicity. Trivial to compute, very predictive.

---

## 3. Modeling Architecture: 3-Track Ensemble

| Track | Model | Input | Strength |
|-------|-------|-------|----------|
| A | LightGBM | Engineered features (~150–300) | Exploits hand-crafted statistics; usually the strongest single model |
| B | CNN-BiLSTM (Xia et al. 2020) | Raw 6×300 sequence | Captures local motion patterns + temporal dependency |
| C | Transformer encoder | Raw 6×300 sequence | Long-range structure; complements CNN errors |
| D (stretch) | Pretrained TS foundation model (MOMENT/Chronos) | 6×300 sequence | Only if A+B+C blend is already top-3 by ~Day 11 |

**Ensemble cap:** 3–4 models total. Locked Day 1 — no expansion to 5+ even if time permits.

**Final submission = weighted blend of OOF probabilities** (weights fitted on OOF; see §Phase 8) + **test-time augmentation** + (conditionally) one round of pseudo-labeling.

---

## 4. Phase Plan

Each phase has a goal, concrete deliverables, a definition of done (DoD), and a prompt skeleton for Claude Code.

### Phase 0 — Repo & Environment Setup (Day 1, ~1h)

**Goal:** Reproducible workspace mirroring the A1 setup.

**Deliverables:**
- Public GitHub repo `DM2026-Assignment-3` (initialized with README, .gitignore, LICENSE)
- `environment.yml` or `requirements.txt`
- Folder structure:
  ```
  DM2026-Assignment-3/
  ├── data/                # gitignored — Kaggle data lands here
  ├── src/
  │   ├── features/        # feature engineering
  │   ├── models/          # model definitions
  │   └── utils/           # CV, IO, metrics
  ├── notebooks/
  ├── submissions/
  │   └── log.md           # submission tracking
  ├── reports/
  │   ├── figures/
  │   └── eda_summary.md
  ├── PROJECT_PLAN.md      # this file
  └── README.md
  ```

**Required packages:**
```
python=3.12
numpy pandas scikit-learn scipy
lightgbm xgboost
torch              # CPU is fine for prototyping; switch to Kaggle GPU for full runs
matplotlib seaborn tqdm pyarrow
optuna             # for hyperparameter tuning
kagglehub          # data download
```

**Claude Code prompt:**
> Initialize a new git repo at `~/projects/DM2026-Assignment-3`. Create the folder structure described in PROJECT_PLAN.md §Phase 0. Write a `requirements.txt` with the listed packages. Add a `.gitignore` that excludes `data/`, `submissions/*.csv`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pkl`, `*.pt`. Write a starter `README.md` with the project title, link to PROJECT_PLAN.md, and a "How to Run" placeholder. Don't install anything yet.

**DoD:** Repo pushed to GitHub, structure visible, no secrets committed.

---

### Phase 1 — EDA & Naive Baselines (Day 1–2)

**Goal:** Understand the data; produce report material for Q1; surface the decisions for later phases.

**Deliverables:**
- `notebooks/01_eda.ipynb`
- `reports/eda_summary.md` and `reports/eda_summary.json`
- `reports/figures/*.png` for key visualizations
- Two naive baseline CV scores logged in `submissions/log.md`

**The seven EDA sections:**

1. **Dataset shape & integrity** — files per user, total train/test counts, file length distribution (all 300?), per-column NaN counts, duplicate file_id check
2. **Label distribution** — overall histogram, per-user heatmap (user × label), label coverage stats
3. **Train/test user overlap** — *the single most important check*. Compare User_xxx folder names. Disjoint → strict GroupKFold. Overlap → user_id is usable.
4. **Signal characteristics by class** — sample 5 files per label, plot mean_x/y/z and std_x/y/z over 300s; PCA/t-SNE of simple window-stats colored by label
5. **Per-user signal variation** — same activity across 5 users to quantify calibration drift
6. **Frequency-domain quick look** — FFT of mean_magnitude per class
7. **Naive baselines via GroupKFold(5)** — (a) majority class, (b) LogisticRegression on 6 simple window-means

**Claude Code prompt:**
> Write a single notebook `notebooks/01_eda.ipynb` that loads the Kaggle data via `kagglehub` (competition: nycu-data-mining-assignment-3), caches it in `data/`, then answers the 7 EDA sections in PROJECT_PLAN.md §Phase 1. Save all figures to `reports/figures/` with descriptive filenames. Save numerical findings to `reports/eda_summary.json`. End the notebook with the two GroupKFold(5, groups=user_id) baseline scores. Set `random_state=42` everywhere. Don't do any modeling beyond the two naive baselines — this phase is descriptive only.

**DoD:** All 7 sections complete; naive baselines committed; EDA findings documented; clear answer to "are train/test users disjoint?"

**This phase output drives:**
- Whether to use user_id as a feature (depends on overlap)
- Whether per-user normalization is needed (depends on inter-user variance in §5)
- Imputation strategy (depends on §1 findings)
- Class weighting strategy (depends on §2 imbalance)

---

### Phase 2 — Validation Framework (Day 2)

**Goal:** A single, deterministic CV harness every model uses. Stop chasing public-LB ghosts.

**Deliverables:**
- `src/utils/cv.py` with:
  - `make_folds(user_ids, n_splits=5, seed=42)` returning fold indices
  - `cv_score(predict_fn, X, y, groups)` returning `(mean_f1, std_f1, oof_preds, oof_probs)`
  - `to_submission(probs, file_ids, path)` writing the Kaggle CSV
- Unit test that the same input + seed gives the same folds

**Claude Code prompt:**
> Write `src/utils/cv.py` per the spec in PROJECT_PLAN.md §Phase 2. Use `sklearn.model_selection.GroupKFold` and `sklearn.metrics.f1_score(average='macro')`. Add a tiny test in `tests/test_cv.py` that confirms determinism. Make sure `predict_fn` interface accepts `(X_train, y_train, X_val) -> probs_val` so it works with both sklearn-style and pytorch-style models.

**DoD:** Tests pass; reused by every subsequent phase.

---

### Phase 3 — Feature Engineering & LightGBM Baseline (Day 3–5)

**Goal:** Beat Baseline 3 with classical ML alone. This is the workhorse track.

**Feature catalog (~150–300 features per window):**

*Per-channel statistics (each of 6 channels: mean_x/y/z, std_x/y/z)*
- mean, std, min, max, median, p10, p25, p75, p90, IQR, range, skew, kurtosis, MAD

*Magnitude features*
- `mag_mean = sqrt(mean_x² + mean_y² + mean_z²)` per second → window stats on this series
- Same for `mag_std`

*Gravity orientation (the free win)*
- Window-mean of (mean_x, mean_y, mean_z) → 3 features
- Spherical angles (theta, phi) of that vector → 2 features
- Variance of orientation over time (split window into 5 chunks, compare gravity vector per chunk)

*Temporal derivatives ("jerk")*
- `diff(mean_channel)` → stats (mean, std, max abs)
- Same for std channels

*Frequency-domain (computed on each mean channel)*
- FFT energy in bands [0, 0.05), [0.05, 0.15), [0.15, 0.5) Hz
- Spectral entropy
- Dominant frequency and its power
- Note: at 1Hz Nyquist = 0.5Hz; gait frequencies are aliased but the *std* channels preserve high-freq energy implicitly

*Autocorrelation*
- Lag-1, lag-5, lag-10, lag-30 autocorrelation of mean_magnitude
- First major peak location and height (rhythmicity proxy)

*Sub-window pooling (5 chunks of 60s)*
- For each of 6 channels: mean and std per chunk → 60 features
- Capture phase changes within the window (e.g., walking → standing)

*Cross-axis*
- pearson(mean_x, mean_y), pearson(mean_x, mean_z), pearson(mean_y, mean_z)
- Same for std channels

*Zero-crossings & peaks*
- Zero-crossing rate of detrended mean signals
- Number of peaks (scipy.signal.find_peaks) on smoothed magnitude

*Data quality features*
- Row count (in case some files < 300)
- NaN count per channel (if applicable)

*Per-user normalization (conditional on EDA §3 + §5)*
- If users disjoint AND high inter-user variance: normalize using only that file's own stats
- If users overlap AND test users in train: subtract per-user training mean

**LightGBM training:**
- Optuna hyperparameter search on OOF F1-macro (50–100 trials)
- Class-weighted loss (`class_weight="balanced"`). EDA confirmed 33× imbalance — non-optional.
- **Minority-class oversampling** is now top-priority based on Day-1 sub02 result (label-2 F1 = 0.166, recall = 0.106). Two flavors to try:
  1. Random oversample on minority classes (labels 2/3/4/5) before training, integer multiplier per class derived from inverse frequency.
  2. SMOTE on the engineered feature space (`imblearn.over_sampling.SMOTE`).
  Both must respect GroupKFold — apply only inside the training fold, never across the val boundary.
- **Per-class F1 acceptance bar (binding constraints, not the macro number):**
  - Label 2 ≥ 0.50 (currently 0.166 — the dominant bottleneck)
  - Label 5 ≥ 0.70 (currently 0.627 — recall-bound, mistaken for something else)
  - Label 3 ≥ 0.75 (currently 0.676)
  - Labels 0, 1, 4 already pass.
  If after Phase 3 the label-2 floor isn't met, escalate to oversampling/SMOTE before tuning HP further.
- Save OOF predictions and OOF probabilities to `oof/lgbm_v1.npy`

**Claude Code prompt:**
> Write `src/features/build.py` implementing the feature catalog in PROJECT_PLAN.md §Phase 3. Each feature group should be a separate function so they can be ablated. Write `src/models/train_lgbm.py` that loads features, runs 5-fold GroupKFold via `src/utils/cv.py`, optionally tunes with Optuna (--tune flag), saves OOF probs and a submission CSV. Log to `submissions/log.md`.

**DoD:** Single LGBM achieves CV F1 ≥ Baseline 2 (verify against submitted public score); OOF probs saved.

---

### Phase 4 — LGBM Ablation Study (Day 5)

**Goal:** Document feature-group importance — directly serves Report Q4.

**Procedure:** Train LGBM with each feature group removed (one-at-a-time) and measure ΔCV F1. Also try cumulative (add one group at a time, starting from base stats).

**Deliverables:**
- `reports/ablation_features.md` with a table:

  | Configuration | CV F1-macro | Δ vs full |
  |---|---|---|
  | All features | 0.xxxx | — |
  | − FFT band energy | 0.xxxx | −0.xxxx |
  | − Sub-window pooling | 0.xxxx | −0.xxxx |
  | − Gravity orientation | 0.xxxx | −0.xxxx |
  | − Jerk | 0.xxxx | −0.xxxx |
  | − Autocorrelation | 0.xxxx | −0.xxxx |
  | Base stats only | 0.xxxx | −0.xxxx |

**DoD:** Table populated with real numbers. **Write the corresponding report Q4 paragraph now while it's fresh.**

---

### Phase 5 — CNN-BiLSTM (Day 6–8)

**Goal:** Capture local motion patterns + sequence dependency on raw 6×300 input.

**Architecture (small, fits CPU/laptop GPU):**
```
Input: (batch, 6, 300)
  → BatchNorm1d
  → Conv1d(6→64, kernel=5, pad=2) + ReLU + Conv1d(64→64, k=5, p=2) + ReLU + MaxPool1d(2)
  → Conv1d(64→128, k=5, p=2) + ReLU + Conv1d(128→128, k=5, p=2) + ReLU + MaxPool1d(2)
  → Conv1d(128→128, k=3, p=1) + ReLU
  → permute → BiLSTM(128→128, num_layers=1)
  → Attention pooling over time
  → Dropout(0.3) → Linear(256→6)
```

**Training:**
- Class-weighted CrossEntropyLoss (weights from EDA §2)
- AdamW, lr=1e-3, cosine schedule, 30–50 epochs
- Early stopping on val F1-macro, patience=8
- Mixed precision (`torch.cuda.amp`)
- 5-fold GroupKFold; save OOF logits → softmax → `oof/cnn_bilstm_v1.npy`

**Augmentation (training only, applied in Dataset.__getitem__):**

| Augmentation | Probability | Strength | Notes |
|---|---|---|---|
| Random 3D rotation of (mean_x, mean_y, mean_z) | 0.5 | uniform random rotation matrix | **Most important** — handles wrist orientation drift |
| Gaussian jitter | 0.5 | σ = 0.02 on means, 0.01 on stds | Cheap regularization |
| Magnitude scaling | 0.3 | multiply by 1 ± 0.1 | Inter-user calibration |
| Time warping | 0.3 | smooth cubic-spline distortion | Handles pace variation |
| Mixup | 0.2 | α = 0.2 | Implicit regularization |

**Skip:** segment shuffling (breaks temporal structure), channel dropout (too aggressive at 6 channels).

**Claude Code prompt:**
> Implement `src/models/cnn_bilstm.py` with the architecture in PROJECT_PLAN.md §Phase 5 and `src/models/train_cnn_bilstm.py` that runs 5-fold GroupKFold via `src/utils/cv.py`, applies the augmentations listed (only in training set), uses class-weighted loss with weights derived from `reports/eda_summary.json`, saves OOF probs to `oof/cnn_bilstm_v1.npy`, and logs to `submissions/log.md`. Use mixed precision if CUDA is available, otherwise CPU.

**DoD:** CV F1 saved; OOF probs saved; ideally beats LGBM by ≥ 0.005 OR makes meaningfully different errors.

---

### Phase 6 — Transformer Encoder (Day 9–10)

**Goal:** Long-range dependencies; complements CNN-BiLSTM in the blend.

**Architecture:**
```
Input: (batch, 6, 300)
  → Linear projection 6 → d_model=128
  → + sinusoidal positional encoding
  → prepend [CLS] token (learnable)
  → 4× TransformerEncoderLayer(d_model=128, nhead=4, ff=256, dropout=0.1)
  → take [CLS] output → LayerNorm → Linear(128→6)
```

**Training:** same as Phase 5 but lr=3e-4, longer warmup (5 epochs), same augmentations except mixup α = 0.4 (Transformers benefit from stronger mixup).

**DoD:** OOF probs saved to `oof/transformer_v1.npy`. Even if it underperforms CNN-BiLSTM, keep it for the blend if its errors differ.

---

### Phase 7 — Imputation Decision Point (Day 8, parallel)

Triggered by Phase 1 §1 findings.

| EDA finding | Action |
|---|---|
| All files ~300 rows, no NaNs | Do nothing |
| Some files < 300 rows | Forward-fill to 300 for sequence models; compute features over actual rows for LGBM |
| NaN gaps inside files | Linear interpolation per channel (signal is locally smooth at 1Hz) |
| Significant missingness (>5% files affected) | Add `missingness_rate` feature to LGBM (free signal) |

**Skip KNN imputation here** — temporal neighbors carry far more signal than feature-space neighbors for time series.

---

### Phase 8 — Blend, TTA, Pseudo-Label (Day 11–12)

**Step 1 — Stacking blend.**
Stack OOF probs from LGBM + CNN-BiLSTM + Transformer (each is an N×6 matrix). Find blend weights by minimizing OOF cross-entropy with `scipy.optimize.minimize`, simplex constraint. Alternative: train a small Logistic Regression meta-learner on stacked probs with GroupKFold. Pick whichever has higher CV F1.

**Step 2 — Test-time augmentation. (CONFIRMED, locked Day 1)**
For sequence models, predict on:
- Original test sequence
- Same sequence with small Gaussian noise (σ = 0.01)
- Same sequence with tiny time-shift
- Same sequence with magnitude scaling (×0.95, ×1.05) — added per Day-1 decision
Average the 5 probability vectors before blending. Note: do NOT include random rotation in TTA — that's a training-time augmentation and breaks the gravity-orientation prior at inference.

**Step 3 — Pseudo-labeling (CONDITIONAL).**
Skeptical concern locked Day 1: pseudo-labeling can amplify model bias on minority classes when the base ensemble itself is uncertain there. Decision:
- **Run only if** the blended ensemble shows CV F1-macro of **labels 2 and 4 individually ≥ 0.70**. Below that, the model isn't confident enough on minorities for pseudo-labels to be reliable.
- **Confidence threshold 0.97** (raised from the original 0.95) given imbalance — fewer minority pseudo-labels but cleaner ones.
- **Per-class quota:** cap pseudo-labels per class at 2× the train-set count for that class. Prevents the majority labels (0, 1) from drowning out minorities further.
- One round only. Retrain LGBM only; DL retrain skipped.
- Compare against no-PL blend in CV. If PL doesn't beat no-PL on per-class F1 for labels 2/3/4/5, drop it.

**Deliverables:**
- `oof/blend_final.npy`
- `submissions/final_v1.csv`, `submissions/final_v2.csv` (kept as the two private-LB picks)

**DoD:** Final two submissions chosen and logged with rationale.

---

### Phase 9 — Report Writing (Day 13–14)

**Mapping report questions to phases:**

| Question | Worth | Source material |
|---|---|---|
| Q1: Preliminary analysis | 10% | Phase 1 EDA + Phase 3 naive baselines |
| Q2: Preprocessing & impact | 10% | Phase 7 imputation decisions + per-user normalization + augmentation effects (each with Δ-F1 from your log) |
| Q3: Temporal alignment | 10% | Phase 3 sub-window pooling + FFT/autocorrelation features + Phase 5/6 sequence models |
| Q4: Ablation study | 10% | Phase 4 ablation table + DL component ablation (LGBM only / +CNN / +Transformer) |

**Required basics (else 0):**
- Public GitHub link in report
- Detailed "How to Run" instructions

**File naming:** `DM_asg3_{studentID}.pdf`

**Submit via:** E3 (and link from report)

**Kaggle Display Name:** **must be your StudentID** — verify before submitting.

---

## 4.5. Checkpointing Policy (locked Day 2)

Every long-running script must be **resumable**. Server interruptions (process kill, SSH disconnect, OOM, OS reboot) must never destroy more than the work of the currently-executing fold/trial/epoch.

**Mechanisms in `src/utils/checkpoint.py`:**

- **Per-fold CV cache.** `cv_score(..., checkpoint_name="<run>")` saves each fold's `(preds, probs)` to `checkpoints/<run>/fold_<k>.npz` on completion (atomic write via `.tmp.npz` → rename). Restart skips already-completed folds.
- **Optuna study persistence.** `maybe_tune(..., study_name="<run>")` opens an SQLite-backed `optuna.Study` at `optuna_studies/<run>.db` with `load_if_exists=True`. Completed trials persist; restart resumes at the next untrained trial.
- **Idempotent orchestration.** Shell scripts (`run_ablation.sh`, `run_phase3_all.sh`) check for sidecar JSON existence and skip already-finished steps.

**Required of every new training script (LGBM / DL / blend):**

1. Pass a `checkpoint_name=args.name` to `cv_score` so per-fold caching activates.
2. If using Optuna, pass `study_name=args.name` to persist trials.
3. For DL specifically (Phase 5/6): save per-epoch model state plus best-val-state separately. On startup, look for `checkpoints/<run>/latest.pt` and resume from its epoch.
4. Save OOF probs incrementally (per fold) and final-model state separately. Never trust an interrupted run; always check what's on disk first.

`checkpoints/` and `optuna_studies/` are gitignored — local-machine state only, not shared across machines via git. Sync via `scp` or rerun locally if needed.

---

## 5. Submission Discipline

### 5.1 Daily limit: 3 submissions
- Don't burn submissions on tiny tweaks. Validate locally on OOF first.
- Submit at meaningful checkpoints: end of Phase 3, 4, 5, 6, 8.
- Save the third daily slot for unexpected fixes.

### 5.2 Submission log template (`submissions/log.md`)
```markdown
| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
| 04-23 | v0 | majority class | 0.0xxx | 0.0xxx | — | sanity check |
| 04-24 | v1 | LR on 6 means | 0.xxxx | 0.xxxx | — | naive baseline |
| ...  | ...  | ... | ... | ... | ... | ... |
```
Track the **CV–public gap**. Growing gap = your CV scheme is leaking; investigate immediately.

### 5.3 Final two private-LB picks
Two submissions are auto-selected for private LB. **Manually override** to:
1. Best **CV** model (most likely to generalize)
2. Best **blend** (highest variance reduction)

NOT your best public LB unless it also has top CV. Public LB can be a fluke on 50% of test data.

---

## 6. Risks & Contingencies

| Risk | Symptom | Mitigation |
|---|---|---|
| CV leak via random fold | Public LB << CV by > 0.05 | Confirm GroupKFold by user; check no user_id derived feature leaking |
| Class imbalance crushes minority F1 | Per-class F1 highly variable | Class-weighted loss; oversample minority; focal loss |
| DL doesn't beat LGBM | OOF F1 of CNN < LGBM | Still keep for blend if errors differ; if redundant, drop and add feature variants |
| Out of time | Behind schedule | Phases 5/6 are *both* DL — drop Transformer, keep CNN-BiLSTM. LGBM + CNN-BiLSTM blend still strong. |
| Kaggle data download fails | kagglehub error | Manual download via Kaggle API CLI; cache aggressively in `data/` |
| Laptop too slow for DL | Training > 4h per fold | Move DL training to Kaggle Notebooks (free T4/P100); inference local |

---

## 6.5. Tier-1 / Tier-2 / Tier-3 expansion (locked Day 14, May 7)

After two weeks of work the standard 3-track plan is at LB 0.7816 vs leaderboard #1 at 0.8106. With ~5 weeks until deadline (Jun 10), the plan was upgraded to four tiers of additional moves. Keep the original phases, add these:

**Tier 1 — already coded, run this week:**
- catch22 features (132 features × 22 catch22-curated stats)
- MiniRocket (~9,996 random conv features + LightGBM head)
- XGBoost + CatBoost on the 271-feature catalog (architectural diversity in GBDTs)
- Stacked features: DL embeddings (CNN/Transformer penultimate layer) → LGBM
- Test-time augmentation on all DL final-models
- Multi-seed CNN/Transformer averaging

**Tier 2 — week of May 14:**
- InceptionTime architecture
- 1D ResNet
- TCN (Temporal Conv Network)
- SWA / EMA on DL training
- Snapshot ensembling per training run
- Probability calibration (Platt / isotonic)
- Focal loss / label smoothing for minority-class boost

**Tier 3 — week of May 21:**
- Self-supervised pretraining on test sequences (masked reconstruction or contrastive)
- Domain adaptation (DANN, CORAL alignment)
- Pseudo-labeling with relaxed gate (revisit; was blocked at L2 ≥ 0.70)
- Multi-level stacking (level-2 meta-learner on level-1 OOFs)

**Final week (May 28 – Jun 10):**
- Final blend tuning with all available base models
- TTA refinement
- Pick top-2 private-LB submissions manually
- Report writing + GitHub polish

**Realistic stretch target:** LB 0.82–0.84 (above current leader 0.8106).

---

## 7. Schedule Summary

| Day | Phase | Output |
|---|---|---|
| 1 | 0 + 1 start | Repo + initial EDA cells |
| 2 | 1 finish + 2 | EDA complete + CV harness |
| 3–5 | 3 + 4 | LGBM + ablation; first competitive submission |
| 6–8 | 5 + 7 | CNN-BiLSTM + imputation decisions |
| 9–10 | 6 | Transformer |
| 11–12 | 8 | Blend + TTA + pseudo-label; final submissions |
| 13–14 | 9 | Report writing & polish |
| 15 | buffer | Reproducibility check; verify Kaggle display name; submit |

If June 10 is the deadline and today's the planning date, **start Phase 0 immediately and lock the schedule on a calendar.** Slip ≤ 2 days; anything more, drop the Transformer track.

---

## 8. Feedback Loop with Strategy Discussion

After Phase 1 EDA completes, paste back to ChatGPT/Claude:

1. Label distribution (counts per class) and user-coverage heatmap summary
2. Train/test user overlap result (overlap or disjoint)
3. NaN/length-irregularity rate
4. Two naive baseline F1-macro scores
5. Any unexpected observations

These five inputs sharpen all subsequent phases — particularly Phase 3 feature priorities, Phase 5 augmentation strength, and Phase 7 imputation strategy.

---

## 9. Quick Reference

**Repo:** `https://github.com/Ashurali/DM2026-Assignment-3-MKS`
**Kaggle competition:** `nycu-data-mining-assignment-3`
**Kaggle display name:** **314540066**
**Report filename:** `DM_asg3_314540066.pdf`
**Deadline:** June 10, 2026, 23:55
**Daily Kaggle submissions:** 3
**Random seed convention:** `42` everywhere

**Data layout (verified Day 1):**
- `data/raw/train/train/User_NNN/NNNNN.csv` — 60 users (User_001…User_060), **11,020 files**
- `data/raw/test/test/User_NNN/NNNNN.csv` — 40 users (User_061…User_100), **6,849 files**
- `data/raw/sample_submission.csv` — 6,850 rows (`Id,Label`); 1 extra row vs. test count, presumably a header anomaly.
- **Train/test users are DISJOINT** by ID range → strict GroupKFold by user is mandatory; `user_id` cannot be a feature.
- CSV columns (train): `index, mean_x, mean_y, mean_z, std_x, std_y, std_z, label, file_id`. Test has the same minus `label`.
- One label per file (replicated on every row). File length: **uniformly 300 rows**, no NaNs anywhere → no imputation needed.
- 6 classes, **imbalance ratio 33×** (label-1=4695, label-4=142). Class-weighted loss everywhere.
- Naive baselines (Day 1, GroupKFold-5): majority=0.1331, LR-on-6-means=0.5315 → Phase 3 LGBM target ≥0.65.
- `file_id` is an integer (e.g., `11021`); on disk it's the zero-padded filename (`11021.csv`). Submission uses the unpadded integer.

---

*Last updated: keep this living. Update CV scores and findings inline as phases complete.*
