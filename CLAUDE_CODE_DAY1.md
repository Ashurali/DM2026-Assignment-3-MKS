# Claude Code — Day 1 Instructions

> **Source of truth:** `PROJECT_PLAN.md` (read this first — it has the architecture and rationale).
> This file tells you what to do *today*. Refer back to PROJECT_PLAN.md for any "why."

## Today's mission

Stand up the repo, run interpretive EDA, build the CV harness, and produce **two Kaggle submissions** (pipeline validator + first competitive model) by end of day. We use 2 of 3 daily Kaggle slots.

You will work through 5 tasks in order. **Stop and surface the result** at each checkpoint marked 🛑 — do not auto-continue past it.

---

## Task 1 — Repo & environment (~30 min)

### 1.1 Create the structure

Working directory: a fresh local folder (user will tell you path; ask if not obvious from context). Create:

```
.
├── data/                           # gitignored
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── features/__init__.py
│   ├── models/__init__.py
│   └── utils/__init__.py
├── notebooks/
├── submissions/
│   └── log.md
├── reports/
│   ├── figures/.gitkeep
│   └── eda_summary.md
├── oof/                            # out-of-fold predictions, gitignored
│   └── .gitkeep
├── tests/
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md                 # already exists — leave as-is
```

### 1.2 `.gitignore`

```
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
.env
data/
oof/
submissions/*.csv
*.pkl
*.pt
.DS_Store
```

### 1.3 `requirements.txt`

```
numpy>=1.26
pandas>=2.1
scipy>=1.11
scikit-learn>=1.4
lightgbm>=4.0
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
pyarrow>=15.0
optuna>=3.5
kagglehub>=0.2
ipykernel
jupyter
```

(Skip torch for today — Day 1 is classical-ML only. We add it on Day 6.)

### 1.4 Set up venv

On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name dm2026-a3 --display-name "DM2026 A3"
```

(If activation fails on Windows, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

### 1.5 Initial git commit

```
git init
git add .
git commit -m "Phase 0: repo skeleton"
```

Do NOT push to GitHub yet — wait until Task 5 so the first push includes real content.

🛑 **Checkpoint 1:** Confirm repo structure and venv, then proceed.

---

## Task 2 — Data download (~15 min)

### 2.1 Kaggle auth

User needs `kaggle.json` API credentials. Check if `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows) exists. If not, instruct the user to:

1. Go to https://www.kaggle.com/settings → "Create New API Token"
2. Download `kaggle.json`
3. Place it at `%USERPROFILE%\.kaggle\kaggle.json` on Windows
4. Make sure file permissions are user-only

### 2.2 Download via kagglehub

Write `src/utils/download.py`:

```python
"""Download competition data via kagglehub and cache to data/."""
import shutil
from pathlib import Path
import kagglehub

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def download_and_link():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading nycu-data-mining-assignment-3...")
    src = Path(kagglehub.competition_download("nycu-data-mining-assignment-3"))
    print(f"Cached at: {src}")
    # Symlink or copy contents into ./data/raw/
    dst = DATA_DIR / "raw"
    if dst.exists():
        print(f"{dst} already exists — skipping copy.")
        return dst
    shutil.copytree(src, dst)
    print(f"Linked into: {dst}")
    return dst

if __name__ == "__main__":
    p = download_and_link()
    print("\nTop-level contents:")
    for child in sorted(p.iterdir()):
        print(f"  {child.name}")
```

Run it once: `python src/utils/download.py`. Verify you see `train/`, `test/`, and `sample_submission.csv` (or whatever the actual structure is).

🛑 **Checkpoint 2:** Print the actual top-level folder structure of the downloaded data. Confirm the file naming convention for train/test files (e.g., are they `User_001/12345.csv`? Just `12345.csv`?). Update PROJECT_PLAN.md §9 if anything differs from what we assumed.

---

## Task 3 — Interpretive EDA (~2h)

Create `notebooks/01_eda.ipynb`. **Key principle: every code cell should be followed by a markdown cell with your interpretation.** Don't dump plots and walk away — write what it means.

Each section ends with a "Decision" line that updates the plan. Carry findings forward into `reports/eda_summary.md` as you go.

### 3.1 Setup cell

```python
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

DATA = Path("../data/raw")
FIG = Path("../reports/figures"); FIG.mkdir(exist_ok=True, parents=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load file index
train_files = sorted(glob(str(DATA / "train" / "**" / "*.csv"), recursive=True))
test_files  = sorted(glob(str(DATA / "test"  / "**" / "*.csv"), recursive=True))
print(f"Train files: {len(train_files)}  |  Test files: {len(test_files)}")
```

### 3.2 Build a metadata table

For every file, extract `(file_id, user_id, n_rows, label)`. Labels for training files: read from a labels CSV if one exists, otherwise from filename or another mechanism — investigate the actual data layout.

Save this as `data/meta_train.parquet` and `data/meta_test.parquet` for reuse in later phases.

### 3.3 Section 1 — Dataset shape & integrity

```python
# Per-file row count distribution
meta_train["n_rows"].describe()
meta_train["n_rows"].value_counts().head(10)
```

Plot histogram of n_rows. Plot per-column NaN counts across a sample of files.

**Markdown interpretation must answer:**
- Are all files exactly 300 rows? If not, what's the distribution?
- How many files have any NaN, and in which columns?
- Any duplicate file_ids?

**Decision line:** Choose imputation strategy from PROJECT_PLAN.md §Phase 7 table.

### 3.4 Section 2 — Label distribution

```python
# Overall
fig, ax = plt.subplots(figsize=(6,4))
meta_train["label"].value_counts().sort_index().plot.bar(ax=ax)
ax.set(title="Train label distribution", xlabel="label", ylabel="count")
plt.savefig(FIG / "label_dist_overall.png", dpi=120, bbox_inches="tight")

# Per-user heatmap
piv = meta_train.pivot_table(index="user_id", columns="label", values="file_id",
                              aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(8, max(6, len(piv)*0.15)))
sns.heatmap(piv, cmap="viridis", ax=ax, cbar_kws={"label": "files"})
plt.savefig(FIG / "label_dist_per_user.png", dpi=120, bbox_inches="tight")
```

Compute imbalance ratio: `max_class_count / min_class_count`.

**Markdown interpretation must answer:**
- Is the class distribution balanced or skewed? What's the imbalance ratio?
- Do all users have all 6 labels, or do users specialize?
- Any class with < 5% of total — at risk of poor F1?

**Decision line:** Class weighting strategy. If imbalance > 2×, use class-weighted loss in all models.

### 3.5 Section 3 — Train/test user overlap (CRITICAL)

```python
train_users = set(meta_train["user_id"])
test_users  = set(meta_test["user_id"])
overlap = train_users & test_users
print(f"Train users: {len(train_users)}  Test users: {len(test_users)}")
print(f"Overlap: {len(overlap)}")
```

**This is the single most important EDA finding.** Write a clear interpretation:

- **If overlap == 0 (disjoint users):** Strict GroupKFold by user_id is mandatory. user_id cannot be a feature. Per-user normalization must use only that file's own stats.
- **If overlap > 0 (some shared users):** Two sub-cases. If overlap is *most* test users, treat as same-user problem and you can use user_id as a feature. If overlap is partial, this is messy — note it, GroupKFold is still safer.

**Decision line:** Confirm CV strategy in PROJECT_PLAN.md §Phase 2.

### 3.6 Section 4 — Signal characteristics by class

For each label 0–5, sample 5 random training files, plot all 6 channels over 300s on a 6×6 grid (rows=class, cols=channel). Save as `class_signals_grid.png`.

Compute simple summary table: per class, the mean of (mean magnitude) and mean of (std magnitude).

Run quick t-SNE on a 12-feature window summary (mean and std of each of 6 channels) and color by label. Save as `tsne_basic.png`.

**Markdown interpretation must answer:**
- Are classes visually distinct in the time series, or do some pairs look very similar?
- Which channels seem most discriminative?
- In t-SNE, are clusters separable? Which classes overlap?

**Decision line:** Identify 1–2 "hardest" class pairs. These are what the ensemble must disambiguate.

### 3.7 Section 5 — Per-user signal variation

For the most common label, plot the same 6 channels across 5 different users. Eyeball: does the same activity look the same across users, or wildly different?

Compute: per-user mean of (mean_x) for files of that label. Plot as a strip plot.

**Decision line:** If inter-user variance is high, per-user normalization is high-priority. Add to feature pipeline in Phase 3. If low, skip.

### 3.8 Section 6 — Frequency-domain spot check

For one file per class, plot FFT of the mean magnitude signal. Note: at 1Hz sampling, Nyquist = 0.5 Hz, so true gait frequencies (~2 Hz) are aliased — but informative patterns may still appear.

**Markdown interpretation:** Do FFT peaks look class-specific? Will FFT features add value, or should we lean on the std channel as the high-frequency proxy?

### 3.9 Section 7 — Naive baselines via GroupKFold

```python
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

# Build a tiny feature matrix: 6 features = mean of each channel over the window
def simple_features(file_path):
    df = pd.read_csv(file_path)
    return df[["mean_x","mean_y","mean_z","std_x","std_y","std_z"]].mean().values

X = np.vstack([simple_features(f) for f in tqdm(train_files)])
y = meta_train["label"].values
groups = meta_train["user_id"].values

gkf = GroupKFold(n_splits=5)

# Majority class
oof = np.zeros_like(y)
for tr, va in gkf.split(X, y, groups):
    clf = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    oof[va] = clf.predict(X[va])
f1_majority = f1_score(y, oof, average="macro")

# Logistic regression
oof = np.zeros_like(y)
for tr, va in gkf.split(X, y, groups):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE).fit(X[tr], y[tr])
    oof[va] = clf.predict(X[va])
f1_lr = f1_score(y, oof, average="macro")

print(f"Majority-class CV F1-macro: {f1_majority:.4f}")
print(f"Logistic Regression CV F1-macro: {f1_lr:.4f}")
```

**Decision line:** These are our floor and our "trivial linear" anchor. Phase 3 LGBM must beat the LR score by a meaningful margin.

### 3.10 Save EDA summary

Write `reports/eda_summary.json` with all numerical findings (label counts, n_rows distribution, NaN rates, user overlap count, baseline F1s, imbalance ratio).

Write `reports/eda_summary.md` with a 1-page narrative summary — this is the seed for Report Q1.

🛑 **Checkpoint 3:** Surface the following five things to the user (these are the inputs needed for sharpening downstream phases):
1. Label distribution counts
2. Train/test user overlap result
3. NaN/length-irregularity rate
4. Two naive baseline F1-macro scores
5. Any unexpected observations

---

## Task 4 — CV harness + minimal feature module (~45 min)

### 4.1 `src/utils/cv.py`

```python
"""Group-aware CV harness used by every model from now on."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from typing import Callable, Tuple

def make_folds(groups: np.ndarray, n_splits: int = 5):
    """Return list of (train_idx, val_idx) tuples for GroupKFold."""
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold doesn't take a seed, but order is deterministic given groups.
    return list(gkf.split(np.zeros(len(groups)), groups=groups))

def cv_score(
    fit_predict: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_classes: int = 6,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    fit_predict: callable(X_tr, y_tr, X_va) -> (preds_va, probs_va)
    Returns: (mean_f1, std_f1, oof_preds, oof_probs)
    """
    folds = make_folds(groups, n_splits=n_splits)
    oof_preds = np.zeros(len(y), dtype=np.int64)
    oof_probs = np.zeros((len(y), n_classes), dtype=np.float64)
    fold_f1 = []
    for k, (tr, va) in enumerate(folds):
        preds, probs = fit_predict(X[tr], y[tr], X[va])
        oof_preds[va] = preds
        oof_probs[va] = probs
        f = f1_score(y[va], preds, average="macro")
        fold_f1.append(f)
        print(f"  Fold {k}: F1-macro = {f:.4f}")
    mean = float(np.mean(fold_f1))
    std  = float(np.std(fold_f1))
    print(f"  CV F1-macro = {mean:.4f} ± {std:.4f}")
    return mean, std, oof_preds, oof_probs

def to_submission(file_ids: np.ndarray, preds: np.ndarray, path: str):
    df = pd.DataFrame({"Id": file_ids, "Label": preds.astype(int)})
    df.to_csv(path, index=False)
    print(f"Wrote submission: {path}  ({len(df)} rows)")
    return df
```

### 4.2 `tests/test_cv.py`

```python
import numpy as np
from src.utils.cv import make_folds

def test_make_folds_deterministic():
    groups = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]*3)
    folds_a = make_folds(groups, n_splits=5)
    folds_b = make_folds(groups, n_splits=5)
    for (tra, vaa), (trb, vab) in zip(folds_a, folds_b):
        assert np.array_equal(tra, trb)
        assert np.array_equal(vaa, vab)

def test_groupkfold_no_leak():
    groups = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4])
    for tr, va in make_folds(groups, n_splits=5):
        assert set(groups[tr]).isdisjoint(set(groups[va])), "user leak!"
```

Run: `pytest tests/`. Confirm pass.

### 4.3 `src/features/basic.py`

A minimal feature builder for today's first model. ~50 features.

```python
"""Basic window-level features for the Day-1 LightGBM submission."""
import numpy as np
import pandas as pd

CHANNELS = ["mean_x","mean_y","mean_z","std_x","std_y","std_z"]
STATS = ["mean", "std", "min", "max", "median", "q25", "q75"]

def _agg(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "std":  series.std(),
        "min":  series.min(),
        "max":  series.max(),
        "median": series.median(),
        "q25":  series.quantile(0.25),
        "q75":  series.quantile(0.75),
    }

def build_features(file_path: str) -> dict:
    df = pd.read_csv(file_path).sort_values("index").reset_index(drop=True)
    feat = {}
    # 6 channels x 7 stats = 42
    for ch in CHANNELS:
        for k, v in _agg(df[ch]).items():
            feat[f"{ch}__{k}"] = v
    # Magnitude on means
    mag = np.sqrt(df["mean_x"]**2 + df["mean_y"]**2 + df["mean_z"]**2)
    for k, v in _agg(mag).items():
        feat[f"mag_mean__{k}"] = v
    # Std-magnitude
    smag = np.sqrt(df["std_x"]**2 + df["std_y"]**2 + df["std_z"]**2)
    for k, v in _agg(smag).items():
        feat[f"mag_std__{k}"] = v
    # Gravity orientation: window-mean of each axis (already captured above as
    # mean_x__mean etc., but expose explicitly for clarity)
    feat["gravity_x"] = df["mean_x"].mean()
    feat["gravity_y"] = df["mean_y"].mean()
    feat["gravity_z"] = df["mean_z"].mean()
    feat["gravity_norm"] = np.sqrt(
        feat["gravity_x"]**2 + feat["gravity_y"]**2 + feat["gravity_z"]**2
    )
    return feat

def build_dataset(file_paths, file_ids):
    rows = [build_features(p) for p in file_paths]
    df = pd.DataFrame(rows)
    df.insert(0, "file_id", file_ids)
    return df
```

🛑 **Checkpoint 4:** Run unit tests, confirm `build_dataset` produces expected shape on a sample of 10 train files, then proceed.

---

## Task 5 — Two submissions (~1.5h)

### 5.1 Submission 1 — Majority-class predictor

This validates the Kaggle submission pipeline. ~5 minutes of work.

```python
# notebooks/02_submit_majority.ipynb (or a script)
from src.utils.cv import to_submission
import pandas as pd
import numpy as np

# Find the majority class from training labels
maj = int(meta_train["label"].mode().iloc[0])
test_ids = meta_test["file_id"].values
preds = np.full(len(test_ids), maj, dtype=int)
to_submission(test_ids, preds, "submissions/sub01_majority.csv")
```

Add a row to `submissions/log.md`. **User uploads `sub01_majority.csv` to Kaggle manually** (Claude Code does not have Kaggle write auth). User reports back the public LB score, which we add to the log.

### 5.2 Submission 2 — LightGBM on basic features

Write `src/models/train_lgbm_basic.py`:

```python
"""Day-1 LightGBM on basic features. 5-fold GroupKFold."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.utils.cv import cv_score, to_submission
from src.features.basic import build_dataset

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw"

def main():
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test  = pd.read_parquet(ROOT / "data" / "meta_test.parquet")

    print("Building train features...")
    Xtr_df = build_dataset(meta_train["path"].tolist(),
                            meta_train["file_id"].tolist())
    print("Building test features...")
    Xte_df = build_dataset(meta_test["path"].tolist(),
                            meta_test["file_id"].tolist())

    feat_cols = [c for c in Xtr_df.columns if c != "file_id"]
    X = Xtr_df[feat_cols].values
    y = meta_train["label"].values
    groups = meta_train["user_id"].values

    params = dict(
        objective="multiclass",
        num_class=6,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        min_data_in_leaf=20,
        verbose=-1,
        random_state=42,
    )

    def fit_predict(Xtr, ytr, Xva):
        ds = lgb.Dataset(Xtr, label=ytr)
        model = lgb.train(params, ds, num_boost_round=500)
        probs = model.predict(Xva)
        preds = probs.argmax(axis=1)
        return preds, probs

    print("\nRunning 5-fold GroupKFold CV...")
    mean, std, oof_preds, oof_probs = cv_score(
        fit_predict, X, y, groups, n_splits=5, n_classes=6
    )

    # Save OOF for later blending
    np.save(ROOT / "oof" / "lgbm_basic_v1_oof.npy", oof_probs)

    # Train on full train, predict test
    full_ds = lgb.Dataset(X, label=y)
    model = lgb.train(params, full_ds, num_boost_round=500)
    test_probs = model.predict(Xte_df[feat_cols].values)
    test_preds = test_probs.argmax(axis=1)

    sub_path = ROOT / "submissions" / "sub02_lgbm_basic.csv"
    to_submission(meta_test["file_id"].values, test_preds, str(sub_path))

    # Append to log
    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a") as f:
        f.write(
            f"| {pd.Timestamp.now().date()} | sub02_lgbm_basic | "
            f"LGBM ~50 basic features | {mean:.4f} | _pending_ | _pending_ | "
            f"first competitive |\n"
        )

if __name__ == "__main__":
    main()
```

Run it: `python -m src.models.train_lgbm_basic`. **User uploads `sub02_lgbm_basic.csv`**, reports public LB score, we update the log.

### 5.3 Initialize `submissions/log.md`

```markdown
# Submission Log

| Date | Version | Model | CV F1 | Public LB | Gap | Notes |
|---|---|---|---|---|---|---|
```

(The `train_lgbm_basic.py` will append rows. Manual edits welcome.)

🛑 **Checkpoint 5:** Surface to user:
- CV F1-macro for the LGBM basic model
- Both submission CSV paths ready for Kaggle upload
- A summary of any blockers or surprises encountered

---

## Task 6 — Push to GitHub

After Checkpoint 5 is acknowledged:

```
git add -A
git commit -m "Day 1: EDA + CV harness + basic LGBM submission"
# (user creates the GitHub repo and adds remote)
git remote add origin https://github.com/<user>/DM2026-Assignment-3.git
git branch -M main
git push -u origin main
```

Update `PROJECT_PLAN.md` §9 with the actual repo URL.

---

## End-of-day handoff

Surface to user a brief end-of-day summary including:

1. **Pipeline status:** repo set up, EDA complete, CV harness tested, 2 submissions generated
2. **Headline numbers:**
   - Naive baseline F1 (LR)
   - LGBM basic CV F1
   - Public LB scores once user uploads
3. **Decisions locked in** (from EDA):
   - CV scheme (strict GroupKFold or relaxed)
   - Class-weighting strategy
   - Imputation needed (yes/no/which)
   - Per-user normalization (yes/no)
4. **Open questions** for the user / for strategy discussion in chat
5. **Tomorrow's priority:** Phase 3 — full feature set + Optuna-tuned LGBM

---

## Guardrails

- **Do NOT commit data files** (data/ is gitignored).
- **Do NOT commit kaggle.json** ever.
- **Do NOT submit to Kaggle on the user's behalf** — generate CSV, user uploads via web.
- **Do NOT skip the markdown interpretation cells in the EDA notebook** — they're worth 10% of the assignment grade.
- **Do NOT change `random_state` / seed values** — keep `42` everywhere.
- If a checkpoint reveals something unexpected (e.g., test users overlap heavily, or files have wildly different lengths), **stop and surface it before continuing**. Don't paper over surprises.
