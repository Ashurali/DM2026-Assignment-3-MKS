# DM2026 Assignment 3 — Human Activity Recognition

NYCU 535703 Data Mining (Fall 2026), Assignment 3.

Kaggle competition: [`nycu-data-mining-assignment-3`](https://www.kaggle.com/competitions/nycu-data-mining-assignment-3)
Kaggle display name: **314540066**

## Plan

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full strategy, phase breakdown, and rationale.
Day-1 task list lives in [`CLAUDE_CODE_DAY1.md`](CLAUDE_CODE_DAY1.md).

## How to Run

_TBD — populated as phases complete._

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name dm2026-a3 --display-name "DM2026 A3"
```

## Layout

```
data/         # raw Kaggle data (gitignored)
src/
  features/   # feature engineering
  models/     # model definitions / training scripts
  utils/      # CV harness, IO, metrics
notebooks/    # exploratory + reporting notebooks
submissions/  # generated CSVs (gitignored) + log.md
reports/      # figures + EDA / ablation summaries
oof/          # out-of-fold predictions (gitignored)
tests/
```
