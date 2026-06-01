# DM2026 Assignment 3 — Human Activity Recognition

NYCU 535703 Data Mining (Spring 2026), Assignment 3.

Kaggle competition: [`nycu-data-mining-assignment-3`](https://www.kaggle.com/competitions/nycu-data-mining-assignment-3)
Kaggle display name: **314540066**

## Status — public leaderboard

| | Submission | OOF F1 | Public LB |
|---|---|---|---|
| **Primary** | `submissions/sub_hier_v6_a842_grid_peak.csv` | 0.7880 | **0.8154** (rank #3) |
| **Backup** | `submissions/sub_hier_v4_a088_cal_thresh.csv` | 0.7856 | 0.8114 |

Both beat Baseline-3 (0.7088) by >0.10. Top of leaderboard: 0.8240.

## Documentation entry points

| File | What it covers |
|---|---|
| [`reports/final_report.md`](reports/final_report.md) | **Comprehensive project report** — problem, methodology, results, discussion, references |
| [`reports/architecture.md`](reports/architecture.md) | Exact reproducible architecture for primary + backup, with all hyperparameters |
| [`reports/experiments.md`](reports/experiments.md) | What worked / didn't — full positive and negative result table |
| [`reports/data_structure.md`](reports/data_structure.md) | Dataset structural facts (1-Hz aggregation, disjoint users, L1↔L2 overlap) |
| [`reports/ablation_features.md`](reports/ablation_features.md) | Feature-group ablation results (Phase 4) |
| [`reports/eda_summary.md`](reports/eda_summary.md), [`reports/eda_deep_summary.md`](reports/eda_deep_summary.md) | Exploratory data analysis |
| [`reports/literature_synthesis.md`](reports/literature_synthesis.md) | Related work and references |
| [`experiments_archive/README.md`](experiments_archive/README.md) | Archived (failed) experiments with reasons |

## Reproducing the submissions

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Phase A — Base models (each produces OOF .npy files; already cached in oof/)
python src/models/train_cnn_bilstm.py
python src/models/train_transformer.py
python src/models/train_xgb_cat.py
python src/models/train_minirocket.py
python src/models/train_lgbm_combo.py --name combo_full_v2

# Phase B — L1↔L2 contrastive embedding (needs CNN-BiLSTM emb)
python scripts/train_l1l2_contrastive.py --gpu --epochs 200 --margin 0.4

# Phase C — Equilibrium Optimizer feature selection
#   (~7 hours on 24-core CPU server; cached: oof/eo_selected_mask.npy)
python scripts/eo_feature_select.py \
  --particles 25 --iterations 50 --inner-folds 3 --n-estimators 150 \
  --min-features 50

# Phase D — Final hierarchical models

#   Backup (full features, ~25 min):
python scripts/train_hier_v4_and_submit.py --gpu --seeds 17 23 41
#   → submissions/sub_hier_v4_a088_cal_thresh.csv

#   Primary (EO-selected features, ~25 min):
python scripts/train_hier_v6_eo_selected.py --seeds 17 23 41
python scripts/threshold_grid_v6.py
#   → submissions/sub_hier_v6_a842_grid_peak.csv
```

## Layout

```
DM2026-Assignment-3-MKS/
├── README.md                  this file
├── PROJECT_PLAN.md            day-0 strategy doc (historical)
├── data/                      cached parquet feature stacks (gitignored)
├── src/
│   ├── features/              feature engineering modules
│   ├── models/                base-model architectures + trainers
│   └── utils/                 CV harness, EO solver, common helpers
├── scripts/                   winning-path training/inference scripts
├── notebooks/                 exploratory notebooks
├── oof/                       OOF probabilities + EO outputs (gitignored)
├── submissions/               generated CSVs + log.md
├── reports/                   docs + figures + final report
└── experiments_archive/       negative-result experiments (preserved)
```

## Key technical choices (one-line summaries)

| Choice | Rationale |
|---|---|
| **GroupKFold(5, groups=user_id)** | Mandatory — train/test users disjoint, naive CV would leak |
| **Hierarchical decomposition** (coarse 3-way × fine binary × fine ternary) | Decouples easy boundaries (L0/walking/other) from the hard L1↔L2 sub-problem |
| **Equilibrium Optimizer feature selection** | Removed all 11 gravity features = user-orientation signatures that don't transfer to new users |
| **Per-class isotonic + log-multiplier thresholds** | Recovers minority-class recall lost by softmax bias toward majority classes |
| **α-blend of flat-LGBM (P1) + hierarchical (P2)** | P1 = broad signal, P2 = regularization against user-shift |
| **Multi-seed averaging** (seeds 17, 23, 41) | Variance reduction across LGBM seed sensitivity |
| **L1↔L2 contrastive triplet-MLP embedding** | Targeted boundary-fix attempt; small but positive contribution |

See `reports/architecture.md` for the full architecture diagram and exact parameters.
