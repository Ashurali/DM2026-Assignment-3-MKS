# DM2026 Assignment 3 — Human Activity Recognition

NYCU 535703 Data Mining (Spring 2026), Assignment 3.

Kaggle competition: [`nycu-data-mining-assignment-3`](https://www.kaggle.com/competitions/nycu-data-mining-assignment-3)
Kaggle display name: **314540066**

## Status — public leaderboard

**Best public LB = 0.8234** (`submissions/sub_pc_b20.csv`). Progression of the winning path:

| Submission | Public LB | Key addition |
|---|---|---|
| `sub_hier_v6_a842_grid_peak.csv` | 0.8154 | GBDT stack + hierarchical blend + per-class isotonic + (L1,L2) threshold grid |
| `sub_robust_orient_inject_w15.csv` | 0.8200 | + orientation "pseudo-gyro" L2-injection |
| `sub_robust_orient_L2_priorcorr.csv` | 0.8220 | + test-prior correction (Saerens label-shift, β=1) |
| **`sub_pc_b20.csv`** | **0.8234** | **+ stronger test-prior correction (β=2.0)** |

Baseline-3 = 0.7088; the final submission beats it by **+0.11**. Reproduce in one command (below).

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

## ✅ Quick reproduction (no training — verifies the 0.8234 result in < 1 min)

The frozen base-model OOF + test probabilities are committed in `oof/`, so the winning
submission regenerates **deterministically from a clean clone, with no GPU and no retraining**:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/reproduce_final.py          # -> submissions/sub_pc_b20.csv  (== public LB 0.8234)
```

`reproduce_final.py` runs the full final pipeline (blend → per-class isotonic → Saerens
test-prior correction β=2.0 → orientation L2-injection → robust threshold) and asserts the
exact winning class counts (L2=314, L3=559). Expected console output ends with
`WROTE sub_pc_b20.csv … [reproduction verified]`.

## Full pipeline (from raw data — retrains every base model)

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

#   Primary base (EO-selected features, ~25 min):
python scripts/train_hier_v6_eo_selected.py --seeds 17 23 41
python scripts/threshold_grid_v6.py
#   → submissions/sub_hier_v6_a842_grid_peak.csv  (0.8154)

# Phase E — Final winning stages
python scripts/orient_pseudogyro_model.py   # orientation pseudo-gyro source -> oof/orient_lgbm_*.npy
python scripts/reproduce_final.py           # blend+isotonic+prior-correct+inject+threshold -> 0.8234
#   → submissions/sub_pc_b20.csv
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
| **Orientation "pseudo-gyro" L2-injection** | Gravity is *not* removed + wrist-worn ⇒ per-second mean traces wrist orientation; its derivative ≈ the missing gyroscope, giving complementary L2 signal (0.8154→0.8200) |
| **Test-prior correction** (Saerens label-shift, β=2.0) | Test has more L2/L3 than train; adapt posteriors to the *measured* test prior before thresholding (0.8200→0.8234). Estimated from the whole test set ⇒ robust for the private split |

See `reports/architecture.md` for the full architecture diagram and exact parameters.
