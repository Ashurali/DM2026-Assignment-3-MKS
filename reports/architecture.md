# Final architecture — DM2026 Assignment 3 (HAR)

> Best public LB: **0.8154** (primary, sub_hier_v6_a842_grid_peak.csv).
> Backup: **0.8114** (sub_hier_v4_a088_cal_thresh.csv).
> Rank #3 of public leaderboard (top is 0.8240).

## 1. Pipeline overview

Both submissions share the same blend skeleton:

```
                ┌──────────────────────────────┐
                │     Pipeline 1 (always       │
   Train data ──┤  combo_full_v2 — flat LGBM)  ├── P1 probs (N, 6)
                └──────────────────────────────┘
                                                       \
                                                        →  α·P1 + (1−α)·P2  →
                                                       /
                ┌──────────────────────────────┐      /
   Train data ──┤  Pipeline 2 (hierarchical)   ├── P2 probs (N, 6)
                │  Coarse 3-way × Fine_walk ×  │
                │  Fine_other, multi-seed      │
                └──────────────────────────────┘

                           │
                           ▼
                ┌──────────────────────────────┐
                │  Per-class isotonic calibr.  │
                │  (5-fold OOF + full-fit)     │
                └──────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────────────┐
                │  Per-class log-multiplier    │
                │  threshold tuning            │
                └──────────────────────────────┘
                           │
                           ▼
                       argmax → Label
```

Only **Pipeline 2's feature stack** and **the final (α, thresholds)** differ between primary and backup.

## 2. Pipeline 1 — combo_full_v2 (identical for both)

Single LGBM multiclass classifier, 5-fold GroupKFold on `user_id`.

**Feature stack (805 cols):**

| Block | Cols | Source |
|---|---|---|
| engineered | 271 | `feat_train_none.parquet` — hand-crafted stats (mean, std, percentiles, FFT, jerk, sliding windows, gravity, etc.) |
| catch22 | 132 | `feat_catch22_train.parquet` — 22 features × 6 channels |
| cnn_emb | 256 | `oof/cnn_bilstm_v1_emb_train.npy` — CNN-BiLSTM penultimate layer (OOF, per-fold) |
| transformer_emb | 128 | `oof/transformer_v1_emb_train.npy` |
| oof_xgb_v1 | 6 | base XGB OOF class probs |
| oof_cat_v1 | 6 | base CatBoost OOF class probs |
| oof_minirocket_v1 | 6 | MiniROCKET base OOF probs |

LGBM params: multiclass, num_leaves=63, learning_rate=0.04, n_estimators=600, feature_fraction=0.7, bagging_fraction=0.8.

Raw OOF F1: ~0.74 macro. Standalone LB: 0.7984.

## 3. Pipeline 2 — hierarchical decomposition

Three sub-classifiers composed into a 6-class prediction:

```
                          Coarse 3-way
                          (multi-seed LGBM)
                          ┌─ P(L0)
                          ├─ P(walking) = P(L1∪L2)
                          └─ P(other)   = P(L3∪L4∪L5)
                                │
                                ├──────────── × Fine_walk (LGBM+XGB) ──→  P(L1), P(L2)
                                │
                                └──────────── × Fine_other (LGBM)    ──→  P(L3), P(L4), P(L5)

P2[k] = P(parent_of_k) × P(k | parent_of_k)
```

**Multi-seed averaging**: each sub-classifier trained 3× with seeds {17, 23, 41}, probs averaged.

Stage F1s (v6 primary):
- Coarse 3-way OOF F1: 0.9046
- Fine_walk binary OOF F1 (on walking): 0.577 (L2 F1 alone 0.190)
- Fine_other ternary OOF F1: 0.897

## 4. Feature stacks — primary vs backup difference

### Primary (v6) — EO-selected feature stack (596 cols)

| Block | Cols | Note |
|---|---|---|
| engineered | **86 of 271** | EO-selected; **all 11 gravity features dropped** |
| catch22 | **38 of 132** | EO-selected (~30% kept) |
| cnn_emb | 256 | always-on |
| transformer_emb | 128 | always-on |
| oof_xgb_v1 | 6 | |
| oof_cat_v1 | 6 | |
| oof_minirocket_v1 | 6 | |
| oof_combo_full_v2 | 6 | stacking-on-stacking |
| l1l2_contrast_emb | 64 | triplet-MLP embedding targeting L1↔L2 |

### Backup (v4) — full feature stack (869 cols)

Same as above but **no EO mask** — all 271 engineered + all 132 catch22 features retained. The l1l2_contrast_emb (64) is also included.

## 5. Calibration & threshold tuning

After the α-blend:

1. **Per-class isotonic regression**, fit out-of-fold:
   - For each fold k, for each class c, fit `IsotonicRegression` on `(blend_oof_probs[tr_idx, c], (y[tr_idx] == c))`. Predict on `blend_oof_probs[va_idx, c]`. Re-normalize rows.
   - For test: full-fit isotonic per class on the OOF, apply to test.

2. **Per-class log-multiplier threshold tuning**:
   - Search 6-D vector `log_w` such that `(cal_probs * exp(log_w)).argmax(axis=1)` maximizes OOF macro-F1.
   - **Primary**: Found via 51×51 fine grid over (L1, L2) with other 4 classes pinned at Nelder-Mead optima.
   - **Backup**: 6-D Nelder-Mead with 9 random starts.

## 6. Final parameters — exact numbers

### Primary: `sub_hier_v6_a842_grid_peak.csv` (LB 0.8154)

- **α**: 0.842 (auto-tuned on OOF before grid search)
- **Class multipliers** `w = exp(log_w)`:

| Class | Multiplier |
|---|---|
| L0 | ~1.51 |
| L1 | **0.741** (grid peak) |
| L2 | **2.460** (grid peak) |
| L3 | ~1.589 |
| L4 | ~0.787 |
| L5 | ~0.651 |

- **L1 and L2 multipliers came from the threshold_grid_v6.py 31×31 grid peak; others came from NM optimization.**
- **OOF F1: 0.7880**
- **Test prediction distribution**: ~L2 count 230 (vs train base rate ~222)

### Backup: `sub_hier_v4_a088_cal_thresh.csv` (LB 0.8114)

- **α**: 0.880 (fixed, LB-validated)
- **Class multipliers**:

| Class | Multiplier |
|---|---|
| L0 | 1.771 |
| L1 | 0.884 |
| L2 | 2.263 |
| L3 | 1.470 |
| L4 | 0.452 |
| L5 | 2.296 |

- All from 9-start Nelder-Mead.
- **OOF F1: 0.7856**

## 7. Why each piece works

| Component | Role |
|---|---|
| **Hierarchical decomposition (P2)** | Decouples the easy L0 vs walking vs other boundary from the hard L1↔L2 sub-problem. Each stage gets a focused training signal. |
| **EO feature selection (primary only)** | Drops user-signature features (e.g., gravity orientation, which encodes how each user holds their wrist). Cleaner generalization to disjoint test users. |
| **L1↔L2 contrastive emb** | A 64-d MLP embedding trained with triplet loss targeting L1↔L2 separation. Provides an additional signal for the stacker. |
| **Multi-seed averaging** | Variance reduction across LGBM seed sensitivity. Improves OOF stability. |
| **α-blend (P1 + P2)** | P1 (combo_v2) gives broad accuracy; P2 (hierarchical) regularizes against user-shift. Optimal α weights P1 more heavily. |
| **Per-class isotonic calibration** | Corrects systematic miscalibration in LGBM softmax probabilities (esp. minority class under-prediction). |
| **Log-multiplier threshold tuning** | Recovers minority class recall by upweighting their probabilities before argmax. Grid Peak (primary) used a focused L1×L2 sweep; the specific (0.741, 2.460) combination transferred dramatically better than NM's (0.884, 2.263). |

## 8. Reproduction commands

```bash
# Stage 1-2 base models (cached as OOF .npy files; if rebuilding):
python src/models/train_cnn_bilstm.py
python src/models/train_transformer.py
python src/models/train_xgb_cat.py
python src/models/train_minirocket.py

# combo_full_v2 (Pipeline 1):
python src/models/train_lgbm_combo.py --name combo_full_v2

# Contrastive emb:
python scripts/train_l1l2_contrastive.py --gpu --epochs 200 --margin 0.4

# EO feature selection (RAN ON SERVER, 7 hours on 24-core CPU):
python scripts/eo_feature_select.py \
  --particles 25 --iterations 50 --inner-folds 3 --n-estimators 150 \
  --min-features 50

# Primary submission: v6 hierarchical + EO mask + grid peak thresholds
python scripts/train_hier_v6_eo_selected.py --seeds 17 23 41
python scripts/threshold_grid_v6.py    # produces sub_hier_v6_a842_grid_peak.csv

# Backup submission: v4 hierarchical (no EO mask) + NM thresholds
python scripts/train_hier_v4_and_submit.py --gpu --seeds 17 23 41
# This produces sub_hier_v4_a088_cal_thresh.csv directly.
```

## 9. Key OOF / LB metrics summary

| Variant | OOF F1 | L2 F1 | Public LB | Notes |
|---|---|---|---|---|
| combo_full_v2 alone | 0.7733 raw | — | 0.7984 | Pipeline 1, no hierarchy |
| v1 hier α=0.88 | ~0.78 | ~0.39 | 0.8107 | First hierarchical |
| v4 a088 cal+thresh | 0.7856 | 0.377 | **0.8114** | Backup submission |
| v6 a842 NM | 0.7880 | 0.388 | 0.7991 | EO mask, NM thresholds (under-tuned) |
| **v6 a842 grid peak** | **0.7880** | **0.384** | **0.8154** | **PRIMARY — grid found better thresholds** |
| v6 a900 grid peak | 0.7882 | 0.398 | 0.7698 | Pushed α too high — collapsed |

**Key insight**: The same OOF (0.7880) can map to very different LBs (0.7991 to 0.8154 to 0.7698) depending on the specific (α, thresholds) combination. The OOF→LB transfer is highly local in threshold space due to the disjoint test users.
