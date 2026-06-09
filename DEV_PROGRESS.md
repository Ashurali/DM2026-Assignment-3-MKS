# DEV_PROGRESS — Development Log (by feature)

Project development from start to finish, organized by the feature/idea added at each step, with its
leaderboard (LB) or out-of-fold (OOF) impact. Derived from the git history and `reports/experiments.md`.
Metric throughout = macro-F1; CV = `GroupKFold(5, groups=user_id)`.

---

### 1. Setup, EDA & naive baselines  *(May 6–8)*
Repo skeleton, data loaders, and exploratory analysis. Established the five facts that drove everything:
pristine data (300 rows/file, 0 NaN), **33× class imbalance**, **disjoint train/test users** (→ GroupKFold
mandatory, `user_id` forbidden), a clear intensity gradient with **L2 buried inside the dominant L1**, and
1 Hz aliasing of gait. Baselines: majority **0.1024** (Kaggle) / 0.1331 (CV); logistic on 6 means 0.5315 CV.

### 2. Engineered feature catalog  *(May 7)*
Built an 11-family, 271-feature catalog (basic-stats, FFT, autocorr, sub-window pooling, jerk, zero-cross,
magnitude, gravity, cross-axis, per-file-norm, quality) + catch22. **LightGBM jumps to 0.7091 CV** (+0.176
over LR); `sub02_lgbm_basic` → `sub_lgbm_full_v1_tuned` = **0.7473 → 0.7808 LB**.

### 3. Tuning: Optuna + SMOTE + post-hoc thresholds  *(May 7)*
Optuna HP search, bounded SMOTE, and per-class log-multiplier thresholds to recover minority recall.
`sub_lgbm_full_tuned_v1_tuned` = **0.7816 LB**. Threshold tuning validated as a reusable lever.

### 4. Base-model stacking (`combo`) + deep-learning ensemble  *(May 7–8)*
Added diverse base models — CNN-BiLSTM, Transformer, MiniRocket, XGBoost, CatBoost, InceptionTime — and a
LightGBM **stacker over their OOF probabilities + features** (`combo_full_v2`, 805 cols). Raw combo
underperformed on the LB (train/test meta-feature shift): `sub_lgbm_combo_combo_full` = 0.7568.

### 5. Per-class isotonic calibration + threshold grid  *(May 8)*
Calibrating the combo's probabilities (5-fold OOF isotonic) and re-thresholding fixed the shift and lifted
the LB sharply: `sub_lgbm_combo_combo_full_cal_thresh` **0.7924**, `...v2_cal_thresh` **0.7984**.

### 6. Hierarchical pipeline + EO feature selection + α-blend  → **0.8154**  *(early May)*
Decomposed the problem into a coarse→fine hierarchy (Coarse → Fine_walk[LGBM+XGB] → Fine_other), pruned
user-orientation-spurious features with an Equilibrium-Optimizer search, and α-blended the flat stacker (P1)
with the hierarchical pipeline (P2) at 0.842/0.158, then applied the (L1,L2) threshold grid.
`sub_hier_v4_a088_cal_thresh` 0.8114 → **`sub_hier_v6_a842_grid_peak` 0.8154** (long-standing best; OOF 0.7880).

### 7. Domain-generalization attempts  *(late May — negative)*
Conditional cross-user supervised-contrastive (DG/CISC), GRU, and evidential models, injected under frozen
thresholds. All **failed to beat 0.8154** (`sub_dg_cisc_*_gated` = 0.8154; GRU 0.71–0.72). Diagnosis: the
L1↔L2 confusion is *intrinsic class overlap*, not a user-spurious attribute.

### 8. Orientation "pseudo-gyroscope" L2-injection  → **0.8200**  *(May 31)*
Because gravity is not removed and the device is wrist-worn, the per-second `mean_x/y/z` trace the wrist's
**orientation trajectory**; its time-derivative approximates the **missing gyroscope**. A LightGBM on these
orientation-dynamics features, injected (gated, nested-CV weight) into the L2 column, gave the project's
first real gain: `sub_orient_lgbm_ms_gated_w10` 0.8184 → **`sub_robust_orient_inject_w15` 0.8200**.

### 9. Test-prior (label-shift) correction  → **0.8234**  *(Jun 3 — final)*
Measured that the **test set has a different class prior** than train (Saerens-EM, label-free: L2≈0.041 vs
0.033, L3≈0.073 vs 0.060). Multiplying the calibrated test posteriors by `(test_prior/train_prior)^β` before
thresholding adapts to it: `sub_robust_orient_L2_priorcorr` (β=1) **0.8220** → **`sub_pc_b20` (β=2.0) 0.8234**
(β=2.5 overshoots to 0.8214). Notably this candidate had the *lowest* OOF (0.7758) yet the *highest* LB — the
OOF↔LB gap is the prior shift, validated only by submitting (the project's "submit, don't infer" lesson).

### Also explored, not in the final pipeline *(all confirmed the operating-point/Bayes-error ceiling)*
ROCKET/MultiRocket on orientation-rich channels, a from-scratch InceptionTime, 2-D CNNs on
Gramian-Angular-Field / spectrogram images, adversarial-validation feature pruning, and Group-DRO-style
robustness. Each carried isolated complementary L2 signal but **none stacked** — every source competes for the
same ~10–15 ambiguous L1↔L2 boundary samples.

**Final result: public LB 0.8234** (`sub_pc_b20.csv`), reproducible via `python scripts/reproduce_final.py`.
