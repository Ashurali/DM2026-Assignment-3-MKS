# TASK_STATE — IEEE Final Report (DM2026 Assignment 3, Group 2)

> Execution-state tracker for producing the final **IEEE-format `.tex` report**. Read this FIRST every session.
> (Separate from `DEV_PROGRESS.md`, which is the project development log deliverable.)

## Overall status
**ALL PHASES DONE; agent-verified clean (zero discrepancies). ~100%** — only open item is author names.
Deliverable `report/report.tex` compiles to **6 pages**, IEEE conference template, all 4 graded questions
answered, reproducible. PDF at `report/report.pdf`.

## Completed
- [x] **Phase 0 — setup.** Copied IEEE template into `report/`; confirmed TeX Live 2026.
- [x] **Phase 1 — `DEV_PROGRESS.md`** (feature-by-feature dev log with LB impact).
- [x] **Phase 2 — results gathered.** All numbers in "Key data" below (Kaggle ladder, CV baselines, feature
  ablation, pipeline-stage OOF/LB, test-prior shift) — verified from repo docs / agent-fact-checked.
- [x] Underlying content: `reports/experiments.md`, `README.md`, `scripts/reproduce_final.py`. (Old `reports/final_report.md` removed — superseded by the IEEE `report/report.tex`.)

- [x] **Phase 3 — presentation plan.** Table I = Kaggle progression; Table II = feature-group ablation;
  Table III = pipeline-stage ablation (OOF+LB). Fig 1 = `pipeline.png`; Fig 2 = `kaggle_progression.png`.
- [x] **Phase 4 — figures generated & verified** (`report/figures/pipeline.png`, `kaggle_progression.png`).

- [x] **Phase 5–6 — `report/report.tex` written** (IEEE conference). All four graded questions are explicit
  subsections in Experiments (Q1 preliminary, Q2 preprocessing+gains, Q3 temporal/label alignment, Q4
  ablation). 3 figures + 5 tables + equations + pseudocode + Discussion + Conclusion. 9 real refs in `reference.bib`.

- [x] **Phase 7 — verify (DONE).** Compiles via `latexmk`: 6 pages, 0 undefined, 0 overfull, preamble
  unchanged. **Cold-agent fact-check returned ZERO discrepancies** — every number matches a primary source,
  reproduction passes, citations resolve, report↔code↔Kaggle consistent. Verdict: safe to submit.

## In progress
- None. Report complete and pushed.

## Next
- **Only remaining item:** fill real author names in `report/report.tex` `\author{}` block (currently
  "Member 1/2/3"); Group ID 2 is already stated in the abstract.

## Key data & decisions
- **Group ID: 2** · GitHub: https://github.com/Ashurali/DM2026-Assignment-3-MKS · Kaggle team 314540066.
- **Best public LB: 0.8234** (`sub_pc_b20.csv`). Reproduce: `python scripts/reproduce_final.py`.
- **Format (HARD RULE — do not modify):** IEEE conference (`\documentclass[conference]{IEEEtran}`); no font/spacing
  changes. Sections: Abstract (must state Group ID 2 + GitHub) / Project Summary ≤1 pg / Proposed Method 1–3 pp /
  Experiments 2–4 pp. References via `reference.bib` (real sources only).
- **The 4 graded questions (10% each), to answer in Experiments:** (Q1) preliminary analysis + naive baselines;
  (Q2) preprocessing techniques + per-technique gain; (Q3) label↔sequence temporal alignment; (Q4) ablation.

### Kaggle public-score ladder (user-provided, verified) — the Experiments backbone
| Submission | Public | Stage / what it adds |
|---|---|---|
| `sub01_majority` | 0.1024 | naive majority-class baseline |
| `sub02_lgbm_basic` | 0.7473 | LightGBM, basic features |
| `sub_lgbm_full_v1_tuned` | 0.7808 | LightGBM, 271-feature catalog + thresholds |
| `sub_lgbm_full_tuned_v1_tuned` | 0.7816 | + Optuna tuning |
| `sub_lgbm_combo_combo_full` | 0.7568 | combo stacker (raw) |
| `sub_lgbm_combo_combo_full_cal_thresh` | 0.7924 | + per-class isotonic + threshold |
| `sub_lgbm_combo_combo_full_v2_cal_thresh` | 0.7984 | combo v2 + cal+thresh |
| `sub_hier_v4_a088_cal_thresh` | 0.8114 | hierarchical pipeline v4 |
| `sub_hier_v6_a842_grid_peak` | 0.8154 | hierarchical v6 + (L1,L2) threshold grid (long-standing best) |
| `sub_dg_cisc_*_gated` | 0.8154 | domain-generalization injection — no gain |
| `sub_orient_lgbm_ms_gated_w10` | 0.8184 | orientation pseudo-gyro (multi-seed) |
| `sub_robust_orient_inject_w15` | 0.8200 | orientation L2-injection (robust thresholds) |
| `sub_robust_orient_L2_priorcorr` | 0.8220 | + test-prior correction (Saerens, β=1) |
| `sub_pc_b13` | 0.8220 | prior β=1.3 |
| **`sub_pc_b20`** | **0.8234** | **prior β=2.0 — BEST submission** |
| `sub_pc_b25` | 0.8214 | prior β=2.5 (overshoot) |
| `sub_dec_L3hi` | 0.8212 | decoupled L3 boost |

Negatives kept concise: GRU/evidential 0.71–0.72, `blend_dlx30` 0.7734, `v6_eaP2` 0.7990.

### Other verified numbers (from repo docs — all re-derivable)
- **CV naive baselines** (GroupKFold-5, `reports/eda_summary.md`): majority 0.1331, LR(6 means) 0.5315, LGBM(271) OOF 0.7091.
- **Feature-group ablation** (`reports/ablation_features.md`): Δmacro jerk −0.0088, subwindow −0.0068, zerocross −0.0060,
  basic_stats −0.0050, fft/magnitude −0.0041, autocorr −0.0031, gravity −0.0027, per_file_norm −0.0010 (noise floor |Δ|<0.005).
- **Final-pipeline OOF:** blend+isotonic+threshold (peak) 0.7880; orient-inject (robust) 0.7873; priorcorr β=1 0.7808; β=2 0.7758.
- **Data:** 6 classes; train 11,020 files / 60 users; test 6,849 / 40 users; disjoint users; 1 Hz aggregated
  (mean/std × 6ch × 300 s); gravity NOT removed; no gyro. Class counts {0:4643,1:4695,2:358,3:656,4:142,5:526}.
- **Test-prior shift (Saerens, label-free):** test L2≈0.041 (train 0.033), L3≈0.073 (train 0.060) → prior-correction β=2.0.

## Open questions for Michael
1. **Author names** for the IEEE `\author{}` block (template has "Member 1/2/3"). Group ID is 2 — who are the members?
   (Proceeding with placeholders + Group ID 2; please provide names before final submission.)
2. Confirm `report/report.tex` is the intended location (kept inside the repo so it compiles standalone with the copied `IEEEtran.cls`).
