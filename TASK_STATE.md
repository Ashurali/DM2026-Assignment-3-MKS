# TASK_STATE — IEEE Final Report (DM2026 Assignment 3, Group 2)

> Execution-state tracker for producing the final **IEEE-format `.tex` report**. Read this FIRST every session.
> (Separate from `DEV_PROGRESS.md`, which is the project development log deliverable.)

## Overall status
**Phase 0 done; on Phase 1.** ~15%. Deliverable = `report/report.tex` (IEEE conference template), 5–8 pages
excl. refs, addressing the 4 graded questions, with real numbers + figures/tables, compilable via TeX Live 2026.

## Completed
- [x] **Phase 0 — Orientation/setup.** Read template (`report.tex`, `reference.bib`); copied `IEEEtran.cls` +
  `report.tex` + `reference.bib` into repo `report/`; confirmed TeX Live 2026 (pdflatex/xelatex/bibtex/latexmk).
- [x] Underlying content already exists & was independently agent-fact-checked: `reports/final_report.md`
  (markdown report), `reports/experiments.md` (full chronological log), `README.md` (run instructions),
  `scripts/reproduce_final.py` (deterministically reproduces the 0.8234 submission).

## In progress
- **Phase 1 — `DEV_PROGRESS.md`** (dev log organized by feature, from git history + experiments.md).

## Next
- Phase 2 — confirm all results/numbers (mostly already gathered; see Key data).
- Phase 3 — data-presentation plan (which tables/figures).
- Phase 4 — generate figures (Kaggle-progression plot, per-class/ablation) + assemble tables.
- Phase 5 — writing plan (section-by-section map to the spec).
- Phase 6 — write `report/report.tex` (IEEE, 5–8 pp, 4 graded questions, figures/tables).
- Phase 7 — verify: compile, page count 5–8, consistency report↔code↔Kaggle, agent fact-check, no hallucinations.

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
