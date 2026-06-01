# Project Index — DM2026 Assignment 3 (HAR, disjoint-user)

Master map of the architecture, results, code, and documents. Start here.

**Status:** public LB **0.8200** (top 4). Private LB via 2 final picks (see §5).

---

## 1. Problem & data
- **Task:** 6-class human-activity recognition (Bruno et al. 2014 wrist-worn ADL dataset).
- **Signal:** 1 Hz *aggregated* accelerometer — per-second `mean_x/y/z, std_x/y/z` × 300 s. Gravity **not** removed; wrist-worn.
- **Core difficulty:** train/test **users are disjoint** → cross-subject distribution shift. **L2** (3.2%, a walking-ish ADL) is the structural bottleneck — confused with the dominant **L1** (62% of L2 have an L1 nearest-neighbour in feature space).
- **Hard rules:** no external data; train on the server, not locally.

## 2. Production architecture (signal flow)
```
RAW (1Hz, 6ch×300, gravity-laden, wrist)
 → FEATURES: 271 stats + catch22 + 12 families (orientation used STATICALLY)
 → BASE: P1 = LGBM combo stacker (805 cols)   P2 = hier (Coarse→Fine_walk[LGBM+XGB]→Fine_other)
 → BLEND  α=0.842·P1 + 0.158·P2
 → per-class ISOTONIC (5-fold GroupKFold OOF)
 → (L1,L2) THRESHOLD GRID  (peak / robust configs)
 → + ORIENTATION L2-INJECTION (post-hoc, nested-CV w)
```
Reproduces from **frozen OOFs** deterministically. Thresholds in `oof/threshold_grid_v6_meta.json`.

## 3. LB progression
| Submission | Public LB | What |
|---|---|---|
| `sub_hier_v6_a842_grid_peak` | 0.8154 | production peak (long-standing best) |
| `sub_hier_v6_a842_grid_robust` | 0.8114 | production robust thresholds |
| `sub_dg_cisc_orient_lgbm_gated_w15` | **0.8184** | peak + orientation L2-injection |
| `sub_orient_lgbm_ms_gated_w10` | 0.8184 | multi-seed orientation (robustness ✓) |
| **`sub_robust_orient_inject_w15`** | **0.8200** | **robust + orientation injection — best** |

## 4. The L2-separation arc (this session)
Full detail: `experiments.md` (rows 16–28 + verdicts) and `findings_and_solution.md`.
- **Diagnosis:** the problem is *learn L2/L3/L5 while removing the user-spurious attribute* → conditional subject-invariant representation. GBDT can't do this; only representation learning can.
- **DG model (CISC):** CNN-BiGRU + *conditional cross-user supervised-contrastive* loss (pulls same-class-different-user together). Carries complementary L2 signal (rescues 40% of production's L2 misses; oracle 0.846) → gated L2-injection +0.0008 OOF.
- **Orientation pseudo-gyro (the win):** wrist + gravity-not-removed ⇒ `mean_x/y/z` trace the wrist orientation trajectory; its derivative ≈ the missing gyroscope. Summary-stat dynamics (angular speed, reversals, CoV) → injection → **0.8200 LB**. The injection helps *robust* thresholds more than peak.
- **Negatives recorded:** stronger/richer source = flat; multi-seed confirms-but-doesn't-exceed; threshold re-search converges back to peak. OOF prefers peak (0.7887) while public prefers robust (0.8200) — *they diverge*, hence the 2-pick hedge.
- **Open lever (next build):** orientation used only as **summary stats**; the time-resolved tilt *trajectory* (motion shape) is the untried representation. See §8.

## 5. Final submissions (private LB, pick 2)
Both carry the validated injection; they hedge the OOF↔public divergence:
1. **`sub_robust_orient_inject_w15.csv`** — 0.8200 public (public-best, conservative thresholds)
2. **`sub_dg_cisc_orient_lgbm_gated_w15.csv`** — 0.8184 public / 0.7887 OOF (OOF-best, peak thresholds)

## 6. Code map
**Production pipeline** (`scripts/`): `train_hier_coarse/_fine_walk/_fine_other`, `train_hier_multi_seed`, `train_hier_v4_and_submit`, `train_hier_v6_eo_selected`, `eo_feature_select`, `threshold_grid_v6`, `rigorous_threshold_search_v6`, `train_l1l2_contrastive`. Combo: `src/models/train_lgbm_combo.py`. Features: `src/features/{basic,build,catch22_features}.py`.

**L2-separation arc** (`scripts/` + `src/`):
- DG: `src/models/cnn_bigru.py`, `src/models/train_dg_cisc.py`, `src/utils/cond_supcon.py`, `dg_feasibility.py`, `dg_gated_integrate.py`, `dg_gated_rescue.py`, `avg_dg_seeds.py`, `run_dg_v2_queue.sh`.
- GRU/EA (earlier, negative): `train_gru.py`, `train_gru_evidential.py`, `src/utils/evidential_align.py`, `blend_v6_plus_gru.py`, `ea_recalibrate_p2.py`.
- **Orientation** (the win): `orient_pseudogyro_model.py` (feature builder `build_orient` + A/B), `orient_integrate.py`, `orient_ms_winner.py`, `orient_strong_source.py`, `rebuild_finewalk_orient.py`, `make_robust_injection.py`, `threshold_research_injected.py`.
- Throwaway debug: `experiments_archive/_*.py` (gitignored).

## 7. Reports
- `00_INDEX.md` (this file) · `experiments.md` (chronological log + verdicts) · `findings_and_solution.md` (diagnosis + solution thesis) · `architecture.md` · `class_structure.md` · `data_structure.md` · `eda_summary.md` / `eda_deep_summary.md` · `ablation_features.md` · `literature_synthesis.md` · `final_report.md`.

## 8. Next build — tilt-trajectory L1↔L2 model
Replace the raw-6-channel sequence input (capped neural at 0.69 — noisy/gravity-dominated) with a **derived, clean, low-dim orientation trajectory** (pitch/roll/inclination(t), angular-speed(t), intensity(t)) fed to a focused **1D-CNN or Shapelet-Transform** as a dedicated L1↔L2 model; inject its L2 with nested-CV `w`. Targets the representation-layer bottleneck (motion-shape), not another downstream band-aid. Set expectations: 1 Hz/no-gyro floor still binds; the summary-stat version already proved the signal is real (+0.0046 LB).

## 9. Reproduction / ops notes
- **Server:** `nycu813@140.113.86.130` (`ICCL-S3-251230`), repo `~/mike/DM2026-Assignment-3-MKS`, conda env `dm2026-a3` at `~/anaconda3/envs/dm2026-a3`. **No git on server** — sync via scp.
- **Launch gotcha:** background jobs die on SSH channel close — use `setsid` (not just `nohup`). `pgrep -fc <pat>` self-matches; use `pgrep -fc "[p]at"`.
- **Threshold trap:** identical OOF → LB 0.7698/0.7991/0.8154 by thresholds alone. Always validate recalibrations under FROZEN thresholds + nested CV.
