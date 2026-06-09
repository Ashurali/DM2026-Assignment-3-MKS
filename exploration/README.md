# `exploration/` — exploratory, negative-result, and superseded scripts

These scripts are **not part of the winning pipeline and not needed to reproduce the
reported result.** They are kept for transparency: they are the experiments behind the
"Negative results" discussion in the report (`report/report.tex`) and the dead-ends/earlier
iterations from development.

> **To reproduce the report's result, you do not need anything in this folder.**
> The winning path lives in [`../scripts/`](../scripts/); run `python scripts/reproduce_final.py`
> (see the top-level [`README.md`](../README.md)).

## Why these are separated, not deleted
The report honestly reports several approaches that were tried and **did not** improve the
score (ROCKET, InceptionTime, GAF/spectrogram CNNs, GRU/evidential models, domain-generalization
/ contrastive injection). Keeping the code makes those claims auditable. They were moved out of
`scripts/` only so a reader can immediately tell the winning path apart from the exploration.

## What's here (by theme)
| Theme | Scripts |
|---|---|
| Orientation "pseudo-gyro" variants & robust injection | `orient_integrate`, `orient_ms_winner`, `orient_strong_source`, `orient_traj_integrate`, `rebuild_finewalk_orient`, `stack_orient_gaf`, `make_robust_injection`, `make_robust_variants`, `max_tried_injection` |
| ROCKET / extra views (did not stack) | `rocket_rich`, `rocket_rich_v2`, `rocket_integrate`, `gen_gaf_hedge` |
| Hierarchical — earlier versions / probes | `train_hier_coarse`, `train_hier_fine_walk`, `train_hier_fine_other`, `hierarchy_vs_flat` |
| Domain generalization / contrastive (no gain) | `dg_feasibility`, `dg_gated_integrate`, `dg_gated_rescue`, `avg_dg_seeds`, `integrate_source` |
| L1↔L2 separability analysis | `l1l2_deep_dive`, `l1l2_headroom`, `l1l2_operating_point`, `l2_typology_bayes`, `headroom_and_orient_scope`, `unsup_separability_probe` |
| Calibration / threshold search | `calibrate`, `post_hoc_threshold`, `rigorous_threshold_search_v6`, `threshold_research_injected`, `ea_recalibrate_p2` |
| Label-shift / adversarial pruning probes | `gapA_label_shift`, `gapB_adversarial_pruning` |
| Blends & candidate-submission generators | `blend`, `blend_v6_plus_gru`, `gen_lb_candidates`, `gen_lb_round2`, `gen_lb_round3`, `gen_lb_round4` |
| Misc / utilities | `try_extratrees`, `summarize_ablation`, `summarize_runs` |
| Shell runners (server batch jobs) | `run_ablation.sh`, `run_dg_v2_queue.sh`, `run_gru_ms.sh`, `run_in_tmux.sh`, `run_phase3_all.sh` |

## Note on paths
A few of these scripts reference the **original `scripts/` layout** in `python scripts/...`
commands (the `.sh` runners) or import the kept module `from scripts.orient_pseudogyro_model`
(which still resolves, since `scripts/` is unchanged). They are historical and not maintained.
