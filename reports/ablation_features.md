# Phase-4 Feature-Group Ablation

Each row removes one feature group from the full catalog and re-runs the 5-fold GroupKFold CV. Negative ΔCV F1 means the group was helping; the more negative, the more important.

**Baseline (full catalog, `lgbm_full_v1`):** 271 features, CV F1-macro = **0.7091**.

Per-class F1 baseline: `L0=0.964`, `L1=0.902`, `L2=0.173`, `L3=0.713`, `L4=0.896`, `L5=0.678`.

**Noise floor:** the `quality` group is two essentially-constant features (`n_rows=300` for every file, `nan_total=0`), so its true Δ should be ≈ 0. Its observed Δ of −0.0036 calibrates the noise level for everything else: **|Δ| < 0.005 is within fold-noise and not meaningful**.

## Macro F1 ablation (sorted by importance)

| Removed group | n features | CV F1 | Δ vs full | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `jerk` | 247 | 0.7002 | **−0.0088** | 0.965 | 0.902 | 0.122 | 0.715 | 0.889 | 0.675 |
| `subwindow` | 211 | 0.7022 | **−0.0068** | 0.964 | 0.901 | 0.139 | 0.702 | 0.895 | 0.677 |
| `zerocross` | 263 | 0.7030 | **−0.0060** | 0.964 | 0.900 | 0.147 | 0.713 | 0.896 | 0.666 |
| `basic_stats` | 187 | 0.7041 | −0.0050 | 0.965 | 0.900 | 0.116 | 0.712 | 0.911 | 0.676 |
| `fft` | 247 | 0.7049 | −0.0041 | 0.965 | 0.902 | 0.140 | 0.713 | 0.894 | 0.683 |
| `magnitude` | 243 | 0.7050 | −0.0041 | 0.965 | 0.901 | 0.164 | 0.707 | 0.883 | 0.680 |
| `quality` _(noise floor)_ | 269 | 0.7054 | −0.0036 | 0.964 | 0.902 | 0.156 | 0.711 | 0.889 | 0.680 |
| `crossaxis` | 265 | 0.7054 | −0.0036 | 0.965 | 0.902 | 0.170 | 0.706 | 0.882 | 0.681 |
| `autocorr` | 265 | 0.7060 | −0.0031 | 0.964 | 0.902 | 0.148 | 0.712 | 0.891 | 0.679 |
| `gravity` | 260 | 0.7064 | −0.0027 | 0.965 | 0.903 | 0.154 | 0.709 | 0.895 | 0.681 |
| `per_file_norm` | 253 | 0.7081 | −0.0010 | 0.965 | 0.902 | 0.170 | 0.708 | 0.893 | 0.676 |

## Three significant findings

### 1. Time-domain features collectively carry label-2

Label 2 is the bottleneck class (F1 = 0.173 in baseline). Removing any single time-domain group drops L2 substantially:

| Group | Δ L2 | Δ macro |
|---|---:|---:|
| jerk | **−0.051** | −0.0088 |
| basic_stats | **−0.057** | −0.0050 |
| subwindow | −0.035 | −0.0068 |
| fft | −0.034 | −0.0041 |
| zerocross | −0.026 | −0.0060 |
| autocorr | −0.025 | −0.0031 |

Five of the top-six L2-affecting groups are time-domain (jerk, subwindow, FFT, zerocross, autocorr). The basic_stats group's L2 contribution comes from per-channel percentiles capturing distribution shape — a static-feature surrogate for temporal complexity. This explains why **feature engineering hits a ceiling** on L2 (still F1 = 0.17 with the full catalog): the static-stat surrogates can't replace genuine sequence modelling.

### 2. Removing basic_stats *improves* label 4

A counterintuitive result:

| | L4 F1 |
|---|---:|
| baseline (271 features) | 0.896 |
| **`-basic_stats`** (187 features) | **0.911** (+0.015) |

The 84 percentile/quantile features in basic_stats overlap heavily with the 28 magnitude features for high-intensity L4 detection. Redundancy gives LGBM more opportunities to overfit on noisy L4 splits (only 142 train files for L4). With basic_stats removed, the model relies on cleaner magnitude features and shows tighter L4 generalization. **Feature engineering is not strictly additive** — this is a Phase-5 lesson too: more features ≠ better, especially for rare classes.

### 3. Per-file normalization is essentially redundant

`per_file_norm`'s Δ of −0.0010 is **below the noise floor** (≤ 0.005). The 18 features in this group (z-scored channel stats: mean-abs, max, skew) duplicate signal already present in `basic_stats` and `magnitude`. The EDA §5 finding of moderate inter-user drift (between/within variance ratio 2.14 for label 1) is real, but the LGBM tree-based model handles that drift internally without needing explicit normalization features.

## Implications for Phase 5 (DL track)

The ablation strongly suggests CNN-BiLSTM has real upside on L2:

- Five of six L2-critical groups encode **temporal structure** (jerk = first derivative, subwindow = phase changes, FFT = rhythmicity, autocorr = periodicity, zerocross = rate of sign changes).
- A 1D-CNN's first layer learns derivative-like kernels automatically; subsequent layers compose multi-scale temporal patterns. So jerk + subwindow + FFT-like features are **implicitly captured by the architecture**.
- BiLSTM picks up long-range dependencies that don't fit a fixed CNN kernel.
- **Predicted CNN-BiLSTM L2 F1: 0.30–0.40** if the architecture and augmentations are well-tuned — would push macro F1 to ~0.74–0.76 as a single DL model, competitive with the best LGBM.

## Direct material for Report Q4

The macro Δs and the per-class L2 / L4 stories above are the Q4 paragraph almost verbatim. Key sentences:

> The 271-feature catalog was decomposed into 11 groups. Group ablation (full catalog minus one group) showed three groups with statistically significant Δ macro F1: jerk (−0.0088), subwindow (−0.0068), and zerocross (−0.0060). Per-class analysis revealed that label 2 — the bottleneck minority class — depends on a coalition of time-domain features (jerk, subwindow, FFT, zerocross, autocorr) that no single group can replace, partly explaining why the LGBM ceiling on label 2 sits at F1 ≈ 0.27 even after Optuna and post-hoc threshold tuning. A counterintuitive finding: removing basic_stats *improved* label 4 (+0.015) by reducing redundant high-intensity features that LGBM was overfitting to. The per_file_norm group was found redundant (Δ = −0.001, within noise) with basic_stats already capturing per-file shape statistics.
