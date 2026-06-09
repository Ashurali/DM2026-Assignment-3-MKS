"""Rigorous α-sweep + fine threshold search for v6 (EO-selected) pipeline.

Lesson learned from the v6 NM vs grid Peak LB result:
  - Identical OOF (0.7880), but different thresholds gave LB 0.7991 vs 0.8154.
  - So the OOF→LB transfer depends meaningfully on WHERE in the OOF F1 plateau
    we pick the thresholds. Naive Nelder-Mead picks a local optimum; the
    plateau is wider than NM realizes.

Strategy:
  Layer 1: α sweep in {0.60, 0.625, ..., 0.95} → 15 blends.
  Layer 2: For each α, fit isotonic + run two threshold search methods on the
           resulting cal_oof:
             (a) 51×51 fine grid over (L1 mult, L2 mult), other classes pinned
                 at NM values.
             (b) 6-D Nelder-Mead with 16 random starts.
  Layer 3: Robustness re-ranking. Among all candidates within 0.0005 of the
           global best OOF F1, pick the one whose **test predicted class
           distribution** is closest to the **train base rates**. The intuition:
           if test users behave like train users on average, the optimal
           threshold is the one whose prediction distribution matches train.
           This is the signal we missed when NM picked v6 thresholds with too
           few L2 predictions.

Output: 5 submission CSVs for the top candidates, ranked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-min", type=float, default=0.60)
    p.add_argument("--alpha-max", type=float, default=0.95)
    p.add_argument("--alpha-step", type=float, default=0.025)
    p.add_argument("--grid", type=int, default=51,
                   help="Grid resolution per axis for L1×L2 sweep.")
    p.add_argument("--range", type=float, default=2.0,
                   help="Log-multiplier range = ±range.")
    p.add_argument("--nm-starts", type=int, default=16,
                   help="Multi-start count for 6-D Nelder-Mead.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of top candidates to write as submissions.")
    p.add_argument("--use-v4", action="store_true",
                   help="Also run the same search on v4 pipeline (p2 = v4 hier compose).")
    return p.parse_args()


def isotonic_oof_then_apply(oof_raw, y, groups, test_raw):
    cal_oof = np.zeros_like(oof_raw)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups):
        for c in range(N_CLASSES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
            iso.fit(oof_raw[tr_idx, c], (y[tr_idx] == c).astype(float))
            cal_oof[va_idx, c] = iso.predict(oof_raw[va_idx, c])
    cal_oof = cal_oof / np.clip(cal_oof.sum(axis=1, keepdims=True), 1e-12, None)
    cal_test = np.zeros_like(test_raw)
    for c in range(N_CLASSES):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        iso.fit(oof_raw[:, c], (y == c).astype(float))
        cal_test[:, c] = iso.predict(test_raw[:, c])
    cal_test = cal_test / np.clip(cal_test.sum(axis=1, keepdims=True), 1e-12, None)
    return cal_oof, cal_test


def nm_thresholds_multistart(probs, y, n_starts=16, seed=42):
    """6-D Nelder-Mead with multi-start."""
    def neg_f1(lw):
        return -float(f1_score(y, (probs * np.exp(lw)).argmax(axis=1), average="macro"))
    rng = np.random.default_rng(seed)
    candidates = [(np.zeros(N_CLASSES), neg_f1(np.zeros(N_CLASSES)))]
    starts = [np.zeros(N_CLASSES)] + [rng.uniform(-1.5, 1.5, N_CLASSES) for _ in range(n_starts - 1)]
    for x0 in starts:
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 800, "adaptive": True})
        candidates.append((res.x, res.fun))
    candidates.sort(key=lambda t: t[1])
    return candidates[0][0], -candidates[0][1], candidates  # best_x, best_f1, all


def grid_l1_l2(probs, y, log_w_base, grid_n, log_range):
    """Fine grid over (L1 mult, L2 mult) holding other 4 classes at log_w_base."""
    log_grid = np.linspace(-log_range, log_range, grid_n)
    surface = np.zeros((grid_n, grid_n), dtype=np.float32)
    for i, l1 in enumerate(log_grid):
        for j, l2 in enumerate(log_grid):
            lw = log_w_base.copy(); lw[1] = l1; lw[2] = l2
            preds = (probs * np.exp(lw)).argmax(axis=1)
            surface[i, j] = f1_score(y, preds, average="macro")
    return surface, log_grid


def find_grid_extrema(surface, log_grid, robust_window=2):
    """Return (peak_idx, peak_f1, robust_idx, robust_f1)."""
    pi, pj = np.unravel_index(np.argmax(surface), surface.shape)
    peak_f1 = float(surface[pi, pj])

    h = robust_window
    pad = np.pad(surface, h, mode="edge")
    robust_min = np.zeros_like(surface)
    for i in range(surface.shape[0]):
        for j in range(surface.shape[1]):
            robust_min[i, j] = pad[i:i + 2 * h + 1, j:j + 2 * h + 1].min()
    ri, rj = np.unravel_index(np.argmax(robust_min), robust_min.shape)
    robust_f1 = float(surface[ri, rj])
    return (pi, pj), peak_f1, (ri, rj), robust_f1


def dist_chi2(pred_counts, target_dist, total):
    """Chi-squared-like distance between empirical class fractions and target."""
    pred_frac = np.array(pred_counts) / total
    return float(np.sum((pred_frac - target_dist) ** 2 / np.clip(target_dist, 1e-6, None)))


def main():
    args = parse_args()
    print(f"=== rigorous_threshold_search_v6.py ===", flush=True)
    print(f"α sweep: [{args.alpha_min}, {args.alpha_max}] step {args.alpha_step}", flush=True)
    print(f"L1×L2 grid: {args.grid}×{args.grid}  range=±{args.range}", flush=True)
    print(f"NM multi-start: {args.nm_starts} starts in 6-D", flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    test_ids = meta_test["file_id"].values.astype(int)

    # Train base rates (target distribution for the robustness re-ranking)
    train_counts = np.bincount(y, minlength=N_CLASSES)
    train_frac = train_counts / len(y)
    print(f"\nTrain class distribution: "
          f"{[f'L{c}: {train_frac[c]:.3f}' for c in range(N_CLASSES)]}", flush=True)

    print("\nLoading v6 + combo_v2 pieces…", flush=True)
    p1_oof = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_oof.npy").astype(np.float64)
    p1_test = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_test_probs.npy").astype(np.float64)
    p2_oof = np.load(ROOT / "oof" / "hier_v6_pipeline2_oof.npy").astype(np.float64)
    p2_test = np.load(ROOT / "oof" / "hier_v6_pipeline2_test_probs.npy").astype(np.float64)
    p2_label = "v6"

    if args.use_v4:
        p2_oof_v4 = np.load(ROOT / "oof" / "hier_v4_pipeline2_oof.npy").astype(np.float64)
        p2_test_v4 = np.load(ROOT / "oof" / "hier_v4_pipeline2_test_probs.npy").astype(np.float64)
        # We'll run BOTH v4 and v6 by treating them as two separate runs below
        runs = [("v6", p2_oof, p2_test), ("v4", p2_oof_v4, p2_test_v4)]
    else:
        runs = [(p2_label, p2_oof, p2_test)]

    alphas = np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)

    all_candidates = []  # list of dicts

    for run_name, p2o, p2t in runs:
        print(f"\n{'='*70}\nRUN: {run_name}\n{'='*70}", flush=True)
        for alpha in alphas:
            print(f"\n  α={alpha:.3f} …", flush=True, end=" ")
            blend_oof = alpha * p1_oof + (1 - alpha) * p2o
            blend_test = alpha * p1_test + (1 - alpha) * p2t
            blend_oof = blend_oof / np.clip(blend_oof.sum(axis=1, keepdims=True), 1e-12, None)
            blend_test = blend_test / np.clip(blend_test.sum(axis=1, keepdims=True), 1e-12, None)
            cal_oof, cal_test = isotonic_oof_then_apply(blend_oof, y, groups, blend_test)

            # 6-D NM multi-start
            log_w_nm, nm_f1, nm_all = nm_thresholds_multistart(cal_oof, y, n_starts=args.nm_starts)

            # Fine L1×L2 grid using NM-pinned other classes
            surface, log_grid = grid_l1_l2(cal_oof, y, log_w_nm, args.grid, args.range)
            (pi, pj), peak_f1, (ri, rj), robust_f1 = find_grid_extrema(surface, log_grid)

            # Top NM candidates
            for k, (lw, f) in enumerate(nm_all):
                test_preds = (cal_test * np.exp(lw)).argmax(axis=1)
                test_counts = np.bincount(test_preds, minlength=N_CLASSES)
                pred_pc = f1_score(y, (cal_oof * np.exp(lw)).argmax(axis=1), average=None)
                all_candidates.append({
                    "run": run_name, "alpha": float(alpha), "method": f"nm_start_{k}",
                    "oof_f1": float(-f),
                    "log_weights": [float(x) for x in lw],
                    "weights": [float(np.exp(x)) for x in lw],
                    "test_counts": [int(x) for x in test_counts],
                    "test_dist_chi2": dist_chi2(test_counts, train_frac, len(test_preds)),
                    "oof_per_class_f1": [float(x) for x in pred_pc],
                    "cal_test_ref": run_name + "_" + f"{alpha:.3f}",  # later we'll
                                                                         # serialise test probs
                })

            # Grid peak
            log_w_peak = log_w_nm.copy()
            log_w_peak[1] = log_grid[pi]; log_w_peak[2] = log_grid[pj]
            test_preds_p = (cal_test * np.exp(log_w_peak)).argmax(axis=1)
            test_counts_p = np.bincount(test_preds_p, minlength=N_CLASSES)
            pc_p = f1_score(y, (cal_oof * np.exp(log_w_peak)).argmax(axis=1), average=None)
            all_candidates.append({
                "run": run_name, "alpha": float(alpha), "method": "grid_peak",
                "oof_f1": peak_f1,
                "log_weights": [float(x) for x in log_w_peak],
                "weights": [float(np.exp(x)) for x in log_w_peak],
                "test_counts": [int(x) for x in test_counts_p],
                "test_dist_chi2": dist_chi2(test_counts_p, train_frac, len(test_preds_p)),
                "oof_per_class_f1": [float(x) for x in pc_p],
                "cal_test_ref": run_name + "_" + f"{alpha:.3f}",
            })

            # Grid robust
            log_w_rob = log_w_nm.copy()
            log_w_rob[1] = log_grid[ri]; log_w_rob[2] = log_grid[rj]
            test_preds_r = (cal_test * np.exp(log_w_rob)).argmax(axis=1)
            test_counts_r = np.bincount(test_preds_r, minlength=N_CLASSES)
            pc_r = f1_score(y, (cal_oof * np.exp(log_w_rob)).argmax(axis=1), average=None)
            all_candidates.append({
                "run": run_name, "alpha": float(alpha), "method": "grid_robust",
                "oof_f1": robust_f1,
                "log_weights": [float(x) for x in log_w_rob],
                "weights": [float(np.exp(x)) for x in log_w_rob],
                "test_counts": [int(x) for x in test_counts_r],
                "test_dist_chi2": dist_chi2(test_counts_r, train_frac, len(test_preds_r)),
                "oof_per_class_f1": [float(x) for x in pc_r],
                "cal_test_ref": run_name + "_" + f"{alpha:.3f}",
            })

            print(f"NM={nm_f1:.4f}  Peak={peak_f1:.4f}  Rob={robust_f1:.4f}  "
                  f"L2_F1_peak={pc_p[2]:.3f}", flush=True)

    print(f"\n\n{'='*70}\nALL CANDIDATES: {len(all_candidates)}\n{'='*70}", flush=True)
    # Sort by OOF F1
    by_oof = sorted(all_candidates, key=lambda c: -c["oof_f1"])
    print("\nTop-15 by OOF F1:", flush=True)
    print(f"{'rank':>4}  {'run':<3}  {'α':>5}  {'method':<14}  {'OOF':>6}  {'L2F1':>5}  "
          f"{'chi2':>6}  {'L2 cnt':>6}", flush=True)
    for i, c in enumerate(by_oof[:15]):
        print(f"  {i+1:>2}  {c['run']:<3}  {c['alpha']:.3f}  {c['method']:<14}  "
              f"{c['oof_f1']:.4f}  {c['oof_per_class_f1'][2]:.3f}  "
              f"{c['test_dist_chi2']:.4f}  {c['test_counts'][2]:>6}", flush=True)

    # Robustness re-rank: pick candidates within 0.001 of top OOF and rank by test_dist_chi2
    top_oof = by_oof[0]["oof_f1"]
    near_top = [c for c in all_candidates if c["oof_f1"] >= top_oof - 0.001]
    print(f"\nCandidates within 0.001 of top OOF ({top_oof:.4f}): {len(near_top)}", flush=True)
    by_chi2 = sorted(near_top, key=lambda c: c["test_dist_chi2"])
    print("\nTop-10 by 'closeness to train base rates' (robustness):", flush=True)
    for i, c in enumerate(by_chi2[:10]):
        print(f"  {i+1:>2}  {c['run']:<3}  {c['alpha']:.3f}  {c['method']:<14}  "
              f"OOF={c['oof_f1']:.4f}  L2F1={c['oof_per_class_f1'][2]:.3f}  "
              f"χ²={c['test_dist_chi2']:.4f}  counts={c['test_counts']}", flush=True)

    # Submissions: top-k by OOF, plus top by robustness if distinct
    out_dir = ROOT / "submissions"
    written = []

    def write_sub(cand: dict, suffix: str):
        # Recompute cal_test for THIS candidate's alpha + run
        run = cand["run"]
        alpha = cand["alpha"]
        if run == "v6":
            p2o = p2_oof; p2t = p2_test
        else:
            p2o = np.load(ROOT / "oof" / "hier_v4_pipeline2_oof.npy").astype(np.float64)
            p2t = np.load(ROOT / "oof" / "hier_v4_pipeline2_test_probs.npy").astype(np.float64)
        blend_oof = alpha * p1_oof + (1 - alpha) * p2o
        blend_test = alpha * p1_test + (1 - alpha) * p2t
        blend_oof = blend_oof / np.clip(blend_oof.sum(axis=1, keepdims=True), 1e-12, None)
        blend_test = blend_test / np.clip(blend_test.sum(axis=1, keepdims=True), 1e-12, None)
        _, cal_test = isotonic_oof_then_apply(blend_oof, y, groups, blend_test)
        w = np.exp(np.array(cand["log_weights"]))
        preds = (cal_test * w).argmax(axis=1)
        path = out_dir / f"sub_hier_{run}_rigorous_{suffix}.csv"
        pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(path, index=False)
        return path

    print(f"\nWriting top-{args.top_k} by OOF…", flush=True)
    for i in range(min(args.top_k, len(by_oof))):
        c = by_oof[i]
        suffix = f"top{i+1}_a{int(c['alpha']*1000):03d}_{c['method']}"
        p = write_sub(c, suffix)
        written.append(("top_oof", i + 1, c, p))
        print(f"  [top OOF #{i+1}] {p.name}  "
              f"OOF={c['oof_f1']:.4f}  L2={c['oof_per_class_f1'][2]:.3f}  "
              f"χ²={c['test_dist_chi2']:.4f}", flush=True)

    # Also write top-3 by robustness if they aren't already in top-OOF
    already = set((c["run"], c["alpha"], c["method"]) for _, _, c, _ in written)
    print(f"\nWriting top-3 robustness picks (not already top-OOF)…", flush=True)
    written_rob = 0
    for c in by_chi2:
        key = (c["run"], c["alpha"], c["method"])
        if key in already:
            continue
        suffix = f"robust{written_rob+1}_a{int(c['alpha']*1000):03d}_{c['method']}"
        p = write_sub(c, suffix)
        written.append(("robust", written_rob + 1, c, p))
        print(f"  [robust #{written_rob+1}] {p.name}  "
              f"OOF={c['oof_f1']:.4f}  L2={c['oof_per_class_f1'][2]:.3f}  "
              f"χ²={c['test_dist_chi2']:.4f}", flush=True)
        already.add(key)
        written_rob += 1
        if written_rob >= 3:
            break

    # Save full candidate list
    meta_path = ROOT / "oof" / "rigorous_threshold_v6_meta.json"
    meta_path.write_text(json.dumps({
        "config": {"alpha_range": [args.alpha_min, args.alpha_max, args.alpha_step],
                   "grid": args.grid, "log_range": args.range,
                   "nm_starts": args.nm_starts},
        "n_candidates": len(all_candidates),
        "top_oof": [
            {**c, "rank": i + 1}
            for i, c in enumerate(by_oof[:20])
        ],
        "top_robustness": [
            {**c, "rank": i + 1}
            for i, c in enumerate(by_chi2[:10])
        ],
    }, indent=2))
    print(f"\nWrote {meta_path}", flush=True)
    print(f"\n=== rigorous_threshold_search_v6.py done ===", flush=True)


if __name__ == "__main__":
    main()
