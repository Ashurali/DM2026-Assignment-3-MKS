"""2-D grid over (L1, L2) class multipliers, applied to v6 (EO-selected) blend
at the auto-tuned α=0.842 (which gave OOF 0.7880 — highest in the suite).

Same grid logic as threshold_grid_l1l2.py but pointed at v6 outputs and
α=0.842 rather than v4 α=0.88.
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
    p.add_argument("--alpha", type=float, default=0.842,
                   help="Blend weight on combo_v2 (P1). v6 auto-tuned to 0.842.")
    p.add_argument("--grid", type=int, default=31)
    p.add_argument("--range", type=float, default=1.5)
    p.add_argument("--robust-window", type=int, default=2)
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


def tune_thresholds_nelder(probs, y) -> np.ndarray:
    def neg_f1(lw):
        return -float(f1_score(y, (probs * np.exp(lw)).argmax(axis=1), average="macro"))
    best_x, best_v = np.zeros(N_CLASSES), neg_f1(np.zeros(N_CLASSES))
    rng = np.random.default_rng(42)
    for x0 in [np.zeros(N_CLASSES)] + [rng.uniform(-1.0, 1.0, N_CLASSES) for _ in range(8)]:
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x


def main():
    args = parse_args()
    print(f"=== threshold_grid_v6.py  α={args.alpha}  grid={args.grid}  range=±{args.range} ===",
          flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    test_ids = meta_test["file_id"].values.astype(int)

    print("Loading v6 + combo_v2 pieces…", flush=True)
    p1_oof = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_oof.npy").astype(np.float64)
    p1_test = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_test_probs.npy").astype(np.float64)
    p2_oof = np.load(ROOT / "oof" / "hier_v6_pipeline2_oof.npy").astype(np.float64)
    p2_test = np.load(ROOT / "oof" / "hier_v6_pipeline2_test_probs.npy").astype(np.float64)

    blend_oof = args.alpha * p1_oof + (1 - args.alpha) * p2_oof
    blend_test = args.alpha * p1_test + (1 - args.alpha) * p2_test
    blend_oof = blend_oof / np.clip(blend_oof.sum(axis=1, keepdims=True), 1e-12, None)
    blend_test = blend_test / np.clip(blend_test.sum(axis=1, keepdims=True), 1e-12, None)

    print("Fitting per-class isotonic…", flush=True)
    cal_oof, cal_test = isotonic_oof_then_apply(blend_oof, y, groups, blend_test)

    # Step 1: NM reference
    print("\n--- Step 1: Nelder-Mead 6-D thresholds (v6 reference) ---", flush=True)
    log_w_nm = tune_thresholds_nelder(cal_oof, y)
    w_nm = np.exp(log_w_nm)
    pred_nm = (cal_oof * w_nm).argmax(axis=1)
    f1_nm = f1_score(y, pred_nm, average="macro")
    pc_nm = f1_score(y, pred_nm, average=None)
    print(f"  v6 NM weights: {[round(float(x), 3) for x in w_nm]}", flush=True)
    print(f"  v6 NM OOF F1: {f1_nm:.4f}", flush=True)
    print(f"  v6 NM per-class: {[round(float(x), 4) for x in pc_nm]}", flush=True)

    # Step 2: 2-D grid over (L1, L2)
    print(f"\n--- Step 2: ({args.grid}×{args.grid}) grid over L1×L2 multipliers ---", flush=True)
    log_grid = np.linspace(-args.range, args.range, args.grid)
    surface = np.zeros((args.grid, args.grid), dtype=np.float32)
    for i, l1 in enumerate(log_grid):
        for j, l2 in enumerate(log_grid):
            lw = log_w_nm.copy(); lw[1] = l1; lw[2] = l2
            preds = (cal_oof * np.exp(lw)).argmax(axis=1)
            surface[i, j] = f1_score(y, preds, average="macro")

    pi, pj = np.unravel_index(np.argmax(surface), surface.shape)
    peak_f1 = float(surface[pi, pj])
    peak_l1, peak_l2 = float(log_grid[pi]), float(log_grid[pj])
    print(f"  Peak F1: {peak_f1:.4f} at log_w_L1={peak_l1:.3f} (×{np.exp(peak_l1):.3f})  "
          f"log_w_L2={peak_l2:.3f} (×{np.exp(peak_l2):.3f})", flush=True)

    h = args.robust_window
    pad = np.pad(surface, h, mode="edge")
    robust = np.zeros_like(surface)
    for i in range(args.grid):
        for j in range(args.grid):
            window = pad[i:i + 2 * h + 1, j:j + 2 * h + 1]
            robust[i, j] = window.min()
    ri, rj = np.unravel_index(np.argmax(robust), robust.shape)
    robust_f1 = float(surface[ri, rj])
    robust_l1, robust_l2 = float(log_grid[ri]), float(log_grid[rj])
    print(f"  Robust plateau: F1={robust_f1:.4f}  "
          f"at log_w_L1={robust_l1:.3f} (×{np.exp(robust_l1):.3f})  "
          f"log_w_L2={robust_l2:.3f} (×{np.exp(robust_l2):.3f})", flush=True)

    print(f"\n  v6 NM   : F1={f1_nm:.4f}", flush=True)
    print(f"  Peak    : F1={peak_f1:.4f}  Δ={peak_f1 - f1_nm:+.4f}", flush=True)
    print(f"  Robust  : F1={robust_f1:.4f}  Δ={robust_f1 - f1_nm:+.4f}", flush=True)

    # Write submissions
    out_dir = ROOT / "submissions"

    def write_sub(suffix: str, log_w: np.ndarray, oof_f1: float, pc: np.ndarray):
        w = np.exp(log_w)
        preds_test = (cal_test * w).argmax(axis=1)
        sub_path = out_dir / f"sub_hier_v6_a{int(args.alpha*1000):03d}_grid_{suffix}.csv"
        pd.DataFrame({"Id": test_ids, "Label": preds_test.astype(int)}).to_csv(sub_path, index=False)
        print(f"  Wrote {sub_path}", flush=True)
        return {
            "log_weights": [float(x) for x in log_w],
            "weights": [float(x) for x in w],
            "oof_f1_macro": float(oof_f1),
            "oof_per_class_f1": [float(x) for x in pc],
            "test_pred_dist": {int(c): int((preds_test == c).sum()) for c in range(N_CLASSES)},
        }

    out_meta = {"alpha": args.alpha, "grid": args.grid, "range": args.range,
                "v6_nm": write_sub("nm_repro", log_w_nm, f1_nm, pc_nm)}

    log_w_peak = log_w_nm.copy(); log_w_peak[1] = peak_l1; log_w_peak[2] = peak_l2
    pred_peak = (cal_oof * np.exp(log_w_peak)).argmax(axis=1)
    pc_peak = f1_score(y, pred_peak, average=None)
    out_meta["peak"] = write_sub("peak", log_w_peak, peak_f1, pc_peak)

    log_w_rob = log_w_nm.copy(); log_w_rob[1] = robust_l1; log_w_rob[2] = robust_l2
    pred_rob = (cal_oof * np.exp(log_w_rob)).argmax(axis=1)
    pc_rob = f1_score(y, pred_rob, average=None)
    out_meta["robust"] = write_sub("robust", log_w_rob, robust_f1, pc_rob)

    meta_path = ROOT / "oof" / "threshold_grid_v6_meta.json"
    meta_path.write_text(json.dumps(out_meta, indent=2))
    print(f"\nWrote {meta_path}", flush=True)

    print("\n=== Peak (OOF) classification report ===", flush=True)
    print(classification_report(y, pred_peak, digits=4))
    print("\n=== Robust (OOF) classification report ===", flush=True)
    print(classification_report(y, pred_rob, digits=4))

    print("\n=== threshold_grid_v6.py done ===", flush=True)


if __name__ == "__main__":
    main()
