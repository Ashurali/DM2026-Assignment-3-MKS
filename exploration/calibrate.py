"""Post-hoc probability calibration on a saved OOF + test prob set.

Applies per-class isotonic regression: for each class c, fit a 1D
isotonic mapping from predicted P(class=c) → true P(class=c | predicted)
using the OOF predictions. Then re-predict via argmax of the calibrated
probabilities.

This stacks on top of threshold tuning (which is decision-level): isotonic
calibration corrects the *score* shape, threshold tuning corrects the
*argmax* boundary. Combined, they often give +0.002 to +0.008 OOF.

Usage:
    python scripts/calibrate.py --run lgbm_combo_combo_full
    python scripts/calibrate.py --run blend_top4
    # also re-applies threshold tuning after calibration:
    python scripts/calibrate.py --run blend_top4 --then-threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


def load_run(name: str):
    """Load <name>_oof.npy and <name>_test_probs.npy (auto-tries common prefixes)."""
    for prefix in ("", "lgbm_full_", "lgbm_combo_", "blend_"):
        oof_p = ROOT / "oof" / f"{prefix}{name}_oof.npy"
        test_p = ROOT / "oof" / f"{prefix}{name}_test_probs.npy"
        if oof_p.exists() and test_p.exists():
            return np.load(oof_p), np.load(test_p), f"{prefix}{name}"
    raise SystemExit(f"Could not locate OOF for run '{name}'")


def calibrate_isotonic(oof_probs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[IsotonicRegression]]:
    """Per-class isotonic calibration via 5-fold GKF (avoids overfitting on the
    same data we then evaluate)."""
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups))
    out_probs = np.zeros_like(oof_probs)

    # For each fold, fit isotonic on (train OOF, train labels) and apply on val
    for tr, va in folds:
        for c in range(N_CLASSES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
            iso.fit(oof_probs[tr, c], (y[tr] == c).astype(float))
            out_probs[va, c] = iso.predict(oof_probs[va, c])

    # Renormalise so each row sums to 1
    out_probs = out_probs / np.clip(out_probs.sum(axis=1, keepdims=True), 1e-12, None)

    # Refit on full data for test inference
    full_isos: list[IsotonicRegression] = []
    for c in range(N_CLASSES):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        iso.fit(oof_probs[:, c], (y == c).astype(float))
        full_isos.append(iso)

    return out_probs, full_isos


def apply_isotonic_to_test(test_probs: np.ndarray, isos: list[IsotonicRegression]) -> np.ndarray:
    out = np.zeros_like(test_probs)
    for c in range(N_CLASSES):
        out[:, c] = isos[c].predict(test_probs[:, c])
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def tune_thresholds(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multi-start Nelder-Mead on log-multipliers (same as post_hoc_threshold.py)."""
    def neg_f1(log_w: np.ndarray) -> float:
        return -float(f1_score(y, (probs * np.exp(log_w)).argmax(axis=1), average="macro"))

    best = {"x": np.zeros(N_CLASSES), "fun": neg_f1(np.zeros(N_CLASSES))}
    rng = np.random.default_rng(42)
    for x0 in [np.zeros(N_CLASSES)] + [rng.uniform(-1.0, 1.0, N_CLASSES) for _ in range(8)]:
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True})
        if res.fun < best["fun"]:
            best = {"x": res.x, "fun": res.fun}
    return best["x"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run name (auto-tries common prefixes).")
    p.add_argument("--then-threshold", action="store_true",
                   help="Re-tune class thresholds after calibration.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    test_ids = meta_test["file_id"].values.astype(int)

    oof, test_probs, full_name = load_run(args.run)
    print(f"Loaded '{full_name}'  oof={oof.shape}  test={test_probs.shape}")

    base_preds = oof.argmax(axis=1)
    base_f1 = float(f1_score(y, base_preds, average="macro"))
    base_pc = f1_score(y, base_preds, average=None)
    print(f"\nBaseline OOF macro: {base_f1:.4f}")
    print(f"  per-class: {[round(float(x), 4) for x in base_pc]}")

    # ── Isotonic calibration ──
    print("\n=== Per-class isotonic calibration (5-fold GKF) ===")
    cal_oof, full_isos = calibrate_isotonic(oof, y, groups)
    cal_test = apply_isotonic_to_test(test_probs, full_isos)
    cal_preds = cal_oof.argmax(axis=1)
    cal_f1 = float(f1_score(y, cal_preds, average="macro"))
    cal_pc = f1_score(y, cal_preds, average=None)
    print(f"Calibrated OOF macro: {cal_f1:.4f}  (Δ {cal_f1 - base_f1:+.4f})")
    print(f"  per-class: {[round(float(x), 4) for x in cal_pc]}")

    final_preds_oof = cal_preds
    final_preds_test = cal_test.argmax(axis=1)
    final_oof_f1 = cal_f1
    final_pc_f1 = cal_pc
    suffix = "_cal"
    log_w = np.zeros(N_CLASSES)

    # ── Optional threshold tuning on top of calibrated probs ──
    if args.then_threshold:
        print("\n=== Threshold tuning on calibrated OOF ===")
        log_w = tune_thresholds(cal_oof, y)
        w = np.exp(log_w)
        thresh_preds = (cal_oof * w).argmax(axis=1)
        final_preds_oof = thresh_preds
        final_preds_test = (cal_test * w).argmax(axis=1)
        final_oof_f1 = float(f1_score(y, thresh_preds, average="macro"))
        final_pc_f1 = f1_score(y, thresh_preds, average=None)
        print(f"Cal+thresh OOF macro: {final_oof_f1:.4f}  (Δ {final_oof_f1 - base_f1:+.4f} total)")
        print(f"  per-class: {[round(float(x), 4) for x in final_pc_f1]}")
        print(f"  weights: {[round(float(x), 3) for x in w]}")
        suffix = "_cal_thresh"

    print("\n=== Final classification report (OOF) ===")
    print(classification_report(y, final_preds_oof, digits=4))

    # ── Outputs ──
    sub_path = ROOT / "submissions" / f"sub_{full_name}{suffix}.csv"
    pd.DataFrame({"Id": test_ids, "Label": final_preds_test.astype(int)}).to_csv(sub_path, index=False)
    print(f"\nWrote {sub_path}")

    np.save(ROOT / "oof" / f"{full_name}_calibrated_oof.npy", cal_oof.astype(np.float32))
    np.save(ROOT / "oof" / f"{full_name}_calibrated_test_probs.npy", cal_test.astype(np.float32))

    sidecar = {
        "source_run": full_name,
        "baseline_oof_macro": base_f1,
        "calibrated_oof_macro": cal_f1,
        "final_oof_macro": final_oof_f1,
        "baseline_per_class_f1": [float(x) for x in base_pc],
        "calibrated_per_class_f1": [float(x) for x in cal_pc],
        "final_per_class_f1": [float(x) for x in final_pc_f1],
        "thresholds_applied": args.then_threshold,
        "threshold_log_weights": [float(x) for x in log_w],
        "test_pred_dist": {int(c): int((final_preds_test == c).sum()) for c in range(N_CLASSES)},
    }
    sidecar_path = ROOT / "oof" / f"{full_name}{suffix}_meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Wrote {sidecar_path}")


if __name__ == "__main__":
    main()
