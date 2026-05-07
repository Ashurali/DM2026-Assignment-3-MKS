"""Post-hoc per-class threshold (multiplier) tuning to maximise OOF F1-macro.

Why: the LGBM is shy on minority classes because the softmax favours the
majority class even when its probability is only marginally higher. Scaling
each class's probability column by a tuned weight before argmax recovers
recall on minorities at modest precision cost.

Optimization: Nelder-Mead (gradient-free, robust to non-smooth F1-macro) on
the log of class multipliers. 6-dim search; runs in < 30 s on CPU.

Usage:
    python scripts/post_hoc_threshold.py [--name v1] [--name smote_v1] ...

For each named run, reads:
- oof/lgbm_full_<name>_oof.npy        (N_train, 6) OOF probs
- oof/lgbm_full_<name>_test_probs.npy (N_test, 6)  test probs
- oof/lgbm_full_<name>_meta.json      (for original per-class F1 reference)

Writes:
- oof/lgbm_full_<name>_oof_tuned.npy            (re-thresholded OOF preds)
- oof/lgbm_full_<name>_tuned_weights.json       (the 6 multipliers + metrics)
- submissions/sub_lgbm_full_<name>_tuned.csv    (re-thresholded test preds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


def load_truth() -> tuple[np.ndarray, np.ndarray]:
    meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    return meta["label"].values.astype(np.int64), meta["file_id"].values


def predict_with_weights(probs: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """Apply per-class multipliers (in log space) and return argmax preds."""
    w = np.exp(log_w)
    return (probs * w).argmax(axis=1)


def neg_f1_macro(log_w: np.ndarray, probs: np.ndarray, y: np.ndarray) -> float:
    preds = predict_with_weights(probs, log_w)
    return -float(f1_score(y, preds, average="macro"))


def tune_one(name: str, probs: np.ndarray, y: np.ndarray) -> dict:
    """Multi-start Nelder-Mead on the log-multipliers."""
    base_preds = probs.argmax(axis=1)
    base_f1 = float(f1_score(y, base_preds, average="macro"))
    base_per_class = f1_score(y, base_preds, average=None)

    print(f"\n=== {name} ===")
    print(f"Baseline OOF macro-F1: {base_f1:.4f}")
    print(f"  per-class: {[round(float(x), 4) for x in base_per_class]}")

    best = {"x": np.zeros(N_CLASSES), "fun": -base_f1}
    rng = np.random.default_rng(42)

    # Multi-start: uniform + several random log-uniform starts in [-1.5, 1.5]
    starts = [np.zeros(N_CLASSES)]
    for _ in range(8):
        starts.append(rng.uniform(-1.0, 1.0, size=N_CLASSES))

    for i, x0 in enumerate(starts):
        res = minimize(
            neg_f1_macro,
            x0,
            args=(probs, y),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True},
        )
        if res.fun < best["fun"]:
            best = {"x": res.x, "fun": res.fun}
            print(f"  [start {i}] new best: macro-F1 = {-res.fun:.4f}")

    log_w = best["x"]
    w = np.exp(log_w)
    tuned_preds = predict_with_weights(probs, log_w)
    tuned_f1 = float(f1_score(y, tuned_preds, average="macro"))
    tuned_per_class = f1_score(y, tuned_preds, average=None)

    print(f"\nTuned OOF macro-F1: {tuned_f1:.4f}  (Δ {tuned_f1 - base_f1:+.4f})")
    print(f"  per-class: {[round(float(x), 4) for x in tuned_per_class]}")
    print(f"  Δ per-class: {[round(float(t - b), 4) for t, b in zip(tuned_per_class, base_per_class)]}")
    print(f"  weights (multipliers): {[round(float(x), 3) for x in w]}")
    print(f"\nFull classification report (tuned, OOF):")
    print(classification_report(y, tuned_preds, digits=4))

    return {
        "name": name,
        "baseline_f1_macro": base_f1,
        "tuned_f1_macro": tuned_f1,
        "delta": tuned_f1 - base_f1,
        "log_weights": [float(x) for x in log_w],
        "weights": [float(x) for x in w],
        "baseline_per_class_f1": [float(x) for x in base_per_class],
        "tuned_per_class_f1": [float(x) for x in tuned_per_class],
        "tuned_preds": tuned_preds,
    }


def write_outputs(name: str, result: dict, test_probs: np.ndarray, file_ids: np.ndarray) -> None:
    log_w = np.array(result["log_weights"])
    test_preds = predict_with_weights(test_probs, log_w)

    sub_path = ROOT / "submissions" / f"sub_lgbm_full_{name}_tuned.csv"
    pd.DataFrame({"Id": file_ids.astype(int), "Label": test_preds.astype(int)}).to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}")

    meta_path = ROOT / "oof" / f"lgbm_full_{name}_tuned_weights.json"
    out = {k: v for k, v in result.items() if k != "tuned_preds"}
    out["tuned_test_pred_dist"] = {int(c): int((test_preds == c).sum()) for c in range(N_CLASSES)}
    meta_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", default=[], help="One or more run names to tune (e.g. v1, smote_v1).")
    args = parser.parse_args()
    names = args.name or ["v1", "smote_v1"]

    y, train_file_ids = load_truth()
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    test_file_ids = meta_test["file_id"].values

    results = []
    for name in names:
        oof_path = ROOT / "oof" / f"lgbm_full_{name}_oof.npy"
        if not oof_path.exists():
            print(f"Skipping {name}: {oof_path} not found.")
            continue
        oof_probs = np.load(oof_path)
        test_probs = np.load(ROOT / "oof" / f"lgbm_full_{name}_test_probs.npy")
        res = tune_one(name, oof_probs, y)
        write_outputs(name, res, test_probs, test_file_ids)
        results.append(res)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Summary across runs (tuned OOF macro-F1):")
        for r in sorted(results, key=lambda r: -r["tuned_f1_macro"]):
            print(f"  {r['name']:20s}  {r['baseline_f1_macro']:.4f} -> {r['tuned_f1_macro']:.4f}  (Δ {r['delta']:+.4f})")
        best = max(results, key=lambda r: r["tuned_f1_macro"])
        print(f"\nBest tuned: {best['name']}  → submissions/sub_lgbm_full_{best['name']}_tuned.csv")


if __name__ == "__main__":
    main()
