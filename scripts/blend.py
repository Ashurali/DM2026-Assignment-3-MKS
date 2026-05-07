"""Phase-8 blend — combine OOF probabilities from multiple base models.

Two methods, both evaluated on the same 5-fold GroupKFold and the better one
(by OOF macro-F1) is picked:

  1. Scipy-simplex weights — N free params (one per base model), constrained
     to sum to 1 and be non-negative. Minimises OOF cross-entropy of the
     weighted-average probability vector. Gradient-free SLSQP.

  2. LR meta-learner — Logistic Regression trained on stacked OOF probs
     (shape (N, K_models * 6)) with class_weight='balanced' and the same
     GroupKFold(5) used by every base model. Output OOF predictions are the
     blend; full-data refit predicts test.

After the blend, post-hoc per-class threshold tuning runs on the blend OOF —
same approach as scripts/post_hoc_threshold.py (Nelder-Mead on log-multipliers,
multi-start, optimising OOF macro-F1).

Usage:
    python scripts/blend.py [--inputs M1 M2 ...] [--name v1] [--no-threshold-tune]

Defaults: blend `lgbm_full_tuned_v1` + `cnn_bilstm_v1`, with threshold tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


# ─── Loading ─────────────────────────────────────────────────────────────────
def load_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, train_groups, test_file_ids)."""
    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    return (
        meta_train["label"].values.astype(np.int64),
        meta_train["user_id"].values,
        meta_test["file_id"].values.astype(np.int64),
    )


def load_inputs(names: list[str], use_tta: bool = False) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load OOF and test prob arrays for the given run names.

    If `use_tta=True`, prefer `<run>_tta_test_probs.npy` over the plain test
    probs whenever available. OOF probs are unchanged regardless — TTA only
    affects test inference, not the OOF (which comes from per-fold models).
    """
    oofs, tests, found = [], [], []
    for n in names:
        oof_p = ROOT / "oof" / f"{n}_oof.npy"
        test_p = ROOT / "oof" / f"{n}_test_probs.npy"
        tta_test_p = ROOT / "oof" / f"{n}_tta_test_probs.npy"
        if not oof_p.exists():
            print(f"!!! Skipping '{n}': missing {oof_p.name}")
            continue
        if use_tta and tta_test_p.exists():
            chosen_test = tta_test_p
            tag = "TTA"
        elif test_p.exists():
            chosen_test = test_p
            tag = "plain"
        else:
            print(f"!!! Skipping '{n}': missing both regular and TTA test probs")
            continue
        oofs.append(np.load(oof_p))
        tests.append(np.load(chosen_test))
        found.append(n)
        print(f"  Loaded '{n}'  (test source: {tag} - {chosen_test.name})")
    if not found:
        raise SystemExit("No usable input runs found.")
    return oofs, tests, found


# ─── Method 1: scipy simplex ─────────────────────────────────────────────────
def fit_simplex_weights(oofs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 that minimise OOF cross-entropy."""
    K = len(oofs)

    def neg_ce(w: np.ndarray) -> float:
        blend = np.zeros_like(oofs[0])
        for wi, P in zip(w, oofs):
            blend += wi * P
        # Clip for numerical stability before log
        blend = np.clip(blend, 1e-12, 1.0)
        # Mean negative log-likelihood
        return -float(np.mean(np.log(blend[np.arange(len(y)), y])))

    x0 = np.full(K, 1.0 / K)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * K
    res = minimize(neg_ce, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def apply_weights(probs_list: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs_list[0])
    for wi, P in zip(w, probs_list):
        out += wi * P
    return out


# ─── Method 2: LR meta-learner ───────────────────────────────────────────────
def fit_lr_meta(oofs: list[np.ndarray], y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, LogisticRegression]:
    """5-fold GroupKFold LR over stacked OOF probs.

    Returns (oof_preds_probs, full_data_lr_for_test_inference).
    """
    from sklearn.model_selection import GroupKFold

    X = np.concatenate(oofs, axis=1)  # (N, K * 6)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups))
    oof_probs = np.zeros((len(y), N_CLASSES), dtype=np.float64)
    for tr, va in folds:
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        )
        clf.fit(X[tr], y[tr])
        oof_probs[va] = clf.predict_proba(X[va])
    # Refit on full data for test inference
    full_clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42,
    ).fit(X, y)
    return oof_probs, full_clf


# ─── Post-hoc threshold tuning (same as scripts/post_hoc_threshold.py) ───────
def tune_thresholds(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multi-start Nelder-Mead on per-class log-multipliers."""
    def neg_f1(log_w: np.ndarray) -> float:
        w = np.exp(log_w)
        preds = (probs * w).argmax(axis=1)
        return -float(f1_score(y, preds, average="macro"))

    best = {"x": np.zeros(N_CLASSES), "fun": neg_f1(np.zeros(N_CLASSES))}
    rng = np.random.default_rng(42)
    starts = [np.zeros(N_CLASSES)] + [rng.uniform(-1.0, 1.0, N_CLASSES) for _ in range(8)]
    for x0 in starts:
        res = minimize(
            neg_f1, x0, method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True},
        )
        if res.fun < best["fun"]:
            best = {"x": res.x, "fun": res.fun}
    return best["x"]


def apply_thresholds(probs: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    return (probs * np.exp(log_w)).argmax(axis=1)


# ─── Driver ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["lgbm_full_tuned_v1", "cnn_bilstm_v1"])
    p.add_argument("--name", default="v1")
    p.add_argument("--no-threshold-tune", dest="threshold_tune", action="store_false")
    p.add_argument("--tta", action="store_true",
                   help="Prefer <run>_tta_test_probs.npy when available (OOF unchanged).")
    p.set_defaults(threshold_tune=True)
    return p.parse_args()


def report_per_class(name: str, y: np.ndarray, preds: np.ndarray) -> tuple[float, np.ndarray]:
    macro = float(f1_score(y, preds, average="macro"))
    per_class = f1_score(y, preds, average=None)
    print(f"  {name:25s}  macro={macro:.4f}  "
          f"per-class=" + ", ".join(f"{f:.3f}" for f in per_class))
    return macro, per_class


def main() -> None:
    args = parse_args()
    y, groups, test_ids = load_truth()
    oofs, tests, names = load_inputs(args.inputs, use_tta=args.tta)

    print(f"Blending {len(names)} model(s):")
    for n, P in zip(names, oofs):
        macro_n = float(f1_score(y, P.argmax(axis=1), average="macro"))
        print(f"  {n:30s}  shape={P.shape}  baseline OOF macro={macro_n:.4f}")

    # ── Method 1: scipy-simplex ──
    print("\n=== Method 1: scipy-simplex on OOF cross-entropy ===")
    w_simplex = fit_simplex_weights(oofs, y)
    print(f"Simplex weights: {dict(zip(names, [round(float(x), 4) for x in w_simplex]))}")
    blend_oof_simplex = apply_weights(oofs, w_simplex)
    blend_test_simplex = apply_weights(tests, w_simplex)
    f1_simplex, pc_simplex = report_per_class("simplex blend OOF", y, blend_oof_simplex.argmax(axis=1))

    # ── Method 2: LR meta ──
    print("\n=== Method 2: LR meta-learner on stacked OOF probs ===")
    blend_oof_lr, full_lr = fit_lr_meta(oofs, y, groups)
    blend_test_lr = full_lr.predict_proba(np.concatenate(tests, axis=1))
    f1_lr, pc_lr = report_per_class("LR-meta blend OOF", y, blend_oof_lr.argmax(axis=1))

    # ── Pick winner ──
    if f1_simplex >= f1_lr:
        winner = "simplex"
        blend_oof, blend_test = blend_oof_simplex, blend_test_simplex
        blend_meta = {"method": "simplex", "weights": dict(zip(names, [float(x) for x in w_simplex]))}
        f1_blend = f1_simplex
        pc_blend = pc_simplex
    else:
        winner = "lr_meta"
        blend_oof, blend_test = blend_oof_lr, blend_test_lr
        blend_meta = {"method": "lr_meta", "stacked_input_dim": np.concatenate(oofs, axis=1).shape[1]}
        f1_blend = f1_lr
        pc_blend = pc_lr
    print(f"\n>>> Winner: {winner}  (OOF macro={f1_blend:.4f})")

    # ── Optional threshold tuning on blend ──
    final_oof_preds = blend_oof.argmax(axis=1)
    final_test_preds = blend_test.argmax(axis=1)
    f1_final = f1_blend
    pc_final = pc_blend
    log_w = np.zeros(N_CLASSES)
    if args.threshold_tune:
        print("\n=== Post-hoc threshold tuning on blend OOF ===")
        log_w = tune_thresholds(blend_oof, y)
        final_oof_preds = apply_thresholds(blend_oof, log_w)
        final_test_preds = apply_thresholds(blend_test, log_w)
        f1_final, pc_final = report_per_class("blend + thresh OOF", y, final_oof_preds)
        print(f"Threshold multipliers: {[round(float(x), 3) for x in np.exp(log_w)]}")

    print("\n=== Final classification report (blend + thresh, OOF) ===")
    print(classification_report(y, final_oof_preds, digits=4))

    # ── Outputs ──
    suffix = "_tuned" if args.threshold_tune else ""
    sub_path = ROOT / "submissions" / f"sub_blend_{args.name}{suffix}.csv"
    pd.DataFrame({"Id": test_ids.astype(int), "Label": final_test_preds.astype(int)}).to_csv(sub_path, index=False)
    print(f"\nWrote {sub_path}")

    np.save(ROOT / "oof" / f"blend_{args.name}_oof.npy", blend_oof.astype(np.float32))
    np.save(ROOT / "oof" / f"blend_{args.name}_test_probs.npy", blend_test.astype(np.float32))

    sidecar = {
        "name": f"blend_{args.name}",
        "inputs": names,
        "input_baseline_oof_macro": {
            n: float(f1_score(y, P.argmax(axis=1), average="macro"))
            for n, P in zip(names, oofs)
        },
        "blend_meta": blend_meta,
        "blend_oof_macro": float(f1_blend),
        "simplex_oof_macro": float(f1_simplex),
        "lr_meta_oof_macro": float(f1_lr),
        "thresholds_applied": args.threshold_tune,
        "threshold_log_weights": [float(x) for x in log_w],
        "threshold_multipliers": [float(x) for x in np.exp(log_w)],
        "final_oof_macro": float(f1_final),
        "final_per_class_f1": [float(x) for x in pc_final],
        "test_pred_dist": {int(c): int((final_test_preds == c).sum()) for c in range(N_CLASSES)},
    }
    sidecar_path = ROOT / "oof" / f"blend_{args.name}_meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Wrote {sidecar_path}")


if __name__ == "__main__":
    main()
