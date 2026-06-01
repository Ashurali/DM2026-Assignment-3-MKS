"""Confidence-gated L1->L2 rescue — capture DG's complementary L2 catches.

The blunt calibrated injection barely moved L2 because DG's L2 *precision* is low
(0.106): adding DG mass everywhere creates false L2. But DG correctly catches 84
of production's 225 L2 misses. If DG's *confidence* separates its correct L2 calls
from its wrong ones, a targeted rescue captures them without the precision cost:

  base = production prediction under FROZEN PEAK_LOGW thresholds.
  rescue: where base == 1 (L1) AND dg_argmax == 2 AND dg[:,2] > tau  ->  flip to L2.

The single parameter tau is chosen by NESTED GroupKFold (tuned on 4 user-folds,
evaluated on the held-out fold) so any gain is an honest cross-user transfer, not
threshold-trap overfitting. Production thresholds are NEVER re-tuned.

Run:  python scripts/dg_gated_rescue.py [--name v1] [--from-any]
"""
from __future__ import annotations
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
N = 6
ALPHA = 0.842
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v1")
ap.add_argument("--from-any", action="store_true", help="rescue from any class (not just L1)")
args = ap.parse_args()

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
dg, dgt = load(f"dg_cisc_{args.name}_oof.npy"), load(f"dg_cisc_{args.name}_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in GroupKFold(5).split(oof_raw, groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


cal_prod, cal_prod_t = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
dgc, dgc_t = norm(dg), norm(dgt)
base_pred = (cal_prod * np.exp(PEAK_LOGW)).argmax(1)
base = f1_score(y, base_pred, average="macro")
print(f"production baseline OOF macro-F1 @ frozen thresholds = {base:.4f}\n")


def rescue(bpred, dgp, tau, from_any):
    pred = bpred.copy()
    src = np.ones(len(pred), dtype=bool) if from_any else (bpred == 1)
    flip = src & (dgp.argmax(1) == 2) & (dgp[:, 2] > tau)
    pred[flip] = 2
    return pred, int(flip.sum())


# full-OOF sweep (diagnostic)
TAUS = np.round(np.linspace(0.20, 0.95, 16), 3)
print(f"{'tau':>6} {'macroF1':>9} {'n_flip':>7} {'L2_f1':>7} {'L1_f1':>7}  (rescue {'ANY' if args.from_any else 'L1'}->L2)")
for tau in TAUS:
    pred, nf = rescue(base_pred, dgc, tau, args.from_any)
    pc = f1_score(y, pred, average=None, labels=list(range(N)))
    print(f"{tau:>6.3f} {f1_score(y, pred, average='macro'):>9.4f} {nf:>7d} {pc[2]:>7.3f} {pc[1]:>7.3f}")

# ---- NESTED CV: tune tau on train users, evaluate on held-out users ----
print("\n--- NESTED CV (tau chosen on disjoint users from eval) ---")
folds = list(GroupKFold(5).split(cal_prod, groups=groups))
nested_pred = base_pred.copy()
chosen = []
for outer_tr, outer_te in folds:
    best_tau, best_f1 = None, -1.0
    for tau in TAUS:
        pr, _ = rescue(base_pred[outer_tr], dgc[outer_tr], tau, args.from_any)
        f = f1_score(y[outer_tr], pr, average="macro")
        if f > best_f1:
            best_f1, best_tau = f, tau
    chosen.append(best_tau)
    pr_te, _ = rescue(base_pred[outer_te], dgc[outer_te], best_tau, args.from_any)
    nested_pred[outer_te] = pr_te
nested_f1 = f1_score(y, nested_pred, average="macro")
npc = f1_score(y, nested_pred, average=None, labels=list(range(N)))
print(f"  chosen tau per fold: {chosen}")
print(f"  production (frozen): macro-F1={base:.4f}")
print(f"  nested DG-rescue:    macro-F1={nested_f1:.4f}  (L2={npc[2]:.3f} L1={npc[1]:.3f} L3={npc[3]:.3f} L5={npc[5]:.3f})")
print(f"  >>> nested delta = {nested_f1 - base:+.4f} <<<")

if nested_f1 > base + 1e-4:
    tau_final = float(np.median(chosen))
    pred_t, nf = rescue((cal_prod_t * np.exp(PEAK_LOGW)).argmax(1), dgc_t, tau_final, args.from_any)
    tag = "any" if args.from_any else "l1"
    sub = ROOT / "submissions" / f"sub_dg_cisc_{args.name}_rescue_{tag}_tau{int(tau_final*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": pred_t.astype(int)}).to_csv(sub, index=False)
    print(f"\nNESTED WIN (+{nested_f1 - base:.4f}). Flipped {nf} test samples -> L2. Wrote {sub}")
else:
    print("\nNo honest (nested) improvement from the rescue gate.")
