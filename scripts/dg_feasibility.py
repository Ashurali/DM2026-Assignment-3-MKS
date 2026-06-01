"""Feasibility gate for the CISC DG model — is its L2/L3/L5 signal COMPLEMENTARY?

Augmentation can only help if the DG model gets right what the production stack
gets WRONG. So we measure, on the clean user-disjoint OOF:
  1. DG standalone per-class F1 (vs production peak: L2=.384 L3=.764 L5=.781).
  2. Complementarity: P(DG argmax correct | production WRONG) for c in {2,3,5}.
  3. Oracle ceiling: macro-F1 if an ideal gate picked the correct source per sample.
  4. L1<->L2 binary separability (the structural bottleneck).

All on frozen OOFs; production final preds use the FROZEN top-1 thresholds.
Run:  python scripts/dg_feasibility.py [--name v1]
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
args = ap.parse_args()

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1 = load("lgbm_combo_combo_full_v2_oof.npy")
p2 = load("hier_v6_pipeline2_oof.npy")
dg = load(f"dg_cisc_{args.name}_oof.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso_oof(raw):
    cal = np.zeros_like(raw)
    for tr, va in GroupKFold(5).split(raw, groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(raw[tr, c], (y[tr] == c).astype(float))
            cal[va, c] = ir.predict(raw[va, c])
    return norm(cal)


# Production final prediction under FROZEN top-1 thresholds
cal_prod = iso_oof(norm(ALPHA * p1 + (1 - ALPHA) * p2))
prod_pred = (cal_prod * np.exp(PEAK_LOGW)).argmax(1)
dg_pred = dg.argmax(1)

prod_f1 = f1_score(y, prod_pred, average=None, labels=list(range(N)))
dg_f1 = f1_score(y, dg_pred, average=None, labels=list(range(N)))
print(f"{'class':6} {'prod_F1':>8} {'DG_F1':>8} {'prevalence':>11}")
for c in range(N):
    print(f"L{c:<5} {prod_f1[c]:>8.3f} {dg_f1[c]:>8.3f} {np.mean(y == c) * 100:>10.1f}%")
print(f"{'macro':6} {prod_f1.mean():>8.4f} {dg_f1.mean():>8.4f}")

print("\n--- Complementarity (does DG rescue production's misses?) ---")
for c in [2, 3, 5]:
    mask_c = y == c
    prod_wrong = mask_c & (prod_pred != c)
    dg_right_where_prod_wrong = prod_wrong & (dg_pred == c)
    print(f"L{c}: true={int(mask_c.sum())}  prod_misses={int(prod_wrong.sum())}  "
          f"of those DG_correct={int(dg_right_where_prod_wrong.sum())} "
          f"({100*dg_right_where_prod_wrong.sum()/max(1,prod_wrong.sum()):.1f}%)")
    # reverse: where DG wrong, does prod save it
    dg_wrong = mask_c & (dg_pred != c)
    prod_right = dg_wrong & (prod_pred == c)
    print(f"    reverse: DG_misses={int(dg_wrong.sum())}  prod_rescues={int(prod_right.sum())}")

# Oracle ceiling: per-sample pick whichever source is correct (upper bound on any gate)
oracle = np.where(prod_pred == y, prod_pred, np.where(dg_pred == y, dg_pred, prod_pred))
print(f"\nOracle (ideal per-sample gate prod|DG): macro-F1={f1_score(y, oracle, average='macro'):.4f}  "
      f"(prod={prod_f1.mean():.4f})  -> max headroom a gate could capture")
opc = f1_score(y, oracle, average=None, labels=list(range(N)))
print(f"  oracle per-class: L2={opc[2]:.3f} L3={opc[3]:.3f} L5={opc[5]:.3f}")

print("\n--- L1<->L2 binary separability (walking subset) ---")
wmask = (y == 1) | (y == 2)
yb = (y[wmask] == 2).astype(int)
for tag, P in [("production", cal_prod), ("DG", norm(dg))]:
    pp = P[wmask][:, [1, 2]]
    pp = pp / np.clip(pp.sum(1, keepdims=True), 1e-12, None)
    pred = pp.argmax(1)
    print(f"  {tag:>10}: L1vL2 macroF1={f1_score(yb, pred, average='macro'):.4f}  "
          f"L2recall={(pred[yb == 1] == 1).mean():.3f}")

print("\nVERDICT: integration is worth pursuing iff DG rescues a non-trivial share "
      "of production's L2/L3/L5 misses AND the oracle ceiling exceeds 0.7880.")
