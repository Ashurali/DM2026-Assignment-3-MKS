"""Re-tune the (L1,L2) grid thresholds FOR the injected blend, nested-CV validated.

Production peak/robust thresholds were grid-searched on the NON-injected blend.
The orientation L2-injection changes the probability distribution, so its optimal
(L1,L2) multipliers differ. Re-run the SAME 31x31 grid on the injected blend, but
NESTED (tune on train-users, eval on held-out) so any gain is honest cross-user
transfer, not threshold-trap overfitting. Only treat as real if nested macro
robustly beats the current robust+injection (0.7873 nested / 0.8200 public).

Run: python scripts/threshold_research_injected.py
"""
from __future__ import annotations
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
W = 0.15
GRID = np.linspace(-1.5, 1.5, 31)
# NM-derived base log-weights (production grid varied only positions 1,2 over this)
LW0 = np.array([0.4124252058711867, -0.35716588246639913, 0.7657415870597865,
                0.4628951701768874, -0.239947242877496, -0.42948082285098554])

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")


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


def inj(cal, src, w=W):
    out = cal.copy(); out[:, 2] = (1 - w) * cal[:, 2] + w * src[:, 2]
    return norm(out)


def grid_best(cal_sub, y_sub):
    best = (-1.0, LW0.copy())
    for l1 in GRID:
        for l2 in GRID:
            lw = LW0.copy(); lw[1] = l1; lw[2] = l2
            s = f1_score(y_sub, (cal_sub * np.exp(lw)).argmax(1), average="macro")
            if s > best[0]:
                best = (s, lw.copy())
    return best[1]


cal, calt = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
dgc, dgc_t = iso(norm(og), norm(ogt))
cal_inj, calt_inj = inj(cal, dgc), inj(calt, dgc_t)

# Reference points (frozen configs on the injected blend)
PEAK = np.array([0.4124252058711867, -0.3, 0.9, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.2, 0.9, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
print(f"injected blend @ PEAK   = {f1_score(y, (cal_inj*np.exp(PEAK)).argmax(1), average='macro'):.4f}", flush=True)
print(f"injected blend @ ROBUST = {f1_score(y, (cal_inj*np.exp(ROBUST)).argmax(1), average='macro'):.4f}", flush=True)

# NESTED grid search: tune (L1,L2) on train-users, apply to held-out
nested = np.zeros(len(y), int); chosen = []
for tr, te in GroupKFold(5).split(cal_inj, groups=groups):
    lw = grid_best(cal_inj[tr], y[tr])
    chosen.append((round(lw[1], 2), round(lw[2], 2)))
    nested[te] = (cal_inj[te] * np.exp(lw)).argmax(1)
nf = f1_score(y, nested, average="macro")
npc = f1_score(y, nested, average=None, labels=list(range(N)))
print(f"\nNESTED grid-retune on injected blend: macro={nf:.4f}  (L2={npc[2]:.3f} L1={npc[1]:.3f})", flush=True)
print(f"  per-fold (L1,L2): {chosen}", flush=True)
print(f"  vs robust+inj nested 0.7873 / peak+inj nested 0.7886  -> delta vs robust {nf - 0.7873:+.4f}", flush=True)

# Full-data grid-optimal config (for the candidate)
lw_full = grid_best(cal_inj, y)
full_f1 = f1_score(y, (cal_inj * np.exp(lw_full)).argmax(1), average="macro")
print(f"\nfull-data grid-optimal: L1={lw_full[1]:.2f} L2={lw_full[2]:.2f}  OOF macro={full_f1:.4f}", flush=True)
preds = (calt_inj * np.exp(lw_full)).argmax(1)
sub = ROOT / "submissions" / "sub_inject_gridretune.csv"
pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
print(f"wrote {sub}", flush=True)
print("\nNOTE: threshold re-tune is the project's highest-variance lever (OOF->LB "
      "instability gave 0.7698/0.7991/0.8154 historically). Treat as real ONLY if "
      "nested macro robustly exceeds 0.7873; submit to test public, keep 0.8200 as a final anchor.", flush=True)
