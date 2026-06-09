"""'Maximally tried' submission: joint nested orientation injection into BOTH the L1
and L2 columns (independent weights w1,w2 chosen by a 2-D nested-CV grid, so the
L1<->L2 interaction is handled), under FROZEN robust & peak thresholds. Same discipline
as every prior win: weights nested-chosen, NEVER tuned to the public LB.

Reports, per threshold config:
  baseline (no injection) | L2-only injection (the 0.8200 lever, sanity) | L1&L2 joint
and writes the deployment CSV using the median nested (w1,w2).

Run: .venv\\Scripts\\python.exe scripts\\max_tried_injection.py
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
OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
SUB.mkdir(exist_ok=True)
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.19999999999999996, 0.9000000000000004,
                   0.4628951701768874, -0.239947242877496, -0.42948082285098554])
WGRID1 = [0.0, 0.05, 0.10, 0.15, 0.20]                 # L1 injection
WGRID2 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]     # L2 injection

meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int)
g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso_pair(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in GroupKFold(5).split(oof_raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


cal, calt = iso_pair(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
dgc, dgc_t = iso_pair(norm(og), norm(ogt))


def inj2(c, src, w1, w2):
    out = c.copy()
    out[:, 1] = (1 - w1) * c[:, 1] + w1 * src[:, 1]
    out[:, 2] = (1 - w2) * c[:, 2] + w2 * src[:, 2]
    return norm(out)


for tag, lw in [("robust", ROBUST), ("peak", PEAK)]:
    base = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    # ---- L2-only nested (sanity vs the 0.8200 lever) ----
    cals_l2 = {w: inj2(cal, dgc, 0.0, w) for w in WGRID2}
    nl2 = np.zeros(len(y), int); ch_l2 = []
    for tr, te in GroupKFold(5).split(cal, groups=g):
        bw, bf = 0.0, -1
        for w in WGRID2:
            f = f1_score(y[tr], (cals_l2[w][tr] * np.exp(lw)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        ch_l2.append(bw); nl2[te] = (cals_l2[bw][te] * np.exp(lw)).argmax(1)
    f_l2 = f1_score(y, nl2, average="macro")
    # ---- L1 & L2 JOINT nested (2-D grid) ----
    cals = {(w1, w2): inj2(cal, dgc, w1, w2) for w1 in WGRID1 for w2 in WGRID2}
    nj = np.zeros(len(y), int); ch1, ch2 = [], []
    for tr, te in GroupKFold(5).split(cal, groups=g):
        bp, bf = (0.0, 0.0), -1
        for w1 in WGRID1:
            for w2 in WGRID2:
                f = f1_score(y[tr], (cals[(w1, w2)][tr] * np.exp(lw)).argmax(1), average="macro")
                if f > bf:
                    bf, bp = f, (w1, w2)
        ch1.append(bp[0]); ch2.append(bp[1]); nj[te] = (cals[bp][te] * np.exp(lw)).argmax(1)
    f_j = f1_score(y, nj, average="macro")
    w1f, w2f = float(np.median(ch1)), float(np.median(ch2))
    print(f"\n=== {tag.upper()} thresholds ===", flush=True)
    print(f"  baseline (no inj)      macro = {base:.4f}", flush=True)
    print(f"  L2-only injection      macro = {f_l2:.4f}  (delta {f_l2-base:+.4f}, "
          f"w2={np.median(ch_l2):.2f})  <- the 0.8200 lever", flush=True)
    print(f"  L1&L2 JOINT injection  macro = {f_j:.4f}  (delta {f_j-base:+.4f}, "
          f"w1={w1f:.2f} w2={w2f:.2f})", flush=True)
    print(f"  joint vs L2-only: {f_j-f_l2:+.4f}  per-fold (w1,w2)="
          f"{list(zip(ch1,ch2))}", flush=True)
    # ---- deployment CSV (median nested weights, NOT public-tuned) ----
    preds = (inj2(calt, dgc_t, w1f, w2f) * np.exp(lw)).argmax(1)
    out = SUB / f"sub_{tag}_orient_L1L2_maxtried_w{int(w1f*100):02d}_{int(w2f*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(out, index=False)
    # diff vs the proven L2-only deployment at this threshold
    l2_deploy = (inj2(calt, dgc_t, 0.0, float(np.median(ch_l2))) * np.exp(lw)).argmax(1)
    print(f"  wrote {out.name}  ({(preds!=l2_deploy).sum()} test rows differ from "
          f"L2-only deployment)", flush=True)

print("\nNOTE: nested-validated, NOT public-tuned. If joint ~= L2-only on OOF, expect a", flush=True)
print("public tie with 0.8200; submit to see the true number (OOF->public is noisy).", flush=True)
