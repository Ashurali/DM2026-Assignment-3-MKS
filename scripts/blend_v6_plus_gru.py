"""Inject gru_pn_ms into the production v6 blend and sweep its weight β.

Production: blend = α·P1(combo_full_v2) + (1−α)·P2(hier_v6), α=0.842 → isotonic
→ NM + 31×31 L1/L2 grid threshold → OOF 0.7880 / LB 0.8154 (all FROZEN OOFs).

Here: blend_β = (1−β)·[α·P1 + (1−α)·P2] + β·P_gru, then the SAME pipeline.
β=0 must reproduce 0.7880 (sanity). If β>0 beats it (esp on L2), the GRU's
L2 signal helps the production ensemble — no retraining, frozen OOFs only.

Run: python scripts/blend_v6_plus_gru.py
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
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
N = 6
ALPHA = 0.842
BETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(np.int64)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)


def load(name):
    p = ROOT / "oof" / name
    if not p.exists():
        raise SystemExit(f"MISSING: {p} — fetch it from the server first.")
    return np.load(p).astype(np.float64)


p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
pg, pgt = load("gru_pn_ms_oof.npy"), load("gru_pn_ms_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def isotonic(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in GroupKFold(5).split(np.zeros(len(y)), groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float))
            cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float))
        calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


def nm_thresholds(probs):
    f = lambda lw: -f1_score(y, (probs * np.exp(lw)).argmax(1), average="macro")
    bx, bv = np.zeros(N), f(np.zeros(N))
    rng = np.random.default_rng(42)
    for x0 in [np.zeros(N)] + [rng.uniform(-1, 1, N) for _ in range(8)]:
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True})
        if r.fun < bv:
            bv, bx = r.fun, r.x
    return bx


def grid_peak(cal, lw0, g=31, rng=1.5):
    lg = np.linspace(-rng, rng, g)
    best = (-1.0, lw0.copy())
    for l1 in lg:
        for l2 in lg:
            lw = lw0.copy(); lw[1] = l1; lw[2] = l2
            s = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
            if s > best[0]:
                best = (s, lw.copy())
    return best


base_oof = norm(ALPHA * p1 + (1 - ALPHA) * p2)
base_test = norm(ALPHA * p1t + (1 - ALPHA) * p2t)

print(f"{'beta':>5} {'peakF1':>8} {'L2':>7} {'L1':>7}   (production β=0 should ≈ 0.7880)", flush=True)
results = []
for beta in BETAS:
    bo = norm((1 - beta) * base_oof + beta * pg)
    bt = norm((1 - beta) * base_test + beta * pgt)
    co, ct = isotonic(bo, bt)
    lw = nm_thresholds(co)
    pf, lwp = grid_peak(co, lw)
    pc = f1_score(y, (co * np.exp(lwp)).argmax(1), average=None)
    print(f"{beta:>5.2f} {pf:>8.4f} {pc[2]:>7.3f} {pc[1]:>7.3f}", flush=True)
    results.append((beta, pf, pc, lwp, ct))

base_f1 = results[0][1]
best = max(results, key=lambda r: r[1])
b, pf, pc, lwp, ct = best
print(f"\nproduction (β=0): {base_f1:.4f}   best: β={b} → {pf:.4f}  (Δ {pf - base_f1:+.4f})", flush=True)
print(f"best per-class: {[round(float(x), 4) for x in pc]}", flush=True)
if b > 0 and pf > base_f1 + 1e-4:
    preds = (ct * np.exp(lwp)).argmax(1)
    sub = ROOT / "submissions" / f"sub_v6_gru_b{int(b*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
    print(f"GRU HELPS — wrote {sub}", flush=True)
else:
    print("GRU does not improve the production blend's OOF.", flush=True)
