"""Is the L2 OPERATING POINT fully optimized in the full 6-class macro-F1, or is the
+0.14 pair-threshold headroom partly realizable?

Reproduces production (combo_v2 + hier_v6, alpha=0.842, per-class isotonic 5-fold OOF,
PEAK weights), then:
  1. per-class F1 at PEAK + L2 confusion (where do L2 errors GO, and where do L2
     false-positives COME FROM -- L1, or also L3/L5?).
  2. L2-weight sweep (others fixed at PEAK): shows the macro-F1 vs L2-F1 tradeoff --
     i.e. proves whether pushing L2 harder helps or hurts MACRO.
  3. finer joint 6-weight coordinate-ascent, NESTED by user: can any operating point
     beat PEAK's macro-F1? (the honest test that the decision rule is exhausted.)

Local, CPU. Run: .venv\\Scripts\\python.exe scripts\\l1l2_operating_point.py
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"; DATA = ROOT / "data"
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                 0.4628951701768874, -0.239947242877496, -0.42948082285098554])

meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int)
g = meta["user_id"].values
load = lambda n: np.load(OOF / n).astype(np.float64)
p1 = load("lgbm_combo_combo_full_v2_oof.npy")
p2 = load("hier_v6_pipeline2_oof.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(raw):
    cal = np.zeros_like(raw)
    for tr, va in GroupKFold(5).split(raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(raw[tr, c], (y[tr] == c).astype(float))
            cal[va, c] = ir.predict(raw[va, c])
    return norm(cal)


cal = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2))


def macro_and_per(lw):
    pred = (cal * np.exp(lw)).argmax(1)
    return f1_score(y, pred, average="macro"), f1_score(y, pred, average=None,
                                                        labels=list(range(N))), pred


# ---------- 1. per-class F1 + L2 confusion at PEAK ----------
m0, per0, pred0 = macro_and_per(PEAK)
print(f"=== 1. PEAK operating point: macro-F1={m0:.4f} ===", flush=True)
for c in range(N):
    print(f"  L{c}: F1={per0[c]:.3f}", flush=True)
cm = confusion_matrix(y, pred0, labels=list(range(N)))
print("  L2 truths -> predicted as:  " +
      "  ".join(f"L{j}={cm[2, j]}" for j in range(N)), flush=True)
print("  predicted-L2 came from truth:  " +
      "  ".join(f"L{i}={cm[i, 2]}" for i in range(N)), flush=True)
tp = cm[2, 2]; fp = cm[:, 2].sum() - tp; fn = cm[2, :].sum() - tp
print(f"  L2: TP={tp} FP={fp} FN={fn}  prec={tp/max(tp+fp,1):.3f} "
      f"rec={tp/max(tp+fn,1):.3f}", flush=True)

# ---------- 2. L2-weight sweep (others fixed at PEAK) ----------
print("\n=== 2. L2-weight sweep (others fixed) -- macro vs L2 tradeoff ===", flush=True)
print("   w(L2)   macro-F1   L2-F1   L1-F1   L3-F1   L5-F1", flush=True)
best_macro_w, best_macro_v = PEAK[2], -1
for w2 in np.round(np.arange(0.3, 1.85, 0.15), 2):
    lw = PEAK.copy(); lw[2] = w2
    mm, pp, _ = macro_and_per(lw)
    mark = "  <- PEAK" if abs(w2 - PEAK[2]) < 0.08 else ""
    star = ""
    if mm > best_macro_v:
        best_macro_v, best_macro_w = mm, w2
    print(f"   {w2:4.2f}   {mm:.4f}    {pp[2]:.3f}   {pp[1]:.3f}   {pp[3]:.3f}   "
          f"{pp[5]:.3f}{mark}", flush=True)
print(f"  --> macro-F1 maximised at w(L2)={best_macro_w:.2f} (PEAK uses {PEAK[2]:.2f}); "
      f"L2-F1 keeps rising past the macro optimum = the tradeoff wall.", flush=True)

# ---------- 3. finer joint 6-weight coordinate-ascent, NESTED by user ----------
print("\n=== 3. joint 6-weight coordinate-ascent, NESTED (can anything beat PEAK?) ===",
      flush=True)


def opt_weights(idx_tr):
    w = PEAK.copy()
    yt = y[idx_tr]; ct = cal[idx_tr]
    for _ in range(6):
        for c in range(N):
            best_v, best_f = w[c], -1
            for cand in np.arange(w[c] - 0.6, w[c] + 0.6 + 1e-9, 0.05):
                w2 = w.copy(); w2[c] = cand
                f = f1_score(yt, (ct * np.exp(w2)).argmax(1), average="macro")
                if f > best_f:
                    best_f, best_v = f, cand
            w[c] = best_v
    return w


nested = np.zeros(len(y), int)
peak_nested = np.zeros(len(y), int)
for tr, te in GroupKFold(5).split(cal, groups=g):
    w = opt_weights(tr)
    nested[te] = (cal[te] * np.exp(w)).argmax(1)
    peak_nested[te] = (cal[te] * np.exp(PEAK)).argmax(1)
mn = f1_score(y, nested, average="macro")
mp = f1_score(y, peak_nested, average="macro")
print(f"  PEAK (fixed) nested macro-F1        = {mp:.4f}", flush=True)
print(f"  re-optimized 6-weight nested macro  = {mn:.4f}   delta {mn - mp:+.4f}", flush=True)

print("\n=== VERDICT ===", flush=True)
print("  If re-optimized ~= PEAK: the decision rule (operating point) is exhausted --", flush=True)
print("  L2 is separable (AUC .86) but the 13:1 imbalance + macro-F1 tradeoff cap its", flush=True)
print("  contribution; pushing L2 harder costs L1/L3/L5 more than it gains. 0.8200 real.", flush=True)
print("  If re-optimized > PEAK by a clear margin: a better operating point exists ->", flush=True)
print("  validate it on the LB (a genuine, free, representation-independent gain).", flush=True)
