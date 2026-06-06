"""General source-integration test: is a new base model COMPLEMENTARY + does it STACK
past production (0.7880 PEAK / 0.7864 ROBUST)? Reuse for any source with OOF+test probs.

Usage: .venv\\Scripts\\python.exe scripts\\integrate_source.py --name inception_time_v1
"""
from __future__ import annotations
import argparse, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); A = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; OOF = ROOT / "oof"; DATA = ROOT / "data"
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.30, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
meta = pd.read_parquet(DATA / "meta_train.parquet"); y = meta["label"].values.astype(int); g = meta["user_id"].values
load = lambda n: np.load(OOF / n).astype(np.float64)
p1 = load("lgbm_combo_combo_full_v2_oof.npy"); p2 = load("hier_v6_pipeline2_oof.npy"); og = load("orient_lgbm_oof.npy")
src = load(f"{A.name}_oof.npy")
gkf = GroupKFold(5)
norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(o):
    cal = np.zeros_like(o)
    for tr, va in gkf.split(o, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(o[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(o[va, c])
    return norm(cal)


cal = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2)); scal = iso(norm(src))
print(f"=== {A.name}: standalone macro={f1_score(y, src.argmax(1), average='macro'):.4f} ===", flush=True)
prod = (cal * np.exp(PEAK)).argmax(1); sp = src.argmax(1)
print("(1) complementarity:", flush=True)
for c in [2, 3, 5]:
    miss = (y == c) & (prod != c)
    print(f"   L{c}: prod-miss={int(miss.sum())} src-rescues={int((miss & (sp == c)).sum())}", flush=True)
print("   AUC prod/src: " + " ".join(f"L{c}:{roc_auc_score((y==c),cal[:,c]):.3f}/{roc_auc_score((y==c),scal[:,c]):.3f}" for c in range(N)), flush=True)
WG = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def inj(c_, s, cols, w):
    o = c_.copy()
    for col in cols: o[:, col] = (1 - w) * c_[:, col] + w * s[:, col]
    return norm(o)


print("(2) nested injection:", flush=True)
for tag, lw in [("PEAK", PEAK), ("ROBUST", ROBUST)]:
    base = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    for cols in [(2,), (3,), (5,), (2, 3), (2, 3, 5)]:
        cals = {w: inj(cal, scal, cols, w) for w in WG}; out = np.zeros(len(y), int)
        for tr, te in gkf.split(cal, groups=g):
            bw, bf = 0.0, -1
            for w in WG:
                f = f1_score(y[tr], (cals[w][tr] * np.exp(lw)).argmax(1), average="macro")
                if f > bf: bf, bw = f, w
            out[te] = (cals[bw][te] * np.exp(lw)).argmax(1)
        nf = f1_score(y, out, average="macro")
        mark = "  <== BEATS" if nf - base > 0.0008 else ""
        print(f"   {tag:6s} L{'+L'.join(map(str,cols)):8s} base={base:.4f} nested={nf:.4f} ({nf-base:+.4f}){mark}", flush=True)
print(f"\n(baseline PEAK=0.7880 ROBUST=0.7864; need clear nested beat to use {A.name})", flush=True)
