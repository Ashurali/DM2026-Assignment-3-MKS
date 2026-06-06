"""ROUND 4 — beta saturated at 0.8234. New lever: DECOUPLE the per-class prior boosts
(uniform beta couples L2/L3). Tune L2,L3 (boost) and L4,L5 (suppress) independently
around the beta=2.0 equivalent (L2x1.59 L3x1.48 L4x0.54 L5x0.69). + orient-L2 inject,
robust threshold. Baseline = 0.8234. Run: .venv\\Scripts\\python.exe scripts\\gen_lb_round4.py
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]; OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
N = 6; ALPHA = 0.842
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
meta = pd.read_parquet(DATA / "meta_train.parquet"); y = meta["label"].values.astype(int); g = meta["user_id"].values
tid = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
Ld = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = Ld("lgbm_combo_combo_full_v2_oof.npy"), Ld("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = Ld("hier_v6_pipeline2_oof.npy"), Ld("hier_v6_pipeline2_test_probs.npy")
gkf = GroupKFold(5); norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(o, t):
    c = np.zeros_like(o); ct = np.zeros_like(t)
    for tr, va in gkf.split(o, groups=g):
        for k in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(o[tr, k], (y[tr] == k).astype(float)); c[va, k] = ir.predict(o[va, k])
    for k in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(o[:, k], (y == k).astype(float)); ct[:, k] = ir.predict(t[:, k])
    return norm(c), norm(ct)


cal, calt = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
oc, oct_ = iso(norm(Ld("orient_lgbm_oof.npy")), norm(Ld("orient_lgbm_test_probs.npy")))
tp = np.bincount(y, minlength=N) / len(y); pri = tp.copy()
for _ in range(300):
    nw = norm(calt * (pri / tp)).mean(0)
    if np.abs(nw - pri).max() < 1e-10: pri = nw; break
    pri = nw
wpri = pri / tp
injL = lambda c_, s, col, w: norm(np.concatenate([c_[:, :col], (1 - w) * c_[:, col:col+1] + w * s[:, col:col+1], c_[:, col+1:]], axis=1))
ref = (injL(norm(calt * wpri ** 2.0), oct_, 2, 0.15) * np.exp(ROBUST)).argmax(1)


def mult(v):  # per-class multiplier vector [L0..L5]
    return np.array([1.0, 1.0, v[0], v[1], v[2], v[3]])


def emit(name, m):
    w = mult(m)
    co = injL(norm(cal * w), oc, 2, 0.15); ct = injL(norm(calt * w), oct_, 2, 0.15)
    oofm = f1_score(y, (co * np.exp(ROBUST)).argmax(1), average="macro")
    pred = (ct * np.exp(ROBUST)).argmax(1); cnt = np.bincount(pred, minlength=N)
    pd.DataFrame({"Id": tid, "Label": pred.astype(int)}).to_csv(SUB / f"sub_{name}.csv", index=False)
    print(f"  {name:18s} L2x{m[0]:.2f} L3x{m[1]:.2f} L4x{m[2]:.2f} L5x{m[3]:.2f}  OOF={oofm:.4f} "
          f"L2={cnt[2]} L3={cnt[3]} L5={cnt[5]} diff={int((pred!=ref).sum()):3d}", flush=True)


print(f"ref 0.8234 (uniform b=2.0 = L2x{wpri[2]**2:.2f} L3x{wpri[3]**2:.2f} L4x{wpri[4]**2:.2f} L5x{wpri[5]**2:.2f}): "
      f"L2={np.bincount(ref,minlength=N)[2]} L3={np.bincount(ref,minlength=N)[3]}\n", flush=True)
# decoupled points: (L2, L3, L4, L5) multipliers
emit("dec_L2hi", [1.9, 1.45, 0.54, 0.69])     # more L2
emit("dec_L3hi", [1.55, 1.7, 0.54, 0.69])     # more L3
emit("dec_both_hi", [1.9, 1.7, 0.50, 0.62])   # more both, harder L4/L5 suppress
emit("dec_L2very", [2.2, 1.45, 0.50, 0.65])   # strong L2
emit("dec_softL5", [1.6, 1.5, 0.54, 0.80])    # keep more L5 (less suppress)
emit("dec_hardL45", [1.7, 1.6, 0.40, 0.55])   # suppress L4/L5 hard, moderate L2/L3
print("\n(decoupled L2/L3 may beat uniform 0.8234 if they want different boosts. LB decides.)", flush=True)
