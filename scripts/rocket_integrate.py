"""Integrate rocket_rich into production: is it COMPLEMENTARY and does it STACK?
Runs locally once oof/rocket_rich_oof.npy + _test_probs.npy are pulled from the server.

Tests (same discipline as every prior source):
  (1) complementarity: does ROCKET rescue production's L2/L3/L5 misses? per-class AUC.
  (2) nested injection into L2/L3/L5 + combos under PEAK & ROBUST -- beat baseline?
  (3) stacking: nested LightGBM meta over [combo,hier,orient,rocket] -- beat 0.7880?
If anything clears the bar, writes a submission.

Run: .venv\\Scripts\\python.exe scripts\\rocket_integrate.py
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.30, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])

if not (OOF / "rocket_rich_oof.npy").exists():
    print("rocket_rich_oof.npy not present yet -- pull it from the server first.", flush=True)
    sys.exit(0)

meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int); g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")
rk, rkt = load("rocket_rich_oof.npy"), load("rocket_rich_test_probs.npy")
gkf = GroupKFold(5)


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso_pair(o, t):
    cal = np.zeros_like(o)
    for tr, va in gkf.split(o, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(o[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(o[va, c])
    ct = np.zeros_like(t)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(o[:, c], (y == c).astype(float)); ct[:, c] = ir.predict(t[:, c])
    return norm(cal), norm(ct)


cal, calt = iso_pair(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
rcal, rcalt = iso_pair(norm(rk), norm(rkt))

# standalone
mr = f1_score(y, rk.argmax(1), average="macro")
per = f1_score(y, rk.argmax(1), average=None, labels=list(range(N)))
print(f"=== ROCKET-rich standalone: macro={mr:.4f} per-class={[round(float(p),3) for p in per]} ===",
      flush=True)

# (1) complementarity
print("\n(1) complementarity vs production:", flush=True)
prod = (cal * np.exp(PEAK)).argmax(1); rp = rk.argmax(1)
for c in [2, 3, 5]:
    miss = (y == c) & (prod != c)
    print(f"   L{c}: prod misses={int(miss.sum())}  rocket-correct={int((miss & (rp == c)).sum())}",
          flush=True)
print("   per-class AUC prod/rocket: " +
      " ".join(f"L{c}:{roc_auc_score((y==c).astype(int),cal[:,c]):.3f}/"
               f"{roc_auc_score((y==c).astype(int),rcal[:,c]):.3f}" for c in range(N)), flush=True)

# (2) nested injection
print("\n(2) nested injection (delta vs baseline):", flush=True)
WG = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def inj(c_, s, cols, w):
    o = c_.copy()
    for col in cols:
        o[:, col] = (1 - w) * c_[:, col] + w * s[:, col]
    return norm(o)


def nested(cols, lw):
    base = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    cals = {w: inj(cal, rcal, cols, w) for w in WG}
    out = np.zeros(len(y), int); ch = []
    for tr, te in gkf.split(cal, groups=g):
        bw, bf = 0.0, -1
        for w in WG:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(lw)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        ch.append(bw); out[te] = (cals[bw][te] * np.exp(lw)).argmax(1)
    return base, f1_score(y, out, average="macro"), float(np.median(ch))


for tag, lw in [("PEAK", PEAK), ("ROBUST", ROBUST)]:
    for cols in [(2,), (3,), (5,), (2, 3), (2, 3, 5)]:
        b, nf, w = nested(cols, lw)
        t = "L" + "+L".join(map(str, cols))
        mark = "  <-- WIN" if nf - b > 0.0008 else ""
        print(f"   {tag:6s} {t:10s} base={b:.4f} nested={nf:.4f} ({nf-b:+.4f}) w={w:.2f}{mark}",
              flush=True)

# (3) stacking meta over base probs
print("\n(3) nested LightGBM meta-stack [combo,hier,orient,rocket]:", flush=True)
F = np.hstack([norm(p1), norm(p2), norm(og), norm(rk)])
Ft = np.hstack([norm(p1t), norm(p2t), norm(ogt), norm(rkt)])
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8,
           colsample_bytree=0.7, min_child_samples=40, reg_lambda=2.0,
           class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
oof = np.zeros((len(y), N))
for tr, va in gkf.split(F, groups=g):
    m = lgb.LGBMClassifier(objective="multiclass", num_class=N, **LGB).fit(F[tr], y[tr])
    oof[np.ix_(va, m.classes_)] = m.predict_proba(F[va])
oof = norm(oof)
scal, _ = iso_pair(oof, oof)
for tag, lw in [("PEAK", PEAK), ("ROBUST", ROBUST)]:
    print(f"   {tag}: meta-stack macro={f1_score(y,(scal*np.exp(lw)).argmax(1),average='macro'):.4f}",
          flush=True)
print("\n(baseline production PEAK=0.7880 ROBUST=0.7864; need a clear nested beat to use ROCKET)",
      flush=True)
