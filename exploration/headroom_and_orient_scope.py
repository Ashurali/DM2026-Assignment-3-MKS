"""Two questions:
 (1) WHERE is error reducible? Per-class one-vs-rest Bayes-error bracket (balanced,
     cross-user kNN) vs our strong classifier. If classifier-err >> Bayes-hi -> a
     better model can still help that class; if classifier-err ~ Bayes -> at the floor.
     (User asked specifically about L0/L1; we report all six.)
 (2) Does the orientation pseudo-gyro source help BEYOND L2? We currently inject it
     ONLY into the L2 column. Check its per-class one-vs-rest AUC vs production, then
     nested-inject it into {L2},{L3},{L5},{L1},{L2,L3},{L2,L5},{L2,L3,L5} and see if
     macro-F1 beats the L2-only injection. (Same discipline: frozen PEAK, nested w.)

Local, CPU. Run: .venv\\Scripts\\python.exe scripts\\headroom_and_orient_scope.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OOF = ROOT / "oof"
N = 6; RNG = 42; ALPHA = 0.842
np.random.seed(RNG)
PEAK = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
FAMILIES = ["basic_stats", "fft", "autocorr", "subwindow", "gravity", "jerk",
            "crossaxis", "zerocross", "per_file_norm", "magnitude", "quality",
            "covariance"]


def load_stack(split):
    frames = []
    for fam in FAMILIES:
        p = DATA / f"feat_{split}_{fam}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "file_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "file_id"})
        frames.append(df.set_index("file_id").pipe(
            lambda d: d.loc[:, ~d.columns.duplicated()]).add_prefix(f"{fam}__"))
    cp = DATA / f"feat_catch22_{split}.parquet"
    if cp.exists():
        dfc = pd.read_parquet(cp)
        if "file_id" not in dfc.columns:
            dfc = dfc.reset_index().rename(columns={"index": "file_id"})
        frames.append(dfc.set_index("file_id").add_prefix("c22__"))
    X = pd.concat(frames, axis=1); return X.loc[:, ~X.columns.duplicated()]


meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int)
g = meta["user_id"].values
X = load_stack("train")
ids = meta["file_id"].values if "file_id" in meta.columns else X.index.values
X = X.reindex(ids)
med = X.replace([np.inf, -np.inf], np.nan).median()
Xv = X.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float64)
Z = StandardScaler().fit_transform(Xv)
P = PCA(n_components=50, random_state=RNG).fit_transform(Z)
gkf = GroupKFold(5)
LGB = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1)
CNT = np.bincount(y)

# ---------- (1) per-class reducible headroom ----------
print("=== (1) Per-class reducible headroom (one-vs-rest, balanced, cross-user) ===",
      flush=True)
print("  class  n     6cls-F1   Bayes-bracket      BayesACC   strongACC   verdict",
      flush=True)
prod_f1 = {0: 0.967, 1: 0.908, 2: 0.384, 3: 0.764, 4: 0.924, 5: 0.781}
for c in range(N):
    ic = np.where(y == c)[0]; ir = np.where(y != c)[0]
    n = min(len(ic), len(ir), 2000)
    e1, e21, sacc = [], [], []
    for s in range(3):
        rs = np.random.RandomState(RNG + s)
        sa = rs.choice(ic, n, replace=False); sb = rs.choice(ir, n, replace=False)
        sel = np.r_[sa, sb]; ys = np.r_[np.ones(n), np.zeros(n)].astype(int); gs = g[sel]
        e1.append((cross_val_predict(KNeighborsClassifier(1), P[sel], ys, groups=gs,
                   cv=gkf) != ys).mean())
        e21.append((cross_val_predict(KNeighborsClassifier(21), P[sel], ys, groups=gs,
                    cv=gkf) != ys).mean())
        sacc.append((cross_val_predict(lgb.LGBMClassifier(**LGB), Xv[sel], ys,
                     groups=gs, cv=gkf) == ys).mean())
    lo, hi = np.mean(e1) / 2, np.mean(e21)
    sa_ = np.mean(sacc); serr = 1 - sa_
    verdict = "AT FLOOR" if serr <= hi + 0.03 else "REDUCIBLE"
    print(f"  L{c}   {CNT[c]:4d}   {prod_f1[c]:.3f}    [{lo:.3f},{hi:.3f}]      "
          f"{1-hi:.3f}      {sa_:.3f}      {verdict}", flush=True)
print("  (BayesACC = balanced one-vs-rest; strongACC = strong LightGBM same setting.", flush=True)
print("   REDUCIBLE => a better base model could still lower that class's error.)",
      flush=True)

# ---------- load production blend + orient source, calibrate ----------
load = lambda n: np.load(OOF / n).astype(np.float64)
p1 = load("lgbm_combo_combo_full_v2_oof.npy"); p2 = load("hier_v6_pipeline2_oof.npy")
orient = load("orient_lgbm_oof.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(raw):
    cal = np.zeros_like(raw)
    for tr, va in gkf.split(raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(raw[va, c])
    return norm(cal)


cal = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2))
ocal = iso(norm(orient))

# ---------- (2a) orient per-class one-vs-rest AUC vs production ----------
print("\n=== (2a) Does orientation carry signal beyond L2? (one-vs-rest AUC) ===",
      flush=True)
print("  class   prod-AUC   orient-AUC   orient complementary?", flush=True)
for c in range(N):
    ap = roc_auc_score((y == c).astype(int), cal[:, c])
    ao = roc_auc_score((y == c).astype(int), ocal[:, c])
    print(f"  L{c}     {ap:.3f}      {ao:.3f}        {'yes-ish' if ao > 0.6 else 'weak'}",
          flush=True)

# ---------- (2b) nested injection into various column sets ----------
print("\n=== (2b) nested orient injection into different columns (PEAK, macro-F1) ===",
      flush=True)
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def inject(c_, src, cols, w):
    out = c_.copy()
    for col in cols:
        out[:, col] = (1 - w) * c_[:, col] + w * src[:, col]
    return norm(out)


base = f1_score(y, (cal * np.exp(PEAK)).argmax(1), average="macro")
print(f"  baseline (no injection) macro-F1 = {base:.4f}", flush=True)
for cols in [(2,), (3,), (5,), (1,), (2, 3), (2, 5), (2, 3, 5), (1, 2, 3, 5)]:
    cals = {w: inject(cal, ocal, cols, w) for w in WGRID}
    nested = np.zeros(len(y), int); chosen = []
    for tr, te in gkf.split(cal, groups=g):
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(PEAK)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        chosen.append(bw); nested[te] = (cals[bw][te] * np.exp(PEAK)).argmax(1)
    nf = f1_score(y, nested, average="macro")
    tag = "L" + "+L".join(str(c) for c in cols)
    print(f"  inject into {tag:12s}: nested macro={nf:.4f}  delta {nf-base:+.4f}  "
          f"w(median)={np.median(chosen):.2f}", flush=True)

print("\n=== READ ===", flush=True)
print("  (1) tells you which classes still have reducible error worth a model push.", flush=True)
print("  (2) tells you whether widening orientation beyond the L2 column helps -- if any", flush=True)
print("  multi-column inject clearly beats L2-only (current 0.8200 lever), it's a new win.",
      flush=True)
