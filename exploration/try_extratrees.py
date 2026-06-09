"""Does ExtraTrees add anything? Three questions, same discipline as every prior test:
 (1) Standalone quality cross-user (macro-F1 + per-class + one-vs-rest AUC).
 (2) Complementarity: does it rescue production's minority misses / out-rank production
     on any class?
 (3) Does it STACK? nested injection of ET into L2/L3, alone and ON TOP OF the orient-L2
     lever, plus ET added to the P1+P2 blend. (The audit showed cat/xgb/cnn carry signal
     individually but collapse to <=+0.0001 when stacked -- is ET different?)

Saves et_v1 OOF + test probs in case it ever helps.
Local, CPU. Run: .venv\\Scripts\\python.exe scripts\\try_extratrees.py
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OOF = ROOT / "oof"
N = 6; RNG = 42; ALPHA = 0.842
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
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
Xtr = load_stack("train"); Xte = load_stack("test")
ids = meta["file_id"].values if "file_id" in meta.columns else Xtr.index.values
Xtr = Xtr.reindex(ids)
cols = [c for c in Xtr.columns if c in Xte.columns]
Xtr, Xte = Xtr[cols], Xte[cols]
med = Xtr.replace([np.inf, -np.inf], np.nan).median()
Xv = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float32)
Xt = Xte.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float32)
gkf = GroupKFold(5)


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


ET = dict(n_estimators=700, max_features="sqrt", min_samples_leaf=2,
          class_weight="balanced_subsample", bootstrap=False, n_jobs=-1,
          random_state=RNG)

# ---------- (1) standalone ET, cross-user OOF ----------
print("=== (1) ExtraTrees standalone (cross-user OOF) ===", flush=True)
et_oof = np.zeros((len(y), N))
for tr, va in gkf.split(Xv, groups=g):
    clf = ExtraTreesClassifier(**ET).fit(Xv[tr], y[tr])
    et_oof[va] = clf.predict_proba(Xv[va])
et_oof = norm(et_oof)
m_raw = f1_score(y, et_oof.argmax(1), average="macro")
per = f1_score(y, et_oof.argmax(1), average=None, labels=list(range(N)))
print(f"  raw-argmax macro-F1={m_raw:.4f}  per-class=" +
      " ".join(f"L{c}:{per[c]:.3f}" for c in range(N)), flush=True)
print("  (ref: production blend per-class [.967 .908 .384 .764 .924 .781], macro .788;"
      " LightGBM-flat-on-same-feats ~0.733)", flush=True)
print("  one-vs-rest AUC: " +
      " ".join(f"L{c}:{roc_auc_score((y==c).astype(int), et_oof[:,c]):.3f}"
               for c in range(N)), flush=True)

# fit ET on full train -> test probs (save regardless)
et_full = ExtraTreesClassifier(**ET).fit(Xv, y)
et_test = norm(et_full.predict_proba(Xt))
np.save(OOF / "et_v1_oof.npy", et_oof.astype(np.float32))
np.save(OOF / "et_v1_test_probs.npy", et_test.astype(np.float32))
print("  saved oof/et_v1_oof.npy + _test_probs.npy", flush=True)

# ---------- production blend + orient, calibrated ----------
load = lambda n: np.load(OOF / n).astype(np.float64)
p1 = load("lgbm_combo_combo_full_v2_oof.npy"); p2 = load("hier_v6_pipeline2_oof.npy")
og = load("orient_lgbm_oof.npy")


def iso(raw):
    cal = np.zeros_like(raw)
    for tr, va in gkf.split(raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(raw[va, c])
    return norm(cal)


cal = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2))
ocal = iso(norm(og))
etc = iso(et_oof)

# ---------- (2) complementarity ----------
print("\n=== (2) Complementarity vs production ===", flush=True)
prod_pred = (cal * np.exp(PEAK)).argmax(1)
et_pred = et_oof.argmax(1)
for c in [2, 3, 5]:
    miss = (y == c) & (prod_pred != c)
    print(f"  L{c}: prod misses={int(miss.sum())}  ET-correct-of-those="
          f"{int((miss & (et_pred == c)).sum())}", flush=True)
print("  per-class AUC  prod vs ET: " +
      " ".join(f"L{c}:{roc_auc_score((y==c).astype(int),cal[:,c]):.3f}/"
               f"{roc_auc_score((y==c).astype(int),etc[:,c]):.3f}" for c in range(N)),
      flush=True)

# ---------- (3) does it STACK? nested injection + blend ----------
print("\n=== (3) Does ET stack? (nested, PEAK thresholds) ===", flush=True)
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
base = f1_score(y, (cal * np.exp(PEAK)).argmax(1), average="macro")
print(f"  baseline macro-F1 = {base:.4f}", flush=True)


def inj(c_, src, cols_, w):
    out = c_.copy()
    for col in cols_:
        out[:, col] = (1 - w) * c_[:, col] + w * src[:, col]
    return norm(out)


def nested_inj(src, cols_, also=None):
    """nested w for injecting src into cols_; `also` = a fixed prior injection (orient L2)."""
    cband = cal if also is None else inj(cal, ocal, [2], 0.15)  # orient-L2 already applied
    cals = {w: inj(cband, src, cols_, w) for w in WGRID}
    nested = np.zeros(len(y), int)
    for tr, te in gkf.split(cal, groups=g):
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(PEAK)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        nested[te] = (cals[bw][te] * np.exp(PEAK)).argmax(1)
    return f1_score(y, nested, average="macro")


orient_l2 = nested_inj(ocal, [2])
print(f"  orient -> L2 (the 0.8200 lever)   = {orient_l2:.4f}  ({orient_l2-base:+.4f})",
      flush=True)
print(f"  ET     -> L2                      = {nested_inj(etc,[2]):.4f}  "
      f"({nested_inj(etc,[2])-base:+.4f})", flush=True)
print(f"  ET     -> L3                      = {nested_inj(etc,[3]):.4f}  "
      f"({nested_inj(etc,[3])-base:+.4f})", flush=True)
print(f"  ET     -> L2+L3                   = {nested_inj(etc,[2,3]):.4f}  "
      f"({nested_inj(etc,[2,3])-base:+.4f})", flush=True)
stack = nested_inj(etc, [3], also=True)
print(f"  orient->L2 THEN ET->L3 (stack?)   = {stack:.4f}  ({stack-orient_l2:+.4f} vs "
      f"orient-L2 alone)", flush=True)

# add ET to the blend
print("\n  -- ET added to the P1+P2 blend --", flush=True)
for beta in [0.0, 0.1, 0.2, 0.3]:
    bl = iso(norm((1 - beta) * norm(ALPHA * p1 + (1 - ALPHA) * p2) + beta * et_oof))
    mf = f1_score(y, (bl * np.exp(PEAK)).argmax(1), average="macro")
    print(f"   beta(ET)={beta:.1f}: macro-F1={mf:.4f}  ({mf-base:+.4f})", flush=True)

print("\n=== VERDICT ===", flush=True)
print("  ET helps only if it (a) ranks some class better than production, AND (b) its", flush=True)
print("  injection STACKS on top of orient-L2 (delta>0 vs orient alone) or its blend", flush=True)
print("  weight is nonzero with a clear macro gain. Otherwise it's another source", flush=True)
print("  fighting the same boundary samples -> dead, like cat/xgb/cnn.", flush=True)
