"""Does a hierarchical CASCADE (L1-vs-rest -> L2-vs-rest -> {L0,L3,L4,L5}) beat a FLAT
6-class model? Head-to-head on identical features + identical post-processing
(per-class isotonic 5-fold OOF + nested coordinate-ascent threshold), cross-user.

The user's hypothesis: since L2 is separable (AUC .86) but capped by the joint
operating point, peeling classes one at a time with dedicated binary stages might
extract more. Test it instead of arguing.

Architectures (same features, same calibration+threshold):
  FLAT     : one 6-class LightGBM.
  CASCADE-U: user's order  L1 | L2 | {L0,L3,L4,L5}  (soft chain-rule, no hard peel).
  CASCADE-E: easy-first    L0 | L4 | L1 | L2 | {L3,L5}  (minimises error propagation).

Soft chain rule (no hard decisions => best case for a cascade): each stage trained on
the not-yet-peeled subset; joint prob = product of conditionals. Report macro-F1 +
per-class F1 (esp. L2) so we see WHERE they differ.

Local, CPU, ~3-5 min. Run: .venv\\Scripts\\python.exe scripts\\hierarchy_vs_flat.py
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OOF = ROOT / "oof"
N = 6; RNG = 42
FAMILIES = ["basic_stats", "fft", "autocorr", "subwindow", "gravity", "jerk",
            "crossaxis", "zerocross", "per_file_norm", "magnitude", "quality",
            "covariance"]
LGB = dict(n_estimators=250, learning_rate=0.04, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           random_state=RNG, n_jobs=-1, verbose=-1)


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
print(f"X={Xv.shape}  classes={np.bincount(y)}", flush=True)
gkf = GroupKFold(5)


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def fit_flat(Xtr, ytr, Xva):
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=N,
                             class_weight="balanced", **LGB).fit(Xtr, ytr)
    proba = np.zeros((len(Xva), N))
    proba[:, clf.classes_] = clf.predict_proba(Xva)
    return proba


def make_cascade(order_peel):
    """order_peel: list of single classes to peel in sequence; the final remaining
    set is modelled jointly as a multiclass stage."""
    def fit_cascade(Xtr, ytr, Xva):
        proba = np.zeros((len(Xva), N))
        remaining_mass = np.ones(len(Xva))
        active = ytr.copy()
        active_mask = np.ones(len(ytr), bool)   # train rows still 'in play'
        for c in order_peel:
            mtr = active_mask
            bclf = lgb.LGBMClassifier(class_weight="balanced", **LGB).fit(
                Xtr[mtr], (ytr[mtr] == c).astype(int))
            pc = bclf.predict_proba(Xva)[:, 1]
            proba[:, c] = remaining_mass * pc
            remaining_mass = remaining_mass * (1 - pc)
            active_mask = active_mask & (ytr != c)
        rem_classes = sorted(set(range(N)) - set(order_peel))
        mtr = active_mask
        remap = {cl: i for i, cl in enumerate(rem_classes)}
        ytr_r = np.array([remap[v] for v in ytr[mtr]])
        mclf = lgb.LGBMClassifier(objective="multiclass", num_class=len(rem_classes),
                                  class_weight="balanced", **LGB).fit(Xtr[mtr], ytr_r)
        pm = np.zeros((len(Xva), len(rem_classes)))
        pm[:, mclf.classes_] = mclf.predict_proba(Xva)
        for i, cl in enumerate(rem_classes):
            proba[:, cl] = remaining_mass * pm[:, i]
        return norm(proba)
    return fit_cascade


def oof_probs(fit_fn):
    oof = np.zeros((len(y), N))
    for tr, va in gkf.split(Xv, groups=g):
        oof[va] = fit_fn(Xv[tr], y[tr], Xv[va])
    return norm(oof)


def iso(raw):
    cal = np.zeros_like(raw)
    for tr, va in gkf.split(raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(raw[tr, c], (y[tr] == c).astype(float))
            cal[va, c] = ir.predict(raw[va, c])
    return norm(cal)


def opt_weights(cal_tr, y_tr):
    w = np.zeros(N)
    for _ in range(4):
        for c in range(N):
            best_v, best_f = w[c], -1
            for cand in np.arange(w[c] - 1.5, w[c] + 1.5 + 1e-9, 0.1):
                w2 = w.copy(); w2[c] = cand
                f = f1_score(y_tr, (cal_tr * np.exp(w2)).argmax(1), average="macro")
                if f > best_f:
                    best_f, best_v = f, cand
            w[c] = best_v
    return w


def nested_eval(cal):
    nested = np.zeros(len(y), int)
    for tr, te in gkf.split(cal, groups=g):
        w = opt_weights(cal[tr], y[tr])
        nested[te] = (cal[te] * np.exp(w)).argmax(1)
    return (f1_score(y, nested, average="macro"),
            f1_score(y, nested, average=None, labels=list(range(N))))


archs = {
    "FLAT      ": fit_flat,
    "CASCADE-U ": make_cascade([1, 2]),            # L1 | L2 | {L0,L3,L4,L5}
    "CASCADE-E ": make_cascade([0, 4, 1, 2]),      # L0 | L4 | L1 | L2 | {L3,L5}
}
print("\n  arch        nested-macro    L0    L1    L2    L3    L4    L5", flush=True)
results = {}
for name, fn in archs.items():
    raw = oof_probs(fn)
    cal = iso(raw)
    m, per = nested_eval(cal)
    results[name] = m
    print(f"  {name}  {m:.4f}        " +
          "  ".join(f"{per[c]:.3f}" for c in range(N)), flush=True)

print(f"\n  (production tuned blend reference: 0.7880)", flush=True)
best = max(results, key=results.get)
print(f"\n=== VERDICT ===", flush=True)
print(f"  best architecture on identical features = {best.strip()} ({results[best]:.4f})",
      flush=True)
print("  If FLAT >= CASCADE: the joint model's cross-class information beats sequential", flush=True)
print("  peeling; the cascade can't escape the imbalance/entanglement cap and pays an", flush=True)
print("  error-propagation cost on L2. If a CASCADE clearly wins: a real architecture", flush=True)
print("  lever exists -> rebuild on the server with the full base + validate on LB.", flush=True)
