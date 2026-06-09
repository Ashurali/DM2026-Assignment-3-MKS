"""L1-vs-L2: are they REALLY inseparable, or just rare? -- the decisive decomposition.

The first probe (unsup_separability_probe.py) used macro-F1, which conflates
"inseparable" with "imbalanced". This one measures PURE separability with metrics
that are robust to the 13:1 imbalance, and -- the crux -- decomposes WITHIN-user vs
CROSS-user. Those two are entirely different problems:

  * within-user AUC HIGH, cross-user AUC LOW  => signal IS there but user-conditional
    => domain-adaptation lever exists (per-user calibration / transductive / TTA).
  * within-user AUC ALSO low                  => intrinsic physical overlap at 1Hz;
    no representation trick recovers it. 0.8200 ceiling confirmed.

Metrics: ROC-AUC + average-precision (imbalance-robust) for L1(neg) vs L2(pos).
Models: LightGBM (strong nonlinear) + balanced Logistic (linear) + Fisher-LDA axis.
Also: per-user within-subject AUC (users with both classes), top discriminative
features, and a PCA-variance check (is the L2 signal in discarded low-var dirs?).

Local, CPU, ~1-2 min. Run: .venv\\Scripts\\python.exe scripts\\l1l2_deep_dive.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = 42
np.random.seed(RNG)
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
        df = df.set_index("file_id"); df = df.loc[:, ~df.columns.duplicated()]
        frames.append(df.add_prefix(f"{fam}__"))
    cp = DATA / f"feat_catch22_{split}.parquet"
    if cp.exists():
        dfc = pd.read_parquet(cp)
        if "file_id" not in dfc.columns:
            dfc = dfc.reset_index().rename(columns={"index": "file_id"})
        frames.append(dfc.set_index("file_id").add_prefix("c22__"))
    X = pd.concat(frames, axis=1); return X.loc[:, ~X.columns.duplicated()]


print("Loading...", flush=True)
meta = pd.read_parquet(DATA / "meta_train.parquet")
y_all = meta["label"].values.astype(int)
g_all = meta["user_id"].values
X = load_stack("train")
ids = meta["file_id"].values if "file_id" in meta.columns else X.index.values
X = X.reindex(ids)
med = X.replace([np.inf, -np.inf], np.nan).median()
Xv = X.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float64)
feat_names = np.array(X.columns)

m = (y_all == 1) | (y_all == 2)
Xb, gb = Xv[m], g_all[m]
yb = (y_all[m] == 2).astype(int)        # L2 = positive
sc = StandardScaler().fit(Xb); Zb = sc.transform(Xb)
print(f"  L1(neg)={int((yb==0).sum())}  L2(pos)={int((yb==1).sum())}  "
      f"users={len(np.unique(gb))}", flush=True)

LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1)


def auc_ap(proba):
    return roc_auc_score(yb, proba), average_precision_score(yb, proba)


# ---------- 1. PURE separability: random vs cross-user (AUC + AP) ----------
print("\n=== 1. L1-vs-L2 separability: RANDOM split vs CROSS-USER split ===", flush=True)
print("   (AUC: 0.5=inseparable, 1.0=perfect. AP baseline = L2 base-rate "
      f"{yb.mean():.3f})", flush=True)
for tag, clf, Xin in [("LightGBM", lgb.LGBMClassifier(**LGB), Xb),
                      ("Logistic", LogisticRegression(max_iter=3000,
                       class_weight="balanced", C=0.5), Zb)]:
    pr = cross_val_predict(clf, Xin, yb, cv=StratifiedKFold(5, shuffle=True,
                           random_state=RNG), method="predict_proba")[:, 1]
    a_r, ap_r = auc_ap(pr)
    pc = cross_val_predict(clf, Xin, yb, groups=gb, cv=GroupKFold(5),
                           method="predict_proba")[:, 1]
    a_c, ap_c = auc_ap(pc)
    print(f"  {tag:9s}  RANDOM AUC={a_r:.4f} AP={ap_r:.4f} | "
          f"CROSS-USER AUC={a_c:.4f} AP={ap_c:.4f} | shift-gap={a_r-a_c:+.4f}",
          flush=True)

# ---------- 2. WITHIN-USER separability (the crux) ----------
print("\n=== 2. WITHIN-USER separability (users having BOTH classes) ===", flush=True)
rows = []
for u in np.unique(gb):
    mu = gb == u; n1 = int((yb[mu] == 0).sum()); n2 = int((yb[mu] == 1).sum())
    if n1 >= 8 and n2 >= 8:
        rows.append((u, n1, n2))
print(f"  qualifying users (>=8 of each): {len(rows)}", flush=True)
aucs = []
for u, n1, n2 in rows:
    mu = gb == u; Xu, yu = Xb[mu], yb[mu]
    k = min(5, n2)
    try:
        pr = cross_val_predict(lgb.LGBMClassifier(**{**LGB, "n_estimators": 200}),
                               Xu, yu, cv=StratifiedKFold(k, shuffle=True,
                               random_state=RNG), method="predict_proba")[:, 1]
        aucs.append((roc_auc_score(yu, pr), n1 + n2))
    except Exception as e:
        print(f"    user {u}: skip ({e})", flush=True)
if aucs:
    a = np.array([x[0] for x in aucs]); w = np.array([x[1] for x in aucs])
    print(f"  WITHIN-USER AUC: mean={a.mean():.4f}  weighted={np.average(a, weights=w):.4f}"
          f"  median={np.median(a):.4f}  min={a.min():.4f} max={a.max():.4f}", flush=True)
    print(f"  (#users with within-AUC>0.8: {(a>0.8).sum()}/{len(a)};  "
          f">0.9: {(a>0.9).sum()}/{len(a)})", flush=True)

# ---------- 3. Fisher-LDA best linear axis (cross-user) ----------
print("\n=== 3. Fisher-LDA single best axis (cross-user AUC) ===", flush=True)
proj = np.zeros(len(yb))
for tr, te in GroupKFold(5).split(Zb, groups=gb):
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Zb[tr], yb[tr])
    proj[te] = lda.transform(Zb[te]).ravel()
print(f"  LDA-axis cross-user AUC={roc_auc_score(yb, proj):.4f}", flush=True)

# ---------- 4. PCA-variance: is L2 signal in discarded low-var directions? ----------
print("\n=== 4. PCA-variance check (cross-user AUC, LightGBM) ===", flush=True)
for nc in [50, 200, 500]:
    nc = min(nc, Zb.shape[1] - 1)
    P = PCA(n_components=nc, random_state=RNG).fit_transform(Zb)
    pr = cross_val_predict(lgb.LGBMClassifier(**LGB), P, yb, groups=gb,
                           cv=GroupKFold(5), method="predict_proba")[:, 1]
    var = PCA(n_components=nc, random_state=RNG).fit(Zb).explained_variance_ratio_.sum()
    print(f"  PCA{nc:4d} (var={var:.3f})  cross-user AUC={roc_auc_score(yb, pr):.4f}",
          flush=True)

# ---------- 5. Which features carry the L1-vs-L2 signal? ----------
print("\n=== 5. Top discriminative features (LightGBM gain, full data) ===", flush=True)
clf = lgb.LGBMClassifier(**LGB).fit(Xb, yb)
imp = clf.feature_importances_
order = np.argsort(imp)[::-1][:18]
for i in order:
    print(f"  {feat_names[i]:45s}  gain={imp[i]}", flush=True)
# family-level rollup
fam_gain = {}
for nm, gn in zip(feat_names, imp):
    fam = nm.split("__")[0]; fam_gain[fam] = fam_gain.get(fam, 0) + gn
print("  -- family rollup --", flush=True)
for fam, gn in sorted(fam_gain.items(), key=lambda x: -x[1]):
    print(f"    {fam:14s} {gn}", flush=True)

print("\n=== READ THIS ===", flush=True)
print("  within-user AUC HIGH (>0.85) but cross-user LOW => SEPARABLE, user-conditional", flush=True)
print("    => new lever: per-user/test-time adaptation, NOT a dead end.", flush=True)
print("  within-user AUC ALSO modest => intrinsic 1Hz overlap; ceiling real.", flush=True)
