"""Unsupervised / self-supervised feasibility probe for the L1<->L2 bottleneck.

Question (user): can self-supervised / unsupervised methods reveal data structure
we're missing? This decides whether an SSL pretraining run on the server is worth it.

Strategy -- measure, don't assume (project lesson). On the FULL engineered feature
stack (already built, local), with honest cross-user (GroupKFold-by-user) estimates:

  PROBE A  L1-vs-L2 separability: supervised UPPER bound (LR / RF, cross-user) vs
           unsupervised LOWER bound (KMeans best-cluster->label map). The gap is the
           verdict: signal absent (both low) => SSL can't help; signal present but
           geometry can't find it (sup high, unsup low) => representation lever exists.
  PROBE B  What does unsupervised geometry actually track -- activity or user?
           KMeans on all train; ARI(clusters,label) vs ARI(clusters,user). If user
           dominates, that's WHY cross-subject shift is the wall (motivates user-
           invariant SSL... but only if A says the signal is recoverable).
  PROBE C  Train<->test covariate shift: domain-classifier AUC (can a model tell a
           train row from a test row?). High = strong shift = transductive SSL on
           pooled train+test is motivated.
  PROBE D  L2 neighbourhood purity: fraction of each L2 sample's kNN that are L1
           (reproduces/extends the "62% have an L1 nearest neighbour" finding across
           the full stack, at several k).

Local, CPU, ~1-2 min. No server, no GPU, no submission. Pure diagnostics.
Run: .venv\\Scripts\\python.exe scripts\\unsup_separability_probe.py
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, adjusted_rand_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = 42
np.random.seed(RNG)

FAMILIES = ["basic_stats", "fft", "autocorr", "subwindow", "gravity", "jerk",
            "crossaxis", "zerocross", "per_file_norm", "magnitude", "quality",
            "covariance"]


def load_stack(split: str) -> pd.DataFrame:
    """Merge all engineered feature families on file_id for a split."""
    frames = []
    for fam in FAMILIES:
        p = DATA / f"feat_{split}_{fam}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "file_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "file_id"})
        df = df.set_index("file_id")
        df = df.loc[:, ~df.columns.duplicated()]
        frames.append(df.add_prefix(f"{fam}__"))
    # catch22 (different naming)
    cp = DATA / f"feat_catch22_{split}.parquet"
    if cp.exists():
        dfc = pd.read_parquet(cp)
        if "file_id" not in dfc.columns:
            dfc = dfc.reset_index().rename(columns={"index": "file_id"})
        dfc = dfc.set_index("file_id")
        frames.append(dfc.add_prefix("c22__"))
    X = pd.concat(frames, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X


print("Loading feature stacks...", flush=True)
meta_tr = pd.read_parquet(DATA / "meta_train.parquet")
meta_te = pd.read_parquet(DATA / "meta_test.parquet")
y = meta_tr["label"].values.astype(int)
groups = meta_tr["user_id"].values
n_users = len(np.unique(groups))

Xtr = load_stack("train")
Xte = load_stack("test")
# align rows to meta order via file_id
tr_ids = meta_tr["file_id"].values if "file_id" in meta_tr.columns else Xtr.index.values
te_ids = meta_te["file_id"].values if "file_id" in meta_te.columns else Xte.index.values
Xtr = Xtr.reindex(tr_ids)
Xte = Xte.reindex(te_ids)
cols = [c for c in Xtr.columns if c in Xte.columns]
Xtr, Xte = Xtr[cols], Xte[cols]
print(f"  train {Xtr.shape}  test {Xte.shape}  users={n_users}  "
      f"class counts={np.bincount(y)}", flush=True)

# clean: inf->nan, fill with train median, standardize on train
med = Xtr.replace([np.inf, -np.inf], np.nan).median()
Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float64)
Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float64)
sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
pca = PCA(n_components=50, random_state=RNG).fit(Ztr)
Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
print(f"  PCA50 cum-var explained = {pca.explained_variance_ratio_.sum():.3f}", flush=True)

gkf = GroupKFold(5)

# ---------- PROBE A: L1-vs-L2 separability, supervised UB vs unsupervised LB ----------
print("\n=== PROBE A: L1-vs-L2 separability (cross-user) ===", flush=True)
m = (y == 1) | (y == 2)
Xa, ya, ga = Ztr[m], (y[m] == 2).astype(int), groups[m]   # raw standardized feats
Pa = Ptr[m]                                               # PCA feats
print(f"  L1={int((ya==0).sum())}  L2={int((ya==1).sum())}  "
      f"base-rate(L2)={ya.mean():.3f}", flush=True)


def cv_f1(clf, X, yy, gg):
    pred = cross_val_predict(clf, X, yy, groups=gg, cv=gkf)
    return f1_score(yy, pred, average="macro"), f1_score(yy, pred, pos_label=1)


# supervised upper bounds (cross-user honest)
lr_f1m, lr_f1l2 = cv_f1(LogisticRegression(max_iter=2000, class_weight="balanced",
                                            C=1.0), Xa, ya, ga)
rf_f1m, rf_f1l2 = cv_f1(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                               n_jobs=-1, random_state=RNG), Xa, ya, ga)
hb_f1m, hb_f1l2 = cv_f1(HistGradientBoostingClassifier(max_iter=400, random_state=RNG),
                        Xa, ya, ga)
print(f"  SUPERVISED  LR   macroF1={lr_f1m:.4f}  L2-F1={lr_f1l2:.4f}", flush=True)
print(f"  SUPERVISED  RF   macroF1={rf_f1m:.4f}  L2-F1={rf_f1l2:.4f}", flush=True)
print(f"  SUPERVISED  HGB  macroF1={hb_f1m:.4f}  L2-F1={hb_f1l2:.4f}", flush=True)

# unsupervised lower bound: KMeans on PCA feats, best cluster->label map
best_unsup = -1.0
for k in [2, 4, 8, 16]:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Pa)
    lab = km.labels_
    mapped = np.zeros_like(ya)
    for c in range(k):                       # majority-vote map each cluster
        cm = lab == c
        if cm.sum():
            mapped[cm] = int(round(ya[cm].mean()))
    fm = f1_score(ya, mapped, average="macro")
    ari = adjusted_rand_score(ya, lab)
    best_unsup = max(best_unsup, fm)
    print(f"  UNSUPERVISED KMeans(k={k:2d}) best-map macroF1={fm:.4f}  ARI={ari:+.4f}",
          flush=True)
print(f"  --> supervised UB (best)={max(lr_f1m,rf_f1m,hb_f1m):.4f}   "
      f"unsupervised LB (best)={best_unsup:.4f}", flush=True)

# ---------- PROBE B: does geometry track activity or user? ----------
print("\n=== PROBE B: unsupervised geometry tracks activity or USER? ===", flush=True)
for k in [6, 20, n_users]:
    km = KMeans(n_clusters=int(k), n_init=10, random_state=RNG).fit(Ptr)
    ari_lab = adjusted_rand_score(y, km.labels_)
    ari_usr = adjusted_rand_score(groups, km.labels_)
    print(f"  KMeans(k={int(k):3d})  ARI(clusters,label)={ari_lab:+.4f}   "
          f"ARI(clusters,user)={ari_usr:+.4f}", flush=True)

# ---------- PROBE C: train<->test covariate shift (domain classifier) ----------
print("\n=== PROBE C: train<->test covariate shift (domain-classifier AUC) ===",
      flush=True)
Xdom = np.vstack([Ztr, Zte])
ydom = np.r_[np.zeros(len(Ztr)), np.ones(len(Zte))]
idx = np.random.permutation(len(ydom))
dom = HistGradientBoostingClassifier(max_iter=300, random_state=RNG)
from sklearn.model_selection import cross_val_predict as cvp
proba = cvp(dom, Xdom[idx], ydom[idx], cv=5, method="predict_proba")[:, 1]
auc = roc_auc_score(ydom[idx], proba)
print(f"  domain AUC = {auc:.4f}   (0.5=no shift, 1.0=fully separable)", flush=True)

# ---------- PROBE D: L2 neighbourhood purity ----------
print("\n=== PROBE D: L2 kNN purity (fraction of neighbours that are L1) ===",
      flush=True)
nn = NearestNeighbors(n_neighbors=21).fit(Ptr)
_, nbr = nn.kneighbors(Ptr)
for k in [5, 10, 20]:
    l2 = np.where(y == 2)[0]
    nb = nbr[l2, 1:k + 1]
    frac_l1 = (y[nb] == 1).mean()
    frac_l2 = (y[nb] == 2).mean()
    print(f"  k={k:2d}:  L2 neighbours that are L1={frac_l1:.3f}   "
          f"that are L2={frac_l2:.3f}", flush=True)

print("\n=== VERDICT GUIDE ===", flush=True)
print("  If SUPERVISED L2-F1 ~= production sep (~0.62) and UNSUP << that:", flush=True)
print("    -> signal is present but weak; geometry alone can't find it; SSL MIGHT", flush=True)
print("       surface a little -- worth a server run.", flush=True)
print("  If SUPERVISED L2-F1 also ceilings near production:", flush=True)
print("    -> information floor confirmed from the label-free side; SSL won't beat", flush=True)
print("       the 1Hz/no-gyro data. 0.8200 stands.", flush=True)
