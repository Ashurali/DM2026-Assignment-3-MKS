"""Characterise L2 in the LITERATURE's own terms, to ground 'why the textbook fixes
failed' in measured data rather than assertion.

(1) Napierala & Stefanowski (2016) minority-example TYPOLOGY: classify each L2 sample
    by how many same-class neighbours it has among its k=5 NN ->
      safe (4-5) | borderline (2-3) | rare (1) | outlier (0).
    'Unsafe' (borderline+rare+outlier) examples are provably hard; SMOTE/resampling
    are known to fail on them (they synthesise into the overlap, blurring boundaries).

(2) BAYES-ERROR bracket for the overlapping pairs L1|L2, L2|L3, L2|L5 (balanced,
    cross-user). kNN error brackets the Bayes error: BER in [e_1NN/2, e_largek].
    If our strong classifier's error ~ the estimated BER, we are AT the irreducible
    floor and no representation/architecture change can help (only new information).

Local, CPU. Run: .venv\\Scripts\\python.exe scripts\\l2_typology_bayes.py
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
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
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

# ---------- (1) Napierala-Stefanowski typology of L2 ----------
print("=== (1) Napierala-Stefanowski typology of L2 (k=5 same-class neighbours) ===",
      flush=True)
nn = NearestNeighbors(n_neighbors=6).fit(P)       # +1 for self
_, idx = nn.kneighbors(P)
l2 = np.where(y == 2)[0]
same = np.array([(y[idx[i, 1:6]] == 2).sum() for i in l2])   # exclude self
typ = {"safe (4-5)": ((same >= 4)).sum(), "borderline (2-3)": ((same >= 2) & (same <= 3)).sum(),
       "rare (1)": (same == 1).sum(), "outlier (0)": (same == 0).sum()}
tot = len(l2)
for k, v in typ.items():
    print(f"  {k:18s}: {v:4d}  ({100*v/tot:5.1f}%)", flush=True)
unsafe = typ["borderline (2-3)"] + typ["rare (1)"] + typ["outlier (0)"]
print(f"  --> UNSAFE (borderline+rare+outlier) = {unsafe}/{tot} = {100*unsafe/tot:.1f}%",
      flush=True)
print("  (SMOTE/resampling are known to fail on unsafe examples -- they synthesise into",
      flush=True)
print("   the majority/overlap region, blurring the boundary.)", flush=True)

# ---------- (2) Bayes-error bracket for overlapping pairs ----------
gkf = GroupKFold(5)
LGB = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1)


def bayes_bracket(ca, cb, name):
    """balanced cross-user error: kNN (brackets BER) + strong LightGBM (achievable)."""
    ia, ib = np.where(y == ca)[0], np.where(y == cb)[0]
    n = min(len(ia), len(ib))
    errs_knn = {}
    e1 = []
    accs_strong = []
    for s in range(3):                              # average over 3 balanced draws
        rs = np.random.RandomState(RNG + s)
        sa = rs.choice(ia, n, replace=False); sb = rs.choice(ib, n, replace=False)
        sel = np.r_[sa, sb]
        Ps, ys, gs = P[sel], (y[sel] == cb).astype(int), g[sel]
        for k in [1, 5, 11, 21]:
            pr = cross_val_predict(KNeighborsClassifier(n_neighbors=k), Ps, ys,
                                   groups=gs, cv=gkf)
            errs_knn.setdefault(k, []).append((pr != ys).mean())
        prs = cross_val_predict(lgb.LGBMClassifier(**LGB), Xv[sel], ys, groups=gs,
                                cv=gkf)
        accs_strong.append((prs == ys).mean())
    e1 = np.mean(errs_knn[1]); elarge = np.mean(errs_knn[21])
    ber_lo, ber_hi = e1 / 2, elarge
    print(f"\n  {name}  (balanced n={n}/class, cross-user):", flush=True)
    print("    kNN error: " + "  ".join(
        f"k={k}:{np.mean(errs_knn[k]):.3f}" for k in [1, 5, 11, 21]), flush=True)
    print(f"    => Bayes-error bracket ~ [{ber_lo:.3f}, {ber_hi:.3f}]  "
          f"(balanced Bayes ACC ~ {1-elarge:.3f})", flush=True)
    print(f"    strong LightGBM balanced ACC = {np.mean(accs_strong):.3f}  "
          f"(err {1-np.mean(accs_strong):.3f}) -> "
          f"{'AT the floor' if 1-np.mean(accs_strong) <= elarge + 0.03 else 'headroom'}",
          flush=True)


print("\n=== (2) Bayes-error brackets for L2's overlaps ===", flush=True)
bayes_bracket(1, 2, "L1 vs L2")
bayes_bracket(2, 3, "L2 vs L3")
bayes_bracket(2, 5, "L2 vs L5")

print("\n=== TAKEAWAY ===", flush=True)
print("  If L2 is mostly UNSAFE and the strong classifier's error ~ the kNN Bayes", flush=True)
print("  bracket, then L2's residual confusion is IRREDUCIBLE given these features:", flush=True)
print("  the failure of resampling/margin/contrastive is expected, and the only", flush=True)
print("  literature-sanctioned remedy is NEW INFORMATION (features/modalities/data).", flush=True)
