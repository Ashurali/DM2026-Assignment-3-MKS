"""GAP B: adversarial-validation feature pruning (remove subject-spurious features).

We measured a train<->test shift (domain AUC 0.73) but never removed the features
driving it. Those high-domain-importance features are subject-specific spurious signals
that can hurt cross-subject generalization. Rank features by train/test discriminability,
ablate the top offenders, and check whether cross-user OOF macro-F1 HOLDS (or improves)
while the domain AUC drops toward 0.5 (= less reliance on spurious signal = more robust).

Local proxy uses a flat 6-class LightGBM (not the full production stack); if pruning
helps the proxy's cross-user OOF, it justifies a production retrain on the server.
Run: .venv\\Scripts\\python.exe scripts\\gapB_adversarial_pruning.py
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = 42; N = 6
FAMILIES = ["basic_stats", "fft", "autocorr", "subwindow", "gravity", "jerk",
            "crossaxis", "zerocross", "per_file_norm", "magnitude", "quality", "covariance"]


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
y = meta["label"].values.astype(int); g = meta["user_id"].values
Xtr = load_stack("train"); Xte = load_stack("test")
ids = meta["file_id"].values if "file_id" in meta.columns else Xtr.index.values
Xtr = Xtr.reindex(ids)
cols = [c for c in Xtr.columns if c in Xte.columns]
Xtr, Xte = Xtr[cols], Xte[cols]
feat = np.array(cols)
med = Xtr.replace([np.inf, -np.inf], np.nan).median()
Xv = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float32)
Xt = Xte.replace([np.inf, -np.inf], np.nan).fillna(med).values.astype(np.float32)
gkf = GroupKFold(5)
LGB = dict(n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1)

# ---------- (1) domain classifier: train(0) vs test(1) -> AUC + shift importance ----------
print("=== (1) Adversarial validation: train-vs-test domain classifier ===", flush=True)
Xd = np.vstack([Xv, Xt]); yd = np.r_[np.zeros(len(Xv)), np.ones(len(Xt))].astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
dp = cross_val_predict(lgb.LGBMClassifier(**{**LGB, "class_weight": None}), Xd, yd,
                       cv=skf, method="predict_proba")[:, 1]
dom_auc_full = roc_auc_score(yd, dp)
dclf = lgb.LGBMClassifier(**{**LGB, "class_weight": None}).fit(Xd, yd)
shift_imp = dclf.feature_importances_
order = np.argsort(shift_imp)[::-1]
print(f"  domain AUC (all {len(feat)} feats) = {dom_auc_full:.4f}", flush=True)
print("  top-12 shift-driving (subject-spurious) features:", flush=True)
for i in order[:12]:
    print(f"    {feat[i]:48s} shift-gain={shift_imp[i]}", flush=True)

# ---------- (2) baseline 6-class cross-user OOF macro-F1 + label importance ----------
print("\n=== (2) Baseline flat 6-class (all feats, cross-user OOF) ===", flush=True)


def cv_macro(Xin):
    pred = cross_val_predict(lgb.LGBMClassifier(objective="multiclass", num_class=N, **LGB),
                             Xin, y, groups=g, cv=gkf)
    return f1_score(y, pred, average="macro")


base_macro = cv_macro(Xv)
print(f"  macro-F1 = {base_macro:.4f}", flush=True)

# ---------- (3) ablate top-K shift features, re-measure macro-F1 + domain AUC ----------
print("\n=== (3) Ablate top-K shift features ===", flush=True)
print("   K     #feats  cross-user macro-F1   delta   domain-AUC(remaining)", flush=True)
for K in [0, 100, 300, 600, 1000]:
    keep = order[K:]                                  # drop the top-K shift drivers
    Xk = Xv[:, keep]
    mac = cv_macro(Xk)
    # recompute domain AUC on the remaining features
    Xdk = np.vstack([Xv[:, keep], Xt[:, keep]])
    dpk = cross_val_predict(lgb.LGBMClassifier(**{**LGB, "class_weight": None}), Xdk, yd,
                            cv=skf, method="predict_proba")[:, 1]
    print(f"   {K:5d} {len(keep):6d}    {mac:.4f}            {mac-base_macro:+.4f}   "
          f"{roc_auc_score(yd, dpk):.4f}", flush=True)

print("\n=== READ ===", flush=True)
print("  If a K holds/improves macro-F1 while dropping domain-AUC toward 0.5 -> the model", flush=True)
print("  relies less on subject-spurious features = more robust to the unseen split;", flush=True)
print("  worth a production retrain on the pruned set. If macro-F1 drops at every K, the", flush=True)
print("  shifted features also carry label signal -> pruning trades accuracy for nothing.", flush=True)
