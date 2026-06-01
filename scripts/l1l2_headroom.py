"""Given L1-vs-L2 IS separable (AUC 0.854 cross-user), is there HEADROOM we're not
extracting -- or does production already capture it?

Three numbers decide it (all on the L1+L2 subset, cross-user honest):
  (a) a dedicated strong binary L1-vs-L2 ranker  -> AUC + max-achievable L2-F1
  (b) PRODUCTION's own L2 ranking (calibrated blend OOF, alpha=0.842) restricted to
      L1-vs-L2 -> AUC + L2-F1 at argmax + L2-F1 at its OWN best threshold
  (c) the existing pairwise oof/pair_l1_v_l2 -> AUC

Logic:
  * If dedicated AUC >> production AUC: production isn't capturing the signal; a
    dedicated L2 source has headroom (inject it, like we did orientation).
  * If AUCs ~equal: production already ranks L2 as well as possible; the limiter is
    the imbalance operating-point, and we measure how much F1 is left vs what we get.
  * gap (max-F1 - argmax-F1) = how much pure threshold/calibration can still buy.

Local, CPU. Run: .venv\\Scripts\\python.exe scripts\\l1l2_headroom.py
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OOF = ROOT / "oof"
RNG = 42
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

m = (y == 1) | (y == 2)
yb = (y[m] == 2).astype(int)
gb = g[m]
gkf = GroupKFold(5)


def best_f1(yt, score):
    """max F1 for the positive (L2) class over all thresholds + the threshold."""
    order = np.argsort(score)[::-1]
    ys = yt[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P = ys.sum()
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    k = int(np.argmax(f1))
    return f1[k], prec[k], rec[k], score[order][k]


LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
           colsample_bytree=0.6, min_child_samples=20, reg_lambda=1.0,
           class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1)

# ---------- (a) dedicated strong binary ranker, cross-user OOF ----------
print("=== (a) DEDICATED binary L1-vs-L2 ranker (cross-user) ===", flush=True)
sc = StandardScaler().fit(Xv[m])
pr = cross_val_predict(lgb.LGBMClassifier(**LGB), Xv[m], yb, groups=gb, cv=gkf,
                       method="predict_proba")[:, 1]
auc_a = roc_auc_score(yb, pr); ap_a = average_precision_score(yb, pr)
f1_a, p_a, r_a, _ = best_f1(yb, pr)
print(f"  AUC={auc_a:.4f}  AP={ap_a:.4f}  MAX L2-F1={f1_a:.4f} (prec={p_a:.3f} rec={r_a:.3f})",
      flush=True)

# ---------- (b) PRODUCTION's own L2 ranking on the same subset ----------
print("\n=== (b) PRODUCTION blend (alpha=0.842) L2 ranking, same L1+L2 subset ===",
      flush=True)
ALPHA = 0.842
load = lambda n: np.load(OOF / n).astype(np.float64)
try:
    p1 = load("lgbm_combo_combo_full_v2_oof.npy")
    p2 = load("hier_v6_pipeline2_oof.npy")
    blend = ALPHA * p1 + (1 - ALPHA) * p2
    blend = blend / blend.sum(1, keepdims=True)
    bl = blend[m]
    # L2-vs-L1 score from production = p(L2) / (p(L1)+p(L2))
    l2score = bl[:, 2] / np.clip(bl[:, 1] + bl[:, 2], 1e-12, None)
    auc_b = roc_auc_score(yb, l2score); ap_b = average_precision_score(yb, l2score)
    # argmax-among-6 reduced to this pair: predict L2 iff p(L2)>p(L1)
    argmax_pred = (bl[:, 2] > bl[:, 1]).astype(int)
    f1_argmax = f1_score(yb, argmax_pred, pos_label=1)
    f1_b, p_b, r_b, _ = best_f1(yb, l2score)
    print(f"  AUC={auc_b:.4f}  AP={ap_b:.4f}", flush=True)
    print(f"  L2-F1 at production argmax (p_L2>p_L1) = {f1_argmax:.4f}", flush=True)
    print(f"  L2-F1 at production's OWN best threshold = {f1_b:.4f} "
          f"(prec={p_b:.3f} rec={r_b:.3f})", flush=True)
    print(f"  --> threshold headroom inside production ranking = "
          f"{f1_b - f1_argmax:+.4f}", flush=True)
except Exception as e:
    print(f"  (could not load production OOF: {e})", flush=True)
    auc_b = None

# ---------- (c) existing pairwise oof ----------
print("\n=== (c) existing oof/pair_l1_v_l2 ===", flush=True)
try:
    pair = load("pair_l1_v_l2_oof.npy")
    ps = pair[m]
    if ps.ndim == 2:
        ps = ps[:, -1] if ps.shape[1] == 2 else ps[:, 2]
    print(f"  shape={pair.shape}  AUC={roc_auc_score(yb, ps):.4f}", flush=True)
except Exception as e:
    print(f"  (n/a: {e})", flush=True)

# ---------- verdict ----------
print("\n=== HEADROOM VERDICT ===", flush=True)
if auc_b is not None:
    print(f"  dedicated AUC {auc_a:.3f}  vs  production-L2 AUC {auc_b:.3f}  "
          f"(delta {auc_a - auc_b:+.3f})", flush=True)
    if auc_a - auc_b > 0.02:
        print("  -> production UNDER-ranks L2: a dedicated L2 source has real headroom.",
              flush=True)
    else:
        print("  -> production ALREADY ranks L2 ~as well as a dedicated model: the", flush=True)
        print("     limiter is the imbalance operating-point, not the representation.", flush=True)
    print(f"  Pure-threshold headroom on production's L2 = {f1_b - f1_argmax:+.4f} "
          f"(argmax {f1_argmax:.3f} -> best {f1_b:.3f}).", flush=True)
    print("  NOTE: the (L1,L2) threshold GRID already exploits this in 6-class macro-F1;", flush=True)
    print("  the question is whether the grid reaches L2's per-pair optimum.", flush=True)
