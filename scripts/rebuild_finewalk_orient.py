"""Rebuild the Fine_walk (binary L1-vs-L2) stage WITH orientation features —
matching the production recipe (multi-seed LightGBM + XGBoost on the rich
engineered stack) so the base rebuild is competitive, then test if orientation
pushes it ABOVE production.

Method: fresh multi-seed (LGBM x3 + XGB x3) binary L1-vs-L2 -> P(L2|walk) q.
Surgically swap q into the FROZEN production P2 (reuse coarse group masses):
   P2'[L2] = (P2[L1]+P2[L2])*q ;  P2'[L1] = *(1-q)
Blend with P1, per-fold isotonic, apply FROZEN PEAK_LOGW (no threshold re-tuning).
Controlled A/B: q_base (rich stack) vs q_orient (rich stack + orientation).

Run: python scripts/rebuild_finewalk_orient.py
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.train_cnn_bilstm import build_or_load_seq_cache
from scripts.orient_pseudogyro_model import build_orient

N = 6
ALPHA = 0.842
LGB_SEEDS = [42, 7, 23]
XGB_SEEDS = [42, 7, 23]
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])
# rich engineered stack (exclude covariance: documented harmful, row 14)
FAMILIES = ["none", "fft", "autocorr", "subwindow", "gravity", "jerk",
            "crossaxis", "zerocross", "magnitude", "basic_stats", "quality"]
DROP = ["file_id", "label", "user_id", "path", "index"]

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def load_rich():
    def _one(split):
        dfs = []
        for fam in FAMILIES:
            p = ROOT / "data" / f"feat_{split}_{fam}.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                d = d.drop(columns=[c for c in DROP if c in d.columns], errors="ignore")
                dfs.append(d.add_prefix(f"{fam}_").reset_index(drop=True))
        pc = ROOT / "data" / f"feat_catch22_{split}.parquet"
        if pc.exists():
            d = pd.read_parquet(pc)
            d = d.drop(columns=[c for c in DROP if c in d.columns], errors="ignore")
            dfs.append(d.add_prefix("c22_").reset_index(drop=True))
        return pd.concat(dfs, axis=1)
    return _one("train"), _one("test")


LGB_P = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
             num_leaves=63, feature_fraction=0.9, bagging_fraction=0.9,
             bagging_freq=5, min_data_in_leaf=20, verbose=-1, num_threads=16)


def _fit_predict(Xtw, yb, w, Xpred):
    preds = []
    ds = lgb.Dataset(Xtw, label=yb, weight=w)
    for s in LGB_SEEDS:
        pr = dict(LGB_P); pr["seed"] = s
        preds.append(lgb.train(pr, ds, num_boost_round=400).predict(Xpred))
    dtr = xgb.DMatrix(Xtw.values, label=yb, weight=w); dpr = xgb.DMatrix(Xpred.values)
    for s in XGB_SEEDS:
        xp = dict(objective="binary:logistic", eta=0.05, max_depth=6, subsample=0.9,
                  colsample_bytree=0.9, seed=s, verbosity=0)
        preds.append(xgb.train(xp, dtr, num_boost_round=400).predict(dpr))
    return np.mean(preds, axis=0)


def binary_oof_q(Xdf, Xte_df, tag):
    wm = (y == 1) | (y == 2)
    q = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(Xdf, y, groups):
        tw = tr[wm[tr]]; yb = (y[tw] == 2).astype(int)
        cnt = np.bincount(yb, minlength=2).astype(float); cnt[cnt == 0] = 1
        q[va] = _fit_predict(Xdf.iloc[tw], yb, (1.0 / cnt)[yb], Xdf.iloc[va])
    f1b = f1_score((y[wm] == 2).astype(int), (q[wm] > 0.5).astype(int), average="macro")
    tw = np.where(wm)[0]; yb = (y[tw] == 2).astype(int)
    cnt = np.bincount(yb, minlength=2).astype(float); cnt[cnt == 0] = 1
    qt = _fit_predict(Xdf.iloc[tw], yb, (1.0 / cnt)[yb], Xte_df)
    print(f"  [{tag}] binary L1vL2 OOF macro-F1={f1b:.4f}", flush=True)
    return q, qt


def recompose(P2, q):
    P = P2.copy(); wmass = P[:, 1] + P[:, 2]
    P[:, 2] = wmass * q; P[:, 1] = wmass * (1.0 - q)
    return norm(P)


def eval_blend(P2oof, P2test, tag):
    bo = norm(ALPHA * p1 + (1 - ALPHA) * P2oof); bt = norm(ALPHA * p1t + (1 - ALPHA) * P2test)
    cal = np.zeros_like(bo); calt = np.zeros_like(bt)
    for tr, va in GroupKFold(5).split(bo, groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(bo[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(bo[va, c])
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(bo[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(bt[:, c])
    cal, calt = norm(cal), norm(calt)
    pred = (cal * np.exp(PEAK_LOGW)).argmax(1)
    f1 = f1_score(y, pred, average="macro"); pc = f1_score(y, pred, average=None, labels=list(range(N)))
    print(f"  {tag:32s} macro={f1:.4f}  L2={pc[2]:.3f} L3={pc[3]:.3f} L5={pc[5]:.3f}", flush=True)
    return f1, calt


def main():
    Xtr, ytr, Xte, _ = build_or_load_seq_cache()
    print("loading rich feature stack ...", flush=True)
    base_tr, base_te = load_rich()
    print(f"  rich stack: {base_tr.shape[1]} cols", flush=True)
    print("building orientation features ...", flush=True)
    o_tr, o_te = build_orient(Xtr), build_orient(Xte)
    Xo_tr = pd.concat([base_tr, o_tr.add_prefix("orient_")], axis=1)
    Xo_te = pd.concat([base_te, o_te.add_prefix("orient_")], axis=1)

    print("\nfresh multi-seed (LGBMx3+XGBx3) Fine_walk:", flush=True)
    q_base, qt_base = binary_oof_q(base_tr, base_te, "base rich")
    q_orient, qt_orient = binary_oof_q(Xo_tr, Xo_te, "base+orient")

    print("\nfull-pipeline macro-F1 @ FROZEN thresholds:", flush=True)
    base_f1, _ = eval_blend(p2, p2t, "production (frozen P2)")
    pb_f1, _ = eval_blend(recompose(p2, q_base), recompose(p2t, qt_base), "P2' rebuilt Fine_walk (base)")
    po_f1, po_calt = eval_blend(recompose(p2, q_orient), recompose(p2t, qt_orient), "P2' rebuilt Fine_walk (+orient)")

    print(f"\n  orientation effect on rebuilt Fine_walk: {po_f1 - pb_f1:+.4f}", flush=True)
    print(f"  P2_orient vs production: {po_f1 - base_f1:+.4f}", flush=True)
    if po_f1 > base_f1 + 1e-4:
        preds = (po_calt * np.exp(PEAK_LOGW)).argmax(1)
        sub = ROOT / "submissions" / "sub_finewalk_orient.csv"
        pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
        print(f"  *** BEATS PRODUCTION (frozen thresholds) -> wrote {sub} ***", flush=True)
    else:
        print("  Rebuilt Fine_walk (+orient) does not beat frozen production.", flush=True)


if __name__ == "__main__":
    main()
