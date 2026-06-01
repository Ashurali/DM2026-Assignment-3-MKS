"""Pseudo-gyroscope / orientation-dynamics features + A/B LightGBM.

Dataset insight (Bruno et al. 2014, wrist-worn ADL): gravity is NOT removed and
the sensor is on the wrist, so the per-second mean_x/y/z trace the WRIST
ORIENTATION trajectory. Its time-derivative approximates the gyroscope signal the
HAR literature says is needed to separate confusable activities — which we can't
add as a sensor but CAN reconstruct from the gravity trajectory.

We currently use orientation only STATICALLY (window-mean angles) + coarse 5-chunk
drift. Here we build the full 300-step orientation DYNAMICS (angular speed,
cumulative rotation, reversals, tilt trajectory shape) + intensity dynamics from
the std channels. Convention-invariant features (angles/norms/correlations) so
they're robust without knowing the exact axis frame.

A/B test: LightGBM (GroupKFold-5 by user, class-weighted) on
  A = base catalog (none[271] + catch22[132])
  B = base + orientation-dynamics
Compare OOF macro-F1, per-class L2/L3/L5, and L1<->L2 binary separability.
Saves B's OOF + test probs as oof/orient_lgbm_* for frozen-threshold integration.

Run (server, dm2026-a3):  python scripts/orient_pseudogyro_model.py
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.train_cnn_bilstm import build_or_load_seq_cache

N = 6
SEED = 42
EPS = 1e-8


def _ac(x, lag):
    if len(x) <= lag:
        return 0.0
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + EPS
    return float((a * b).sum() / d)


def _npeaks(x):
    if len(x) < 3:
        return 0
    m = x.mean()
    return int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > m)))


def _safecorr(a, b):
    if len(a) < 3 or np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def orient_feats_one(g, s):
    """g=(3,300) mean/orientation channels, s=(3,300) std/intensity channels."""
    f = {}
    gn = np.linalg.norm(g, axis=0) + EPS                       # (300,)
    gu = g / gn                                                # unit vectors
    # 1) consecutive angular change (pseudo angular speed)
    cosd = np.clip(np.sum(gu[:, 1:] * gu[:, :-1], axis=0), -1, 1)
    dth = np.arccos(cosd)                                      # (299,)
    f["o_dth_mean"], f["o_dth_std"] = dth.mean(), dth.std()
    f["o_dth_max"], f["o_dth_sum"] = dth.max(), dth.sum()
    f["o_dth_med"], f["o_dth_q90"] = np.median(dth), np.quantile(dth, 0.9)
    f["o_dth_fracbig"] = np.mean(dth > 0.05)
    # 2) tilt vs window-mean gravity (mounting-invariant orientation)
    gbar = g.mean(axis=1); gbar = gbar / (np.linalg.norm(gbar) + EPS)
    th = np.arccos(np.clip(gu.T @ gbar, -1, 1))               # (300,)
    f["o_th_mean"], f["o_th_std"] = th.mean(), th.std()
    f["o_th_range"], f["o_th_max"] = th.max() - th.min(), th.max()
    thc = th - th.mean()
    f["o_th_cross"] = int(np.sum(np.diff(np.sign(thc)) != 0))
    f["o_th_ac1"], f["o_th_ac5"] = _ac(th, 1), _ac(th, 5)
    f["o_th_npeaks"] = _npeaks(th)
    # 3) angular jerk
    ddth = np.diff(dth)
    f["o_jerk_mae"], f["o_jerk_std"] = np.mean(np.abs(ddth)), ddth.std()
    # 4) gravity-magnitude dynamics (dynamic acceleration bursts)
    f["o_gmag_std"], f["o_gmag_range"] = gn.std(), gn.max() - gn.min()
    f["o_gmag_npeaks"] = _npeaks(gn)
    # 5) pitch/roll (standard convention; extras)
    pitch = np.arctan2(g[0], np.sqrt(g[1] ** 2 + g[2] ** 2) + EPS)
    roll = np.arctan2(g[1], g[2])
    for nm, ang in [("pitch", pitch), ("roll", roll)]:
        f[f"o_{nm}_std"] = ang.std(); f[f"o_{nm}_range"] = ang.max() - ang.min()
        d = np.diff(ang)
        f[f"o_{nm}_nrev"] = int(np.sum(np.diff(np.sign(d)) != 0))
        f[f"o_{nm}_absvel"] = np.mean(np.abs(d))
    # 6) intensity (std-channel) dynamics
    In = np.linalg.norm(s, axis=0)                            # (300,)
    f["i_mean"], f["i_std"], f["i_max"] = In.mean(), In.std(), In.max()
    f["i_sum"], f["i_cov"] = In.sum(), In.std() / (In.mean() + EPS)
    f["i_npeaks"], f["i_ac1"] = _npeaks(In), _ac(In, 1)
    # 7) coupling: does the wrist rotate when it moves?
    f["c_dth_intens"] = _safecorr(dth, In[1:])
    return f


def build_orient(X):
    rows = [orient_feats_one(X[i, 0:3, :].astype(np.float64), X[i, 3:6, :].astype(np.float64))
            for i in range(len(X))]
    return pd.DataFrame(rows).astype(np.float32)


def load_base():
    def _load(split):
        dfs = []
        for nm in ["feat_%s_none.parquet" % split, "feat_catch22_%s.parquet" % split]:
            p = ROOT / "data" / nm
            if p.exists():
                d = pd.read_parquet(p)
                d = d.drop(columns=[c for c in ["file_id", "label", "user_id", "path", "index"] if c in d.columns])
                dfs.append(d.reset_index(drop=True))
        return pd.concat(dfs, axis=1)
    # catch22 split naming differs (train/test)
    return _load("train"), _load("test")


def class_weights(y):
    cnt = np.bincount(y, minlength=N).astype(float); cnt[cnt == 0] = 1
    return (len(y) / (N * cnt))[y]


def cv_oof(Xdf, y, groups, Xte_df, tag):
    oof = np.zeros((len(y), N)); test = np.zeros((len(Xte_df), N))
    params = dict(objective="multiclass", num_class=N, metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
                  verbose=-1, seed=SEED, num_threads=16)
    for k, (tr, va) in enumerate(GroupKFold(5).split(Xdf, y, groups)):
        ds = lgb.Dataset(Xdf.iloc[tr], label=y[tr], weight=class_weights(y[tr]))
        m = lgb.train(params, ds, num_boost_round=600)
        oof[va] = m.predict(Xdf.iloc[va])
        test += m.predict(Xte_df) / 5
    f1 = f1_score(y, oof.argmax(1), average="macro")
    pc = f1_score(y, oof.argmax(1), average=None, labels=list(range(N)))
    wm = (y == 1) | (y == 2)
    pb = oof[wm][:, [1, 2]]; pb = pb / np.clip(pb.sum(1, keepdims=True), 1e-12, None)
    l1l2 = f1_score((y[wm] == 2).astype(int), pb.argmax(1), average="macro")
    print(f"[{tag}] OOF macro={f1:.4f}  L2={pc[2]:.3f} L3={pc[3]:.3f} L5={pc[5]:.3f}  "
          f"L1vL2sep={l1l2:.4f}", flush=True)
    return oof, test, f1


def main():
    Xtr, ytr, Xte, test_ids = build_or_load_seq_cache()
    meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta["user_id"].values
    base_tr, base_te = load_base()
    print(f"base features: {base_tr.shape[1]}  train={len(ytr)}", flush=True)
    print("building orientation-dynamics features ...", flush=True)
    o_tr, o_te = build_orient(Xtr), build_orient(Xte)
    print(f"orientation features: {o_tr.shape[1]}", flush=True)

    A_tr, A_te = base_tr, base_te
    B_tr = pd.concat([base_tr, o_tr], axis=1)
    B_te = pd.concat([base_te, o_te], axis=1)

    print("\n=== A/B comparison (LightGBM, GroupKFold-5 by user) ===", flush=True)
    cv_oof(A_tr, ytr, groups, A_te, "A base")
    oofB, testB, f1B = cv_oof(B_tr, ytr, groups, B_te, "B base+orient")

    np.save(ROOT / "oof" / "orient_lgbm_oof.npy", oofB.astype(np.float32))
    np.save(ROOT / "oof" / "orient_lgbm_test_probs.npy", testB.astype(np.float32))
    print("\nSaved oof/orient_lgbm_oof.npy + _test_probs.npy", flush=True)
    # feature importance of orientation features (gain) on a full-data fit
    m = lgb.train(dict(objective="multiclass", num_class=N, metric="multi_logloss",
                       learning_rate=0.05, num_leaves=63, verbose=-1, seed=SEED, num_threads=16),
                  lgb.Dataset(B_tr, label=ytr, weight=class_weights(ytr)), num_boost_round=400)
    imp = pd.Series(m.feature_importance(importance_type="gain"), index=B_tr.columns)
    o_cols = [c for c in B_tr.columns if c.startswith(("o_", "i_", "c_"))]
    print(f"\norientation feats: {len(o_cols)} cols, total gain share="
          f"{imp[o_cols].sum() / imp.sum() * 100:.1f}%", flush=True)
    print("top-10 orientation feats by gain:", flush=True)
    print(imp[o_cols].sort_values(ascending=False).head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
