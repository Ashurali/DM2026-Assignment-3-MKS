"""Multi-seed version of the WINNING orient L2 source (none+catch22 + orientation,
3-seed 6-class LGBM). The single-seed version (orient_lgbm) scored 0.8184 via
gated L2-injection; multi-seed reduces single-seed variance -> a more ROBUST
version of the same recipe for the private split. Inject at nested-CV-chosen w.

Run: python scripts/orient_ms_winner.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.train_cnn_bilstm import build_or_load_seq_cache
from scripts.orient_pseudogyro_model import build_orient, load_base

N = 6
ALPHA = 0.842
SEEDS = [42, 7, 23]
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def cw(yv):
    cnt = np.bincount(yv, minlength=N).astype(float); cnt[cnt == 0] = 1
    return (len(yv) / (N * cnt))[yv]


LGB_P = dict(objective="multiclass", num_class=N, metric="multi_logloss",
             learning_rate=0.05, num_leaves=63, feature_fraction=0.9,
             bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
             verbose=-1, num_threads=16)


def iso(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in GroupKFold(5).split(oof_raw, groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


def inject(cal, src, w):
    out = cal.copy(); out[:, 2] = (1 - w) * cal[:, 2] + w * src[:, 2]
    return norm(out)


def main():
    Xtr, ytr, Xte, _ = build_or_load_seq_cache()
    base_tr, base_te = load_base()
    o_tr, o_te = build_orient(Xtr), build_orient(Xte)
    Xtr_df = pd.concat([base_tr, o_tr], axis=1)
    Xte_df = pd.concat([base_te, o_te], axis=1)
    print(f"winning recipe features: {Xtr_df.shape[1]} (none+catch22+orient), {len(SEEDS)} seeds", flush=True)

    oof = np.zeros((len(y), N)); test = np.zeros((len(Xte_df), N))
    for tr, va in GroupKFold(5).split(Xtr_df, y, groups):
        for s in SEEDS:
            pr = dict(LGB_P); pr["seed"] = s
            m = lgb.train(pr, lgb.Dataset(Xtr_df.iloc[tr], label=y[tr], weight=cw(y[tr])), num_boost_round=600)
            oof[va] += m.predict(Xtr_df.iloc[va]) / len(SEEDS)
            test += m.predict(Xte_df) / (5 * len(SEEDS))
    pc = f1_score(y, oof.argmax(1), average=None, labels=list(range(N)))
    print(f"orient_lgbm_ms OOF macro={f1_score(y, oof.argmax(1), average='macro'):.4f}  L2={pc[2]:.3f}", flush=True)
    np.save(ROOT / "oof" / "orient_lgbm_ms_oof.npy", oof.astype(np.float32))
    np.save(ROOT / "oof" / "orient_lgbm_ms_test_probs.npy", test.astype(np.float32))

    cal_prod, cal_prod_t = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
    dgc, dgc_t = iso(norm(oof), norm(test))
    base = f1_score(y, (cal_prod * np.exp(PEAK_LOGW)).argmax(1), average="macro")
    print(f"\nproduction baseline @ frozen = {base:.4f}", flush=True)
    cals = {w: inject(cal_prod, dgc, w) for w in WGRID}
    for w in WGRID:
        print(f"  w={w:.2f} macro={f1_score(y, (cals[w]*np.exp(PEAK_LOGW)).argmax(1), average='macro'):.4f}", flush=True)
    nested = np.zeros(len(y), int); chosen = []
    for tr, te in GroupKFold(5).split(cal_prod, groups=groups):
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(PEAK_LOGW)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        chosen.append(bw); nested[te] = (cals[bw][te] * np.exp(PEAK_LOGW)).argmax(1)
    nf = f1_score(y, nested, average="macro")
    print(f"  nested chosen w={chosen} nested macro={nf:.4f} delta {nf - base:+.4f}", flush=True)
    wfin = float(np.median(chosen))
    preds = (inject(cal_prod_t, dgc_t, wfin) * np.exp(PEAK_LOGW)).argmax(1)
    sub = ROOT / "submissions" / f"sub_orient_lgbm_ms_gated_w{int(wfin*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
    print(f"  wrote {sub} (w={wfin})", flush=True)


if __name__ == "__main__":
    main()
