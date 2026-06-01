"""Stronger complementary L2 source: 3-seed 6-class LGBM on the RICH engineered
stack + orientation features. Then gated L2-injection into frozen production at a
NESTED-CV-chosen w (frozen thresholds). Generates a candidate submission.

Discipline: we improve the SOURCE (real, nested-validated signal). We do NOT tune
w to the public leaderboard — w is chosen by nested GroupKFold on disjoint users.

Run: python scripts/orient_strong_source.py
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
from scripts.orient_pseudogyro_model import build_orient

N = 6
ALPHA = 0.842
SEEDS = [42, 7, 23]
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])
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


def load_rich(split):
    dfs = []
    for fam in FAMILIES:
        p = ROOT / "data" / f"feat_{split}_{fam}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d = d.drop(columns=[c for c in DROP if c in d.columns], errors="ignore")
            dfs.append(d.add_prefix(f"{fam}_").reset_index(drop=True))
    pc = ROOT / "data" / f"feat_catch22_{split}.parquet"
    if pc.exists():
        d = pd.read_parquet(pc); d = d.drop(columns=[c for c in DROP if c in d.columns], errors="ignore")
        dfs.append(d.add_prefix("c22_").reset_index(drop=True))
    return pd.concat(dfs, axis=1)


def cw(yv):
    cnt = np.bincount(yv, minlength=N).astype(float); cnt[cnt == 0] = 1
    return (len(yv) / (N * cnt))[yv]


LGB_P = dict(objective="multiclass", num_class=N, metric="multi_logloss",
             learning_rate=0.05, num_leaves=63, feature_fraction=0.6,
             bagging_fraction=0.9, bagging_freq=5, min_data_in_leaf=20,
             verbose=-1, num_threads=16)


def build_source():
    Xtr, ytr, Xte, _ = build_or_load_seq_cache()
    base_tr, base_te = load_rich("train"), load_rich("test")
    o_tr, o_te = build_orient(Xtr), build_orient(Xte)
    Xtr_df = pd.concat([base_tr, o_tr.add_prefix("orient_")], axis=1)
    Xte_df = pd.concat([base_te, o_te.add_prefix("orient_")], axis=1)
    print(f"strong source features: {Xtr_df.shape[1]}", flush=True)
    oof = np.zeros((len(y), N)); test = np.zeros((len(Xte_df), N))
    for k, (tr, va) in enumerate(GroupKFold(5).split(Xtr_df, y, groups)):
        for s in SEEDS:
            pr = dict(LGB_P); pr["seed"] = s
            m = lgb.train(pr, lgb.Dataset(Xtr_df.iloc[tr], label=y[tr], weight=cw(y[tr])), num_boost_round=500)
            oof[va] += m.predict(Xtr_df.iloc[va]) / len(SEEDS)
            test += m.predict(Xte_df) / (5 * len(SEEDS))
    f1 = f1_score(y, oof.argmax(1), average="macro")
    pc = f1_score(y, oof.argmax(1), average=None, labels=list(range(N)))
    wm = (y == 1) | (y == 2)
    pb = oof[wm][:, [1, 2]]; pb = pb / np.clip(pb.sum(1, keepdims=True), 1e-12, None)
    print(f"strong source OOF macro={f1:.4f}  L2={pc[2]:.3f}  L1vL2sep="
          f"{f1_score((y[wm] == 2).astype(int), pb.argmax(1), average='macro'):.4f}", flush=True)
    np.save(ROOT / "oof" / "orient_strong_oof.npy", oof.astype(np.float32))
    np.save(ROOT / "oof" / "orient_strong_test_probs.npy", test.astype(np.float32))
    return oof, test


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
    src_oof, src_test = build_source()
    cal_prod, cal_prod_t = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
    dgc, dgc_t = iso(norm(src_oof), norm(src_test))
    base = f1_score(y, (cal_prod * np.exp(PEAK_LOGW)).argmax(1), average="macro")
    print(f"\nproduction baseline @ frozen = {base:.4f}", flush=True)
    cals = {}
    for w in WGRID:
        cals[w] = inject(cal_prod, dgc, w)
        pc = f1_score(y, (cals[w] * np.exp(PEAK_LOGW)).argmax(1), average=None, labels=list(range(N)))
        print(f"  w={w:.2f}  macro={f1_score(y, (cals[w]*np.exp(PEAK_LOGW)).argmax(1), average='macro'):.4f}  L2={pc[2]:.3f}", flush=True)
    # nested CV pick w
    folds = list(GroupKFold(5).split(cal_prod, groups=groups))
    nested = np.zeros(len(y), int); chosen = []
    for tr, te in folds:
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(PEAK_LOGW)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        chosen.append(bw); nested[te] = (cals[bw][te] * np.exp(PEAK_LOGW)).argmax(1)
    nf = f1_score(y, nested, average="macro")
    print(f"\n  nested chosen w: {chosen}  nested macro={nf:.4f}  delta {nf - base:+.4f}", flush=True)
    wfin = float(np.median(chosen))
    calt = inject(cal_prod_t, dgc_t, wfin)
    preds = (calt * np.exp(PEAK_LOGW)).argmax(1)
    sub = ROOT / "submissions" / f"sub_orient_strong_gated_w{int(wfin*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
    print(f"  wrote {sub} (w={wfin}); differs-from-prod check downstream.", flush=True)


if __name__ == "__main__":
    main()
