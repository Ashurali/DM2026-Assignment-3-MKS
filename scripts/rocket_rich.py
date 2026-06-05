"""BREAKING-CHANGE attempt to beat 0.8200: extract ROCKET features (random conv kernels,
SOTA time-series) from the RAW 6x300 sequence enriched with physically-derived channels
(magnitude, orientation pitch/roll/incl, jerk, orientation-velocity). ROCKET captures
temporal SHAPE that our summary stats (mean/std/fft/catch22) miss -> richer X -> lower
Bayes floor -> potential to break the ceiling, especially on the motion classes L2/L3/L5.

Runs on the SERVER (CPU, 24 cores, no GPU -> won't touch other jobs). Produces cross-user
OOF + test probs for integration into the production blend under the usual discipline.

Run: ~/anaconda3/envs/dm2026-a3/bin/python scripts/rocket_rich.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
N = 6
t0 = time.time()
seqtr = np.load(ROOT / "data" / "seq_train.npy")     # (N,6,300): mx,my,mz,sx,sy,sz
seqte = np.load(ROOT / "data" / "seq_test.npy")
y = np.load(ROOT / "data" / "seq_y_train.npy").astype(int)
meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
assert len(y) == len(meta) and (y == meta["label"].values.astype(int)).all(), "seq/meta misalign!"
g = meta["user_id"].values
test_ids = np.load(ROOT / "data" / "seq_test_ids.npy").astype(int)
print(f"seq train {seqtr.shape} test {seqte.shape}  users={len(np.unique(g))}", flush=True)


def build_channels(seq):
    mx, my, mz, sx, sy, sz = [seq[:, i, :].astype(np.float64) for i in range(6)]
    eps = 1e-8
    mag_m = np.sqrt(mx**2 + my**2 + mz**2)
    mag_s = np.sqrt(sx**2 + sy**2 + sz**2)
    pitch = np.arctan2(mx, np.sqrt(my**2 + mz**2) + eps)
    roll = np.arctan2(my, mz + eps)
    incl = np.arctan2(np.sqrt(mx**2 + my**2), mz + eps)

    def d(a):
        dd = np.diff(a, axis=1)
        return np.concatenate([dd[:, :1], dd], axis=1)
    jerk = np.sqrt(d(mx)**2 + d(my)**2 + d(mz)**2)
    chans = [mx, my, mz, sx, sy, sz, mag_m, mag_s, pitch, roll, incl, jerk, d(pitch), d(roll)]
    return np.stack(chans, axis=1).astype(np.float32)   # (N, 14, 300)


Xtr = build_channels(seqtr); Xte = build_channels(seqte)
print(f"rich channels: {Xtr.shape}  ({time.time()-t0:.0f}s)", flush=True)

# ---- ROCKET transform: MiniRocket (light ~10k feats, good shared-box citizen) ----
from sktime.transformations.panel.rocket import MiniRocketMultivariate as RK
rk = RK(num_kernels=10000, random_state=42, n_jobs=8); name = "MiniRocket"   # cap cores
print(f"using {name}", flush=True)
rk.fit(Xtr)
Ftr = np.asarray(rk.transform(Xtr), dtype=np.float32)
Fte = np.asarray(rk.transform(Xte), dtype=np.float32)
Ftr = np.nan_to_num(Ftr); Fte = np.nan_to_num(Fte)
print(f"{name} features: {Ftr.shape}  ({time.time()-t0:.0f}s)", flush=True)
sc = StandardScaler().fit(Ftr)
Ftr = sc.transform(Ftr).astype(np.float32); Fte = sc.transform(Fte).astype(np.float32)

# ---- cross-user OOF logistic (try a couple C for regularization) ----
best = None
for C in [0.1, 0.5]:
    oof = np.zeros((len(y), N))
    for tr, va in GroupKFold(5).split(Ftr, groups=g):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                 multi_class="multinomial", n_jobs=-1)
        clf.fit(Ftr[tr], y[tr]); oof[va, clf.classes_] = clf.predict_proba(Ftr[va])
    oof = oof / oof.sum(1, keepdims=True)
    m = f1_score(y, oof.argmax(1), average="macro")
    wm = (y == 1) | (y == 2); pb = oof[wm][:, [1, 2]]; pb = pb / pb.sum(1, keepdims=True)
    sep = f1_score((y[wm] == 2).astype(int), pb.argmax(1), average="macro")
    per = f1_score(y, oof.argmax(1), average=None, labels=list(range(N)))
    print(f"  C={C}: OOF macro={m:.4f}  L1vL2sep={sep:.4f}  per-class="
          f"{[round(float(p),3) for p in per]}  ({time.time()-t0:.0f}s)", flush=True)
    if best is None or m > best[0]:
        best = (m, C, oof)

m, C, oof = best
print(f"=== {name}-rich BEST: OOF macro={m:.4f} at C={C} ===", flush=True)
print("  (refs: flat-LGBM 0.733, production blend 0.788; ROCKET helps if COMPLEMENTARY)", flush=True)
clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                         multi_class="multinomial", n_jobs=-1).fit(Ftr, y)
test = np.zeros((len(test_ids), N)); test[:, clf.classes_] = clf.predict_proba(Fte)
test = test / test.sum(1, keepdims=True)
np.save(ROOT / "oof" / "rocket_rich_oof.npy", oof.astype(np.float32))
np.save(ROOT / "oof" / "rocket_rich_test_probs.npy", test.astype(np.float32))
print(f"saved oof/rocket_rich_oof.npy + _test_probs.npy  (total {time.time()-t0:.0f}s)", flush=True)
