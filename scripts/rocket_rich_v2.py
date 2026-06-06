"""ROCKET-rich v2: MultiRocket (SOTA, far stronger than MiniRocket) + canonical
RidgeClassifierCV on the same 14 rich channels. v1 (MiniRocket+logistic) hit the best
single-source L1vL2 sep (0.661) but its per-class AUC trailed production -- the logistic
was the weak link. MultiRocket extracts more pooling operators (richer X); RidgeCV is the
classifier ROCKET was designed for. Goal: lift per-class AUC into complementary territory.

Server CPU (num_kernels capped for shared-box memory). Probs via softmax(decision_function).
Run: ~/anaconda3/envs/dm2026-a3/bin/python scripts/rocket_rich_v2.py
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
N = 6
t0 = time.time()
seqtr = np.load(ROOT / "data" / "seq_train.npy"); seqte = np.load(ROOT / "data" / "seq_test.npy")
y = np.load(ROOT / "data" / "seq_y_train.npy").astype(int)
meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
assert len(y) == len(meta) and (y == meta["label"].values.astype(int)).all()
g = meta["user_id"].values
test_ids = np.load(ROOT / "data" / "seq_test_ids.npy").astype(int)


def build_channels(seq):
    mx, my, mz, sx, sy, sz = [seq[:, i, :].astype(np.float64) for i in range(6)]
    eps = 1e-8
    mag_m = np.sqrt(mx**2 + my**2 + mz**2); mag_s = np.sqrt(sx**2 + sy**2 + sz**2)
    pitch = np.arctan2(mx, np.sqrt(my**2 + mz**2) + eps); roll = np.arctan2(my, mz + eps)
    incl = np.arctan2(np.sqrt(mx**2 + my**2), mz + eps)

    def d(a):
        dd = np.diff(a, axis=1); return np.concatenate([dd[:, :1], dd], axis=1)
    jerk = np.sqrt(d(mx)**2 + d(my)**2 + d(mz)**2)
    return np.stack([mx, my, mz, sx, sy, sz, mag_m, mag_s, pitch, roll, incl, jerk,
                     d(pitch), d(roll)], axis=1).astype(np.float32)


Xtr = build_channels(seqtr); Xte = build_channels(seqte)
print(f"rich channels {Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)
from sktime.transformations.panel.rocket import MultiRocketMultivariate as RK
rk = RK(num_kernels=3000, random_state=42, n_jobs=8)
rk.fit(Xtr)
Ftr = np.nan_to_num(np.asarray(rk.transform(Xtr), dtype=np.float32))
Fte = np.nan_to_num(np.asarray(rk.transform(Xte), dtype=np.float32))
print(f"MultiRocket features {Ftr.shape} ({time.time()-t0:.0f}s)", flush=True)
sc = StandardScaler().fit(Ftr); Ftr = sc.transform(Ftr).astype(np.float32); Fte = sc.transform(Fte).astype(np.float32)

ALPHAS = np.logspace(-3, 3, 13)
oof = np.zeros((len(y), N))
for tr, va in GroupKFold(5).split(Ftr, groups=g):
    clf = RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced").fit(Ftr[tr], y[tr])
    d = clf.decision_function(Ftr[va])
    oof[va] = softmax(d, axis=1) if d.ndim == 2 else np.eye(N)[clf.predict(Ftr[va])]
m = f1_score(y, oof.argmax(1), average="macro")
per = f1_score(y, oof.argmax(1), average=None, labels=list(range(N)))
wm = (y == 1) | (y == 2); pb = oof[wm][:, [1, 2]]; pb = pb / pb.sum(1, keepdims=True)
sep = f1_score((y[wm] == 2).astype(int), pb.argmax(1), average="macro")
aucs = " ".join(f"L{c}:{roc_auc_score((y==c).astype(int), oof[:,c]):.3f}" for c in range(N))
print(f"=== MultiRocket-rich: OOF macro={m:.4f}  L1vL2sep={sep:.4f} ===", flush=True)
print(f"  per-class={[round(float(p),3) for p in per]}", flush=True)
print(f"  one-vs-rest AUC: {aucs}  (prod L2 AUC 0.898; v1 rocket L2 0.862)", flush=True)
clf = RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced").fit(Ftr, y)
dt = clf.decision_function(Fte)
test = softmax(dt, axis=1) if dt.ndim == 2 else np.eye(N)[clf.predict(Fte)]
np.save(ROOT / "oof" / "rocket_v2_oof.npy", oof.astype(np.float32))
np.save(ROOT / "oof" / "rocket_v2_test_probs.npy", test.astype(np.float32))
print(f"saved rocket_v2_oof + test_probs ({time.time()-t0:.0f}s)", flush=True)
