"""Generate the GAF-CNN orthogonal hedge submission: robust thresholds + gaf-L2 injection
(nested w). Reaches OOF 0.7873 (= the 0.8200 config) but via DIFFERENT predictions
(orient/gaf rescue zero-overlap L2 sets) -> a genuinely diversified final pick for the
private split. Run: .venv\\Scripts\\python.exe scripts\\gen_gaf_hedge.py
"""
from __future__ import annotations
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]; OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
N = 6; ALPHA = 0.842
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
meta = pd.read_parquet(DATA / "meta_train.parquet"); y = meta["label"].values.astype(int); g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
gf, gft = load("gaf_cnn_v1_oof.npy"), load("gaf_cnn_v1_test_probs.npy")
gkf = GroupKFold(5); norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso_pair(o, t):
    cal = np.zeros_like(o)
    for tr, va in gkf.split(o, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(o[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(o[va, c])
    ct = np.zeros_like(t)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(o[:, c], (y == c).astype(float)); ct[:, c] = ir.predict(t[:, c])
    return norm(cal), norm(ct)


cal, calt = iso_pair(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
gcal, gcalt = iso_pair(norm(gf), norm(gft))
WG = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
inj = lambda c_, s, w: norm(np.where(np.arange(N) == 2, (1 - w) * c_ + w * s, c_)) if False else None


def injc(c_, s, w):
    o = c_.copy(); o[:, 2] = (1 - w) * c_[:, 2] + w * s[:, 2]; return norm(o)


cals = {w: injc(cal, gcal, w) for w in WG}; nested = np.zeros(len(y), int); ch = []
for tr, te in gkf.split(cal, groups=g):
    bw, bf = 0.0, -1
    for w in WG:
        f = f1_score(y[tr], (cals[w][tr] * np.exp(ROBUST)).argmax(1), average="macro")
        if f > bf: bf, bw = f, w
    ch.append(bw); nested[te] = (cals[bw][te] * np.exp(ROBUST)).argmax(1)
wf = float(np.median(ch))
print(f"gaf robust+L2 nested OOF={f1_score(y, nested, average='macro'):.4f}  w={wf:.2f}", flush=True)
preds = (injc(calt, gcalt, wf) * np.exp(ROBUST)).argmax(1)
# diff vs the proven 0.8200 (orient robust w=0.15)
ocal, ocalt = iso_pair(norm(load("orient_lgbm_oof.npy")), norm(load("orient_lgbm_test_probs.npy")))
o200 = (injc(calt, ocalt, 0.15) * np.exp(ROBUST)).argmax(1)
out = SUB / f"sub_robust_gaf_inject_w{int(wf*100):02d}.csv"
pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(out, index=False)
print(f"wrote {out.name}  ({int((preds != o200).sum())} of {len(preds)} rows differ from the 0.8200 sub "
      f"-> genuinely diversified hedge)", flush=True)
