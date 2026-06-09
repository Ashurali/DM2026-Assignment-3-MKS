"""THE key test: do orient (summary-stat dynamics) and GAF-CNN (2D image view) rescue
DIFFERENT L2 samples -> do they STACK past 0.8200? Both give +0.0009 robust-L2 alone;
if orthogonal, joint ~ +0.0018. If they fight the same samples, joint ~ +0.0009 (wall).

Joint nested (w_orient, w_gaf) injection into L2 under ROBUST + PEAK, vs each alone,
plus rescue-set overlap. Writes a submission if the joint clearly beats either alone.
Run: .venv\\Scripts\\python.exe scripts\\stack_orient_gaf.py
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]; OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.30, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
meta = pd.read_parquet(DATA / "meta_train.parquet"); y = meta["label"].values.astype(int); g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")
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
ocal, ocalt = iso_pair(norm(og), norm(ogt)); gcal, gcalt = iso_pair(norm(gf), norm(gft))

# rescue-set overlap (do they rescue different L2s?)
prod = (cal * np.exp(ROBUST)).argmax(1)
miss = (y == 2) & (prod != 2)
o_resc = miss & (og.argmax(1) == 2); g_resc = miss & (gf.argmax(1) == 2)
inter = int((o_resc & g_resc).sum()); uni = int((o_resc | g_resc).sum())
print(f"L2 rescue sets: orient={int(o_resc.sum())} gaf={int(g_resc.sum())} "
      f"overlap={inter} union={uni}  (low overlap => orthogonal => should stack)", flush=True)

WG = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]


def inj2(c_, w_o, w_g):
    o = c_.copy(); o[:, 2] = (1 - w_o - w_g) * c_[:, 2] + w_o * ocal[:, 2] + w_g * gcal[:, 2]
    return norm(o)


def injt(c_, w_o, w_g):
    o = c_.copy(); o[:, 2] = (1 - w_o - w_g) * c_[:, 2] + w_o * ocalt[:, 2] + w_g * gcalt[:, 2]
    return norm(o)


for tag, lw in [("ROBUST", ROBUST), ("PEAK", PEAK)]:
    base = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    grid = {(wo, wg): inj2(cal, wo, wg) for wo in WG for wg in WG if wo + wg <= 0.35}
    nested = np.zeros(len(y), int); ch = []
    for tr, te in gkf.split(cal, groups=g):
        bp, bf = (0.0, 0.0), -1
        for k, c_ in grid.items():
            f = f1_score(y[tr], (c_[tr] * np.exp(lw)).argmax(1), average="macro")
            if f > bf: bf, bp = f, k
        ch.append(bp); nested[te] = (grid[bp][te] * np.exp(lw)).argmax(1)
    nf = f1_score(y, nested, average="macro")
    # orient-only and gaf-only references (1-D nested)
    def one(src_idx):
        gg = {w: inj2(cal, w, 0.0) if src_idx == 0 else inj2(cal, 0.0, w) for w in WG}
        o = np.zeros(len(y), int)
        for tr, te in gkf.split(cal, groups=g):
            bw, bf = 0.0, -1
            for w in WG:
                f = f1_score(y[tr], (gg[w][tr] * np.exp(lw)).argmax(1), average="macro")
                if f > bf: bf, bw = f, w
            o[te] = (gg[bw][te] * np.exp(lw)).argmax(1)
        return f1_score(y, o, average="macro")
    fo, fgf = one(0), one(1)
    wo_f, wg_f = float(np.median([c[0] for c in ch])), float(np.median([c[1] for c in ch]))
    print(f"\n{tag}: base={base:.4f} | orient-only={fo:.4f} | gaf-only={fgf:.4f} | "
          f"JOINT={nf:.4f}  (joint vs best-single {nf-max(fo,fgf):+.4f})  w=(o{wo_f:.2f},g{wg_f:.2f})", flush=True)
    if nf - max(fo, fgf) > 0.0004:
        preds = (injt(calt, wo_f, wg_f) * np.exp(lw)).argmax(1)
        out = SUB / f"sub_{tag.lower()}_orient_gaf_stack_w{int(wo_f*100):02d}_{int(wg_f*100):02d}.csv"
        pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(out, index=False)
        print(f"   *** STACKS! wrote {out.name} ***", flush=True)
print("\n(if JOINT clearly > best single, orient+gaf break the stacking wall -> submit it)", flush=True)
