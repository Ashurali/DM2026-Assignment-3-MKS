"""Given the private LB rewards ROBUSTNESS (public-overfitters crash), build lower-
variance variants of the 0.8200 submission as selectable finals:
  - robust + single-seed orient (= the proven 0.8200)             [sanity]
  - robust + MULTI-SEED orient (orient_lgbm_ms)                   [variance-reduced]
  - robust + AVERAGE(single, multi-seed) source                  [most stable source]
Same discipline: per-class isotonic, FROZEN robust thresholds, nested-CV w (never
public-tuned). Reports nested OOF + how many test rows differ from the proven 0.8200,
and writes CSVs. Also writes the peak counterparts for completeness.

Run: .venv\\Scripts\\python.exe scripts\\make_robust_variants.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "oof"; DATA = ROOT / "data"; SUB = ROOT / "submissions"
N = 6; ALPHA = 0.842
PEAK = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.19999999999999996, 0.9000000000000004,
                   0.4628951701768874, -0.239947242877496, -0.42948082285098554])
WGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int); g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")
ogm, ogmt = load("orient_lgbm_ms_oof.npy"), load("orient_lgbm_ms_test_probs.npy")
gkf = GroupKFold(5)


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso_pair(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in gkf.split(oof_raw, groups=g):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


cal, calt = iso_pair(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
sources = {
    "single": iso_pair(norm(og), norm(ogt)),
    "ms": iso_pair(norm(ogm), norm(ogmt)),
    "avg": iso_pair(norm((og + ogm) / 2), norm((ogt + ogmt) / 2)),
}


def inj(c_, src, w):
    out = c_.copy(); out[:, 2] = (1 - w) * c_[:, 2] + w * src[:, 2]; return norm(out)


def nested_deploy(src_oof, lw):
    cals = {w: inj(cal, src_oof, w) for w in WGRID}
    nested = np.zeros(len(y), int); chosen = []
    for tr, te in gkf.split(cal, groups=g):
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(lw)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        chosen.append(bw); nested[te] = (cals[bw][te] * np.exp(lw)).argmax(1)
    return f1_score(y, nested, average="macro"), float(np.median(chosen))


# reference: the proven 0.8200 deployment (robust, single, w=0.15)
ref_pred = (inj(calt, sources["single"][1], 0.15) * np.exp(ROBUST)).argmax(1)

for tag, lw in [("robust", ROBUST), ("peak", PEAK)]:
    base = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    print(f"\n=== {tag.upper()} (baseline no-inj OOF {base:.4f}) ===", flush=True)
    for sname, (src_oof, src_t) in sources.items():
        nf, wf = nested_deploy(src_oof, lw)
        preds = (inj(calt, src_t, wf) * np.exp(lw)).argmax(1)
        out = SUB / f"sub_{tag}_orient_{sname}_inject_w{int(wf*100):02d}.csv"
        pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(out, index=False)
        d = int((preds != ref_pred).sum())
        print(f"  {sname:6s}: nested OOF={nf:.4f} (+{nf-base:.4f})  w={wf:.2f}  "
              f"rows-differ-vs-0.8200={d}  -> {out.name}", flush=True)

print("\nNOTE: 'single'+robust+w15 == the proven 0.8200. 'ms'/'avg' are variance-reduced", flush=True)
print("variants for the PRIVATE final (lower seed variance). Submit, then pick 2 finals", flush=True)
print("leaning robust -- private rewards conservatism (public #1 -> private #9).", flush=True)
