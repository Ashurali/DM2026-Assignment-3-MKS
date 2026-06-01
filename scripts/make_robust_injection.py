"""Pick-2 hedge candidate: ROBUST thresholds + the validated orient L2-injection.

Combines the conservative robust thresholds (less L1 suppression -> safer on
private) with the nested-validated orientation L2-injection (the real gain behind
0.8184). Nested-CV picks w under the ROBUST thresholds (not tuned to public).

Run: python scripts/make_robust_injection.py
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
N = 6
ALPHA = 0.842
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
PEAK = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.19999999999999996, 0.9000000000000004,
                   0.4628951701768874, -0.239947242877496, -0.42948082285098554])

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


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


cal, calt = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
dgc, dgc_t = iso(norm(og), norm(ogt))

for tag, lw in [("ROBUST", ROBUST), ("PEAK", PEAK)]:
    plain = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
    cals = {w: inject(cal, dgc, w) for w in WGRID}
    # nested CV pick w under THIS threshold config
    nested = np.zeros(len(y), int); chosen = []
    for tr, te in GroupKFold(5).split(cal, groups=groups):
        bw, bf = 0.0, -1
        for w in WGRID:
            f = f1_score(y[tr], (cals[w][tr] * np.exp(lw)).argmax(1), average="macro")
            if f > bf:
                bf, bw = f, w
        chosen.append(bw); nested[te] = (cals[bw][te] * np.exp(lw)).argmax(1)
    nf = f1_score(y, nested, average="macro")
    print(f"{tag}: plain={plain:.4f}  nested+inject={nf:.4f}  delta {nf - plain:+.4f}  chosen_w={chosen}", flush=True)

# Write the ROBUST + injection candidate (nested-median w under robust)
cals = {w: inject(cal, dgc, w) for w in WGRID}
chosen = []
for tr, te in GroupKFold(5).split(cal, groups=groups):
    bw, bf = 0.0, -1
    for w in WGRID:
        f = f1_score(y[tr], (cals[w][tr] * np.exp(ROBUST)).argmax(1), average="macro")
        if f > bf:
            bf, bw = f, w
    chosen.append(bw)
wfin = float(np.median(chosen))
preds = (inject(calt, dgc_t, wfin) * np.exp(ROBUST)).argmax(1)
sub = ROOT / "submissions" / f"sub_robust_orient_inject_w{int(wfin*100):02d}.csv"
pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
print(f"\nwrote {sub} (robust thresholds + orient injection, w={wfin})", flush=True)
