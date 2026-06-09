"""Integrate the orientation-feature LGBM (orient_lgbm) into frozen production.

3-way blend: norm((1-b)*[0.842*P1 + 0.158*P2] + b*orient) -> isotonic -> FROZEN
PEAK_LOGW. Sweep b; NESTED GroupKFold picks b on disjoint users from eval. Write
a submission only if the nested macro-F1 beats production 0.7880. Never re-tune
thresholds (frozen top-1 config).

Run: python scripts/orient_integrate.py
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
BGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
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


base = norm(ALPHA * p1 + (1 - ALPHA) * p2)
baset = norm(ALPHA * p1t + (1 - ALPHA) * p2t)


def blended(b):
    return iso(norm((1 - b) * base + b * og), norm((1 - b) * baset + b * ogt))


cal0, _ = blended(0.0)
prod = f1_score(y, (cal0 * np.exp(PEAK_LOGW)).argmax(1), average="macro")
print(f"production baseline (b=0) @ frozen thresholds = {prod:.4f}  (must be 0.7880)\n")
print(f"{'b':>5} {'macroF1':>9} {'L2':>6} {'L3':>6} {'L5':>6}")
cals = {}
for b in BGRID:
    cal, _ = blended(b); cals[b] = cal
    pred = (cal * np.exp(PEAK_LOGW)).argmax(1)
    pc = f1_score(y, pred, average=None, labels=list(range(N)))
    print(f"{b:>5.2f} {f1_score(y, pred, average='macro'):>9.4f} {pc[2]:>6.3f} {pc[3]:>6.3f} {pc[5]:>6.3f}")

print("\n--- NESTED CV (b chosen on disjoint users) ---")
folds = list(GroupKFold(5).split(base, groups=groups))
nested = np.zeros(len(y), int); chosen = []
for tr, te in folds:
    bestb, bestf = 0.0, -1
    for b in BGRID:
        f = f1_score(y[tr], (cals[b][tr] * np.exp(PEAK_LOGW)).argmax(1), average="macro")
        if f > bestf:
            bestf, bestb = f, b
    chosen.append(bestb)
    nested[te] = (cals[bestb][te] * np.exp(PEAK_LOGW)).argmax(1)
nf = f1_score(y, nested, average="macro")
print(f"  chosen b per fold: {chosen}")
print(f"  production: {prod:.4f}   nested orient-blend: {nf:.4f}   delta {nf - prod:+.4f}")
if nf > prod + 1e-4:
    bb = float(np.median(chosen))
    _, calt = blended(bb)
    preds = (calt * np.exp(PEAK_LOGW)).argmax(1)
    sub = ROOT / "submissions" / f"sub_orient_blend_b{int(bb*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
    print(f"  NESTED WIN -> wrote {sub}")
else:
    print("  No honest improvement from blending orient_lgbm into production.")
