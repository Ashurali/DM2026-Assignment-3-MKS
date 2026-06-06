"""Generate GENUINELY-DIFFERENT LB candidates to SUBMIT (don't infer from OOF).
Rationale: (a) orthogonal sources (orient/gaf/InceptionTime) rescue DIFFERENT real L2s;
(b) the test set has MORE L2/L3 than train (Saerens), but thresholds are train-tuned ->
pushing L2/L3 harder should help the actual test even when train-OOF says flat.

Writes several candidates spanning the orthogonal sources + the test-prior shift, each
meaningfully different from the proven 0.8200. Reports OOF + test L2/L3 counts + rows-diff.
Run: .venv\\Scripts\\python.exe scripts\\gen_lb_candidates.py
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
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
tid = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
L = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = L("lgbm_combo_combo_full_v2_oof.npy"), L("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = L("hier_v6_pipeline2_oof.npy"), L("hier_v6_pipeline2_test_probs.npy")
gkf = GroupKFold(5); norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(o, t):
    c = np.zeros_like(o); ct = np.zeros_like(t)
    for tr, va in gkf.split(o, groups=g):
        for k in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(o[tr, k], (y[tr] == k).astype(float)); c[va, k] = ir.predict(o[va, k])
    for k in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(o[:, k], (y == k).astype(float)); ct[:, k] = ir.predict(t[:, k])
    return norm(c), norm(ct)


cal, calt = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
oc, oct_ = iso(norm(L("orient_lgbm_oof.npy")), norm(L("orient_lgbm_test_probs.npy")))
ic, ict = iso(norm(L("inception_time_v1_oof.npy")), norm(L("inception_time_v1_test_probs.npy")))
gc, gct = iso(norm(L("gaf_cnn_v1_oof.npy")), norm(L("gaf_cnn_v1_test_probs.npy")))

# Saerens test prior (re-derive)
tp = np.bincount(y, minlength=N) / len(y)
pri = tp.copy()
for _ in range(200):
    adj = norm(calt * (pri / tp)); new = adj.mean(0)
    if np.abs(new - pri).max() < 1e-9: pri = new; break
    pri = new
wpri = pri / tp

# reference 0.8200 sub = robust + orient-L2 w0.15
def injL(c_, src, col, w):
    o = c_.copy(); o[:, col] = (1 - w) * c_[:, col] + w * src[:, col]; return norm(o)
ref_pred = (injL(calt, oct_, 2, 0.15) * np.exp(ROBUST)).argmax(1)


def emit(name, cal_oof, cal_test, lw):
    m = f1_score(y, (cal_oof * np.exp(lw)).argmax(1), average="macro")
    pred = (cal_test * np.exp(lw)).argmax(1)
    cnt = np.bincount(pred, minlength=N)
    d = int((pred != ref_pred).sum())
    out = SUB / f"sub_{name}.csv"
    pd.DataFrame({"Id": tid, "Label": pred.astype(int)}).to_csv(out, index=False)
    print(f"  {name:32s} OOF={m:.4f}  L2={cnt[2]} L3={cnt[3]}  rows-diff-vs-8200={d:4d}", flush=True)


print(f"Saerens test prior boost: L2x{wpri[2]:.2f} L3x{wpri[3]:.2f} L4x{wpri[4]:.2f} L5x{wpri[5]:.2f}", flush=True)
print(f"ref 0.8200 test counts: L2={np.bincount(ref_pred,minlength=N)[2]} L3={np.bincount(ref_pred,minlength=N)[3]}\n", flush=True)

# --- candidates (all robust base unless noted) ---
# 1: orient-L2 + InceptionTime-L3 (push L3, test has more L3)
c = injL(injL(cal, oc, 2, 0.15), ic, 3, 0.15); ct = injL(injL(calt, oct_, 2, 0.15), ict, 3, 0.15)
emit("robust_orient_L2_incep_L3", c, ct, ROBUST)
# 2: orient-L2 + InceptionTime-L2 (more L2 recall via most-rescuing source)
c = injL(injL(cal, oc, 2, 0.15), ic, 2, 0.12); ct = injL(injL(calt, oct_, 2, 0.15), ict, 2, 0.12)
emit("robust_orient_incep_L2", c, ct, ROBUST)
# 3: triple-orthogonal L2 (orient+gaf+incep)
c = cal.copy(); c[:, 2] = 0.6 * cal[:, 2] + 0.15 * oc[:, 2] + 0.13 * gc[:, 2] + 0.12 * ic[:, 2]; c = norm(c)
ct = calt.copy(); ct[:, 2] = 0.6 * calt[:, 2] + 0.15 * oct_[:, 2] + 0.13 * gct[:, 2] + 0.12 * ict[:, 2]; ct = norm(ct)
emit("robust_orient_gaf_incep_L2", c, ct, ROBUST)
# 4: Saerens prior-corrected + orient-L2 (adapt to test prior: more L2/L3)
c = injL(norm(cal * wpri), oc, 2, 0.15); ct = injL(norm(calt * wpri), oct_, 2, 0.15)
emit("robust_orient_L2_priorcorr", c, ct, ROBUST)
# 5: orient-L2 + MORE aggressive L2 threshold (grab L2 recall, test has more L2)
lw5 = ROBUST.copy(); lw5[2] = 1.15
emit("robust_orient_L2_aggrThresh", injL(cal, oc, 2, 0.15), injL(calt, oct_, 2, 0.15), lw5)
# 6: peak + orient-L2 + incep-L3 (OOF-best base + L3 push)
c = injL(injL(cal, oc, 2, 0.15), ic, 3, 0.12); ct = injL(injL(calt, oct_, 2, 0.15), ict, 3, 0.12)
emit("peak_orient_L2_incep_L3", c, ct, PEAK)
print("\n(submit the genuinely-different ones; OOF is only a noisy proxy -- the LB decides)", flush=True)
