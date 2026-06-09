"""GAP A: test-prior / label-shift correction (Saerens EM 2002 + BBSE Lipton 2018).

The threshold grid is tuned to TRAIN priors; the private split (disjoint users) may
carry different class priors. Estimate the test prior from the model's own predictions
(no labels), quantify the shift, VALIDATE the correction mechanism on cross-user OOF
under simulated prior shift, and emit prior-corrected submissions.

Principled (adapts to the test distribution via the model's confusion structure) and
NOT public-chasing. Local. Run: .venv\\Scripts\\python.exe scripts\\gapA_label_shift.py
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
N = 6; ALPHA = 0.842; RNG = 42
np.random.seed(RNG)
PEAK = np.array([0.4124252058711867, -0.30, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])
ROBUST = np.array([0.4124252058711867, -0.20, 0.90, 0.4628951701768874, -0.239947242877496, -0.42948082285098554])

meta = pd.read_parquet(DATA / "meta_train.parquet")
y = meta["label"].values.astype(int); g = meta["user_id"].values
test_ids = pd.read_parquet(DATA / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(OOF / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
og, ogt = load("orient_lgbm_oof.npy"), load("orient_lgbm_test_probs.npy")
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
# fold orientation L2-injection (w=0.15) into the production posteriors so we correct
# the actual 0.8200 model:
dgc, dgc_t = iso_pair(norm(og), norm(ogt))
cal[:, 2] = 0.85 * cal[:, 2] + 0.15 * dgc[:, 2]; cal = norm(cal)
calt[:, 2] = 0.85 * calt[:, 2] + 0.15 * dgc_t[:, 2]; calt = norm(calt)

train_prior = np.bincount(y, minlength=N) / len(y)


def saerens(post, init=None, iters=200, tol=1e-9):
    """EM estimate of the test prior from posteriors `post` under label shift."""
    pri = train_prior.copy() if init is None else init.copy()
    for _ in range(iters):
        ratio = pri / np.clip(train_prior, 1e-12, None)
        adj = norm(post * ratio)
        new = adj.mean(0)
        if np.abs(new - pri).max() < tol:
            pri = new; break
        pri = new
    return pri


def bbse(oof_post, oof_y, test_post):
    """Black-Box Shift Estimation: solve C_cond @ p_test = q_test (hard preds)."""
    yhat_oof = oof_post.argmax(1); yhat_test = test_post.argmax(1)
    Ccond = np.zeros((N, N))
    for j in range(N):                       # P(yhat=i | y=j)
        mj = oof_y == j
        if mj.sum():
            Ccond[:, j] = np.bincount(yhat_oof[mj], minlength=N) / mj.sum()
    q = np.bincount(yhat_test, minlength=N) / len(yhat_test)
    p, *_ = np.linalg.lstsq(Ccond, q, rcond=None)
    p = np.clip(p, 0, None)
    return p / p.sum()


p_saer = saerens(calt)
p_bbse = bbse(cal, y, calt)
print("=== (1) Estimated TEST prior vs TRAIN prior ===", flush=True)
print("  class :   train   Saerens   BBSE    pred-dist(uncorr)", flush=True)
qpred = np.bincount((calt * np.exp(ROBUST)).argmax(1), minlength=N) / len(test_ids)
for c in range(N):
    print(f"  L{c}    :  {train_prior[c]:.4f}  {p_saer[c]:.4f}  {p_bbse[c]:.4f}   {qpred[c]:.4f}",
          flush=True)
shift = np.abs(p_saer - train_prior).sum()
print(f"  total |Saerens-train| L1 shift = {shift:.4f}  "
      f"({'NON-trivial' if shift>0.05 else 'small'})", flush=True)

# ---------- (2) mechanism validation on cross-user OOF under SIMULATED shift ----------
print("\n=== (2) Mechanism check: simulated prior shift on OOF (PEAK thresholds) ===",
      flush=True)
print("  scenario        uncorrected  Saerens-corr  oracle-corr", flush=True)
rs = np.random.RandomState(RNG)
scenarios = {"2x minorities": np.array([1, 1, 2.5, 2.5, 2.5, 2.5]),
             "0.5x minorities": np.array([1, 1, 0.4, 0.4, 0.4, 0.4]),
             "boost L1": np.array([1, 2.2, 1, 1, 1, 1]),
             "flatten": np.array([1, 1, 4, 4, 8, 4])}
for _ in range(4):
    scenarios[f"dirichlet#{_}"] = rs.dirichlet(np.ones(N) * 0.7) / train_prior
for name, mult in scenarios.items():
    targ = norm((train_prior * mult)[None])[0]
    w = targ / train_prior; w = w / w.sum() * N
    pr = np.clip(w[y], 1e-6, None); pr = pr / pr.sum()
    idx = rs.choice(len(y), 4000, p=pr)            # resample OOF to target prior
    ci, yi = cal[idx], y[idx]
    unc = f1_score(yi, (ci * np.exp(PEAK)).argmax(1), average="macro")
    est = saerens(ci)                              # estimate shifted prior from preds
    cor = f1_score(yi, (ci * np.exp(PEAK) * (est / train_prior)).argmax(1), average="macro")
    true_pri = np.bincount(yi, minlength=N) / len(yi)
    ora = f1_score(yi, (ci * np.exp(PEAK) * (true_pri / train_prior)).argmax(1), average="macro")
    print(f"  {name:15s}  {unc:.4f}      {cor:.4f}      {ora:.4f}  "
          f"({cor-unc:+.4f})", flush=True)

# ---------- (3) corrected submissions (frozen thresholds x estimated test-prior) ----------
print("\n=== (3) prior-corrected submissions ===", flush=True)
for tag, lw in [("robust", ROBUST), ("peak", PEAK)]:
    base_pred = (calt * np.exp(lw)).argmax(1)
    for pname, pt in [("saerens", p_saer), ("bbse", p_bbse)]:
        corr = (calt * np.exp(lw) * (pt / train_prior)).argmax(1)
        out = SUB / f"sub_{tag}_orient_priorcorr_{pname}.csv"
        pd.DataFrame({"Id": test_ids, "Label": corr.astype(int)}).to_csv(out, index=False)
        print(f"  {tag}+{pname}: {int((corr!=base_pred).sum())} rows differ from "
              f"uncorrected -> {out.name}", flush=True)

print("\nNOTE: if the shift is small AND simulated-shift correction barely helps, prior", flush=True)
print("correction is a no-op here (we'd already be robust); if shift is real and Saerens", flush=True)
print("~ oracle > uncorrected, the corrected CSV is a more robust private pick.", flush=True)
