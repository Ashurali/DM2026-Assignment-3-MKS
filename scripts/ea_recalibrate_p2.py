"""(b) EA recalibration of P2 (hier_v6) on frozen OOF.

Fit a Dirichlet evidential head on P2's log-probs: Stage-1 SORM + Stage-2
uncertainty-weighted calibration on held-out USERS (class-balanced). Produce a
recalibrated P2 (outer GroupKFold so the OOF stays valid), then run the exact
production blend [alpha*P1 + (1-alpha)*EA_P2 -> isotonic -> NM + 31x31 L1/L2
grid] and compare to plain-P2 (must reproduce 0.7880). All frozen OOFs, CPU.

Run: python scripts/ea_recalibrate_p2.py
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
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.evidential_align import (
    EvidentialHead, sorm_loss, reweighted_ce, anchor_penalty,
    probs_from_evidence, lambda_anneal,
)

N = 6
ALPHA = 0.842
SEED = 42

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(np.int64)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")

torch.manual_seed(SEED)


def to_logp(P):
    return np.log(np.clip(P, 1e-6, 1.0)).astype(np.float32)


def train_ea_head(P_tr, y_tr, g_tr, epochs_s=25, epochs_c=15, eta=8, beta=1.0):
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
    si, ci = next(gss.split(P_tr, y_tr, g_tr))
    X = torch.tensor(to_logp(P_tr))
    Y = torch.tensor(y_tr, dtype=torch.long)
    head = EvidentialHead(N, n_classes=N, hidden_dim=32)
    cnt = np.bincount(y_tr[si], minlength=N).astype(float); cnt[cnt == 0] = 1
    cw = torch.tensor(1.0 / np.sqrt(cnt), dtype=torch.float32); cw = cw / cw.mean()
    # Stage 1 — SORM
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(TensorDataset(X[si], Y[si]), batch_size=256, shuffle=True)
    for ep in range(epochs_s):
        lam = lambda_anneal(ep + 1, eta)
        for xb, yb in ld:
            opt.zero_grad(); loss, _ = sorm_loss(head(xb), yb, lam, class_weights=cw)
            loss.backward(); opt.step()
    theta1 = {k: v.detach().clone() for k, v in head.named_parameters()}
    # Stage 2 — uncertainty-weighted calibration, class-balanced
    cc = np.bincount(y_tr[ci], minlength=N).astype(float); cc[cc == 0] = 1
    sw = (1.0 / cc)[y_tr[ci]]
    samp = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(ci), replacement=True)
    ldc = DataLoader(TensorDataset(X[ci], Y[ci]), batch_size=256, sampler=samp)
    opt2 = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs_c):
        for xb, yb in ldc:
            opt2.zero_grad(); ce, _ = reweighted_ce(head(xb), yb)
            loss = ce + beta * anchor_penalty(head, theta1)
            loss.backward(); opt2.step()
    return head


@torch.no_grad()
def ea_predict(head, P):
    head.eval()
    return probs_from_evidence(head(torch.tensor(to_logp(P)))).numpy().astype(np.float64)


print("Building EA-recalibrated P2 (outer GroupKFold)…", flush=True)
ea_p2 = np.zeros_like(p2)
for tr, va in GroupKFold(5).split(p2, y, groups):
    head = train_ea_head(p2[tr], y[tr], groups[tr])
    ea_p2[va] = ea_predict(head, p2[va])
ea_p2t = ea_predict(train_ea_head(p2, y, groups), p2t)


def norm(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-12, None)


def iso(oof_raw, test_raw):
    cal = np.zeros_like(oof_raw)
    for tr, va in GroupKFold(5).split(np.zeros(len(y)), groups=groups):
        for c in range(N):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof_raw[tr, c], (y[tr] == c).astype(float)); cal[va, c] = ir.predict(oof_raw[va, c])
    calt = np.zeros_like(test_raw)
    for c in range(N):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof_raw[:, c], (y == c).astype(float)); calt[:, c] = ir.predict(test_raw[:, c])
    return norm(cal), norm(calt)


def nm(probs):
    f = lambda lw: -f1_score(y, (probs * np.exp(lw)).argmax(1), average="macro")
    bx, bv = np.zeros(N), f(np.zeros(N)); rng = np.random.default_rng(42)
    for x0 in [np.zeros(N)] + [rng.uniform(-1, 1, N) for _ in range(8)]:
        r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True})
        if r.fun < bv:
            bv, bx = r.fun, r.x
    return bx


def grid(cal, lw0, g=31, rng=1.5):
    lg = np.linspace(-rng, rng, g); best = (-1.0, lw0.copy())
    for l1 in lg:
        for l2 in lg:
            lw = lw0.copy(); lw[1] = l1; lw[2] = l2
            s = f1_score(y, (cal * np.exp(lw)).argmax(1), average="macro")
            if s > best[0]:
                best = (s, lw.copy())
    return best


def run_blend(P2oof, P2test, tag):
    bo = norm(ALPHA * p1 + (1 - ALPHA) * P2oof); bt = norm(ALPHA * p1t + (1 - ALPHA) * P2test)
    co, ct = iso(bo, bt); lw = nm(co); pf, lwp = grid(co, lw)
    pc = f1_score(y, (co * np.exp(lwp)).argmax(1), average=None)
    print(f"{tag:>22}: peakF1={pf:.4f}  L2={pc[2]:.3f}  per-class={[round(float(x), 3) for x in pc]}", flush=True)
    return pf, pc, lwp, co, ct


print(f"raw OOF: plain-P2={f1_score(y, p2.argmax(1), average='macro'):.4f}  "
      f"EA-P2={f1_score(y, ea_p2.argmax(1), average='macro'):.4f}", flush=True)
pf0, _, _, co0, ct0 = run_blend(p2, p2t, "plain-P2 (production)")
pf1, pc1, lwp1, co1, ct1 = run_blend(ea_p2, ea_p2t, "EA-P2 (re-tuned)")

# ── Fair test: FREEZE the top-1 (0.8154) grid-peak thresholds — NO re-tuning ──
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])
print("\n--- FROZEN top-1 (0.8154) thresholds, no re-tuning ---", flush=True)
for tag, co in [("plain-P2", co0), ("EA-P2", co1)]:
    pr = (co * np.exp(PEAK_LOGW)).argmax(1)
    print(f"  {tag:>10} @ top1-thresh: OOF F1={f1_score(y, pr, average='macro'):.4f}  "
          f"L2={f1_score(y, pr, average=None)[2]:.3f}", flush=True)
preds = (ct1 * np.exp(PEAK_LOGW)).argmax(1)
sub = ROOT / "submissions" / "sub_v6_eaP2_top1thresh.csv"
pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
print(f"  EA-P2 @ frozen top-1 thresholds → wrote {sub}", flush=True)
print(f"\nNote: re-tuned EA-P2 LB was 0.7990 (worse — threshold trap). This frozen-threshold "
      f"version is the clean test of EA's effect under the proven config.", flush=True)
