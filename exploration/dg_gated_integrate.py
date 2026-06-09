"""Augment production with the CISC DG model — gated injection under FROZEN thresholds.

Production recipe (frozen): cal = isotonic(norm(0.842*P1 + 0.158*P2)); the final
prediction is (cal * exp(PEAK_LOGW)).argmax(1)  -> OOF 0.7880 / LB 0.8154.

Augmentation (the AUGMENT, not replace, decision): inject the DG model's
probability mass for the under-served classes {L2,L3,L5} into the calibrated
probs, then apply the SAME frozen PEAK_LOGW (never re-tuned):
    cal_aug[:,c] = (1-w)*cal[:,c] + w*dg_cal[:,c]   for c in targets ; renorm
w=0 must reproduce 0.7880 exactly (sanity).

Honest evaluation (defeats the OOF->LB threshold trap):
  - NESTED GroupKFold: the injection weight w is tuned on 4 user-folds and the
    macro-F1 is measured on the held-out 5th fold's users. Assembled across all
    folds, this estimates whether the gain TRANSFERS to unseen users.
  - We only write a submission if the *nested* macro-F1 beats production.

Also reports an OPTIMISTIC stacker (re-tuned thresholds) purely as an upper bound.

Run:  python scripts/dg_gated_integrate.py [--name v1]
"""
from __future__ import annotations
import argparse
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
N = 6
ALPHA = 0.842
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
PEAK_LOGW = np.array([0.4124252058711867, -0.2999999999999998, 0.9000000000000004,
                      0.4628951701768874, -0.239947242877496, -0.42948082285098554])

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v1")
ap.add_argument("--targets", default="2,3,5", help="comma-separated classes to inject DG mass into")
args = ap.parse_args()
TARGETS = tuple(int(t) for t in args.targets.split(","))
print(f"Injecting DG mass into classes {TARGETS}\n")

meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
y = meta["label"].values.astype(int)
groups = meta["user_id"].values
test_ids = pd.read_parquet(ROOT / "data" / "meta_test.parquet")["file_id"].values.astype(int)
load = lambda n: np.load(ROOT / "oof" / n).astype(np.float64)
p1, p1t = load("lgbm_combo_combo_full_v2_oof.npy"), load("lgbm_combo_combo_full_v2_test_probs.npy")
p2, p2t = load("hier_v6_pipeline2_oof.npy"), load("hier_v6_pipeline2_test_probs.npy")
dg, dgt = load(f"dg_cisc_{args.name}_oof.npy"), load(f"dg_cisc_{args.name}_test_probs.npy")


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


def inject(cal, dgc, w, targets=TARGETS):
    out = cal.copy()
    for c in targets:
        out[:, c] = (1 - w) * cal[:, c] + w * dgc[:, c]
    return norm(out)


def macro(cal):
    return f1_score(y, (cal * np.exp(PEAK_LOGW)).argmax(1), average="macro")


cal_prod, cal_prod_t = iso(norm(ALPHA * p1 + (1 - ALPHA) * p2), norm(ALPHA * p1t + (1 - ALPHA) * p2t))
dgc, dgc_t = iso(norm(dg), norm(dgt))

base = macro(cal_prod)
print(f"production baseline OOF macro-F1 @ frozen thresholds = {base:.4f}  (must be 0.7880)\n")

print(f"{'w':>5} {'macroF1':>9} {'L2':>6} {'L3':>6} {'L5':>6}  (frozen PEAK_LOGW; inject {TARGETS})")
for w in WGRID:
    ca = inject(cal_prod, dgc, w)
    pred = (ca * np.exp(PEAK_LOGW)).argmax(1)
    pc = f1_score(y, pred, average=None, labels=list(range(N)))
    print(f"{w:>5.2f} {f1_score(y, pred, average='macro'):>9.4f} {pc[2]:>6.3f} {pc[3]:>6.3f} {pc[5]:>6.3f}")

# ---- NESTED GroupKFold: tune w on train users, evaluate on held-out users ----
print("\n--- NESTED CV (honest transfer; w chosen on disjoint users from eval) ---")
folds = list(GroupKFold(5).split(cal_prod, groups=groups))
nested_pred = np.zeros(len(y), dtype=int)
prod_pred_full = (cal_prod * np.exp(PEAK_LOGW)).argmax(1)
chosen_ws = []
for outer_tr, outer_te in folds:
    best_w, best_f1 = 0.0, -1.0
    for w in WGRID:
        ca = inject(cal_prod[outer_tr], dgc[outer_tr], w)
        f = f1_score(y[outer_tr], (ca * np.exp(PEAK_LOGW)).argmax(1), average="macro")
        if f > best_f1:
            best_f1, best_w = f, w
    chosen_ws.append(best_w)
    ca_te = inject(cal_prod[outer_te], dgc[outer_te], best_w)
    nested_pred[outer_te] = (ca_te * np.exp(PEAK_LOGW)).argmax(1)
nested_f1 = f1_score(y, nested_pred, average="macro")
npc = f1_score(y, nested_pred, average=None, labels=list(range(N)))
print(f"  chosen w per fold: {chosen_ws}")
print(f"  production (frozen):   macro-F1={f1_score(y, prod_pred_full, average='macro'):.4f}")
print(f"  nested DG-augment:     macro-F1={nested_f1:.4f}  (L2={npc[2]:.3f} L3={npc[3]:.3f} L5={npc[5]:.3f})")
print(f"  >>> nested delta = {nested_f1 - base:+.4f} <<<")

# ---- OPTIMISTIC reference: stacker over [cal_prod, dgc] with re-tuned thresholds ----
def nm(probs):
    f = lambda lw: -f1_score(y, (probs * np.exp(lw)).argmax(1), average="macro")
    bx, bv = np.zeros(N), f(np.zeros(N)); rng = np.random.default_rng(42)
    for x0 in [np.zeros(N)] + [rng.uniform(-1, 1, N) for _ in range(6)]:
        r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 500, "adaptive": True})
        if r.fun < bv:
            bv, bx = r.fun, r.x
    return bx

Xs = np.hstack([cal_prod, dgc])
stk = np.zeros((len(y), N))
for tr, va in GroupKFold(5).split(Xs, groups=groups):
    lr = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    lr.fit(Xs[tr], y[tr]); stk[va] = lr.predict_proba(Xs[va])
lw = nm(stk)
print(f"\n  [optimistic] stacker(cal,dg)+RE-TUNED thresholds OOF macro-F1="
      f"{f1_score(y, (stk * np.exp(lw)).argmax(1), average='macro'):.4f}  "
      f"(upper bound only; re-tuned thresholds do NOT transfer reliably)")

# ---- Write submission ONLY if nested gain is real ----
if nested_f1 > base + 1e-4:
    # use the median chosen w applied to the full-data calibration -> test
    w_final = float(np.median(chosen_ws))
    ca_t = inject(cal_prod_t, dgc_t, w_final)
    preds = (ca_t * np.exp(PEAK_LOGW)).argmax(1)
    sub = ROOT / "submissions" / f"sub_dg_cisc_{args.name}_gated_w{int(w_final*100):02d}.csv"
    pd.DataFrame({"Id": test_ids, "Label": preds.astype(int)}).to_csv(sub, index=False)
    print(f"\nNESTED WIN (+{nested_f1 - base:.4f}). Wrote {sub}  (w={w_final})  -> user uploads to Kaggle.")
else:
    print(f"\nNo honest (nested) improvement over production. DG augmentation does not transfer.")
