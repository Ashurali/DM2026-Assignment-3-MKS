"""Hierarchical v6: identical multi-seed structure to v4, but with the EO-
selected feature mask applied to engineered + catch22 cols. Always-on
blocks (cnn_emb, transformer_emb, base OOFs, combo_v2 OOF, contrastive
emb) are kept exactly as in v4.

Hypothesis: the EO search dropped the engineered+catch22 features that
encode user-specific signature (wrist orientation baseline, individual
gait peculiarities) — these correlate with user id and are therefore
useless or counterproductive on the disjoint test users. v4 LGBM picks
them up because they're slightly predictive on the SAME users in train.
v6 with these dropped should generalise better to test users — i.e. the
OOF→LB inversion should be smaller (or in the right direction).

After training, compose pipeline 2 v6, blend with combo_full_v2 at
α=0.88 (LB-validated on v4), apply cal+thresh, write submission.

Usage:
  python scripts/train_hier_v6_eo_selected.py --seeds 17 23 41
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize, minimize_scalar

print("=== train_hier_v6_eo_selected.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6
CLASS_NAMES = ["L0", "L1", "L2", "L3", "L4", "L5"]

from src.utils.hier_common import build_feature_blocks, label_to_super
from src.utils.cv import make_folds

sys.path.insert(0, str(ROOT / "scripts"))
from train_hier_multi_seed import (
    train_coarse_one_seed, fine_walk_lgbm_seed, fine_walk_xgb_seed, fine_other_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    p.add_argument("--include-contrast", action="store_true", default=True,
                   help="Include l1l2 contrastive emb (matches v4 stack).")
    return p.parse_args()


def compose_pipeline2(coarse, fw, fo):
    n = len(coarse)
    out = np.zeros((n, N_CLASSES), dtype=np.float64)
    out[:, 0] = coarse[:, 0]
    out[:, 1] = coarse[:, 1] * fw[:, 0]
    out[:, 2] = coarse[:, 1] * fw[:, 1]
    out[:, 3] = coarse[:, 2] * fo[:, 0]
    out[:, 4] = coarse[:, 2] * fo[:, 1]
    out[:, 5] = coarse[:, 2] * fo[:, 2]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def isotonic_oof(oof_raw, y, groups):
    cal_oof = np.zeros_like(oof_raw)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(np.zeros(len(y)), groups=groups):
        for c in range(N_CLASSES):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
            iso.fit(oof_raw[tr_idx, c], (y[tr_idx] == c).astype(float))
            cal_oof[va_idx, c] = iso.predict(oof_raw[va_idx, c])
    cal_oof = cal_oof / np.clip(cal_oof.sum(axis=1, keepdims=True), 1e-12, None)
    full_isos = []
    for c in range(N_CLASSES):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        iso.fit(oof_raw[:, c], (y == c).astype(float))
        full_isos.append(iso)
    return cal_oof, full_isos


def apply_isos(test_raw, full_isos):
    out = np.zeros_like(test_raw)
    for c in range(N_CLASSES):
        out[:, c] = full_isos[c].predict(test_raw[:, c])
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def tune_thresholds(probs, y):
    def neg_f1(lw):
        return -float(f1_score(y, (probs * np.exp(lw)).argmax(axis=1), average="macro"))
    best_x, best_v = np.zeros(N_CLASSES), neg_f1(np.zeros(N_CLASSES))
    rng = np.random.default_rng(42)
    for x0 in [np.zeros(N_CLASSES)] + [rng.uniform(-1.0, 1.0, N_CLASSES) for _ in range(8)]:
        res = minimize(neg_f1, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600, "adaptive": True})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x


def tune_alpha(p1, p2, y):
    def neg_f1(a):
        a = float(np.clip(a, 0, 1))
        return -float(f1_score(y, (a * p1 + (1 - a) * p2).argmax(axis=1), average="macro"))
    res = minimize_scalar(neg_f1, bounds=(0, 1), method="bounded", options={"xatol": 1e-3})
    return float(res.x), float(-res.fun)


def make_submission(p1_oof, p2_oof, p1_test, p2_test, alpha, y, groups, test_ids, name):
    blend_oof = alpha * p1_oof + (1 - alpha) * p2_oof
    blend_test = alpha * p1_test + (1 - alpha) * p2_test
    blend_oof = blend_oof / np.clip(blend_oof.sum(axis=1, keepdims=True), 1e-12, None)
    blend_test = blend_test / np.clip(blend_test.sum(axis=1, keepdims=True), 1e-12, None)
    cal_oof, full_isos = isotonic_oof(blend_oof, y, groups)
    cal_test = apply_isos(blend_test, full_isos)
    log_w = tune_thresholds(cal_oof, y)
    w = np.exp(log_w)
    final_pred_oof = (cal_oof * w).argmax(axis=1)
    final_pred_test = (cal_test * w).argmax(axis=1)
    f1 = float(f1_score(y, final_pred_oof, average="macro"))
    pc = f1_score(y, final_pred_oof, average=None)
    print(f"  α={alpha:.3f}  cal+thresh OOF F1: {f1:.4f}  L2={pc[2]:.3f}", flush=True)
    sub_path = ROOT / "submissions" / f"{name}.csv"
    pd.DataFrame({"Id": test_ids, "Label": final_pred_test.astype(int)}).to_csv(sub_path, index=False)
    print(f"  Wrote {sub_path}", flush=True)
    return f1, pc


def main():
    args = parse_args()

    # Load EO-selected mask
    mask_path = ROOT / "oof" / "eo_selected_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"EO mask not found at {mask_path}. "
                                "Run scripts/eo_feature_select.py first.")
    eo_mask = np.load(mask_path)
    print(f"Loaded EO mask: {int(eo_mask.sum())}/{len(eo_mask)} features kept "
          f"({100 * eo_mask.sum() / len(eo_mask):.1f}%)", flush=True)

    X, Xte, block_names = build_feature_blocks(
        include_combo_oof=True,
        include_l1l2_contrast_emb=args.include_contrast,
        eo_mask=eo_mask,
    )
    print(f"Features (v6): {block_names}", flush=True)
    print(f"X {X.shape}  Xte {Xte.shape}", flush=True)

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    meta_test = pd.read_parquet(ROOT / "data" / "meta_test.parquet")
    y6 = meta_train["label"].values.astype(np.int64)
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)
    test_ids = meta_test["file_id"].values.astype(int)

    # ─── Stage 1 ───
    print(f"\n── Stage 1: coarse 3-way v6 (seeds={args.seeds}) ──", flush=True)
    y3 = label_to_super(y6)
    oof_sum = np.zeros((len(y3), 3), dtype=np.float64)
    test_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for seed in args.seeds:
        oof_s, test_s = train_coarse_one_seed(X, Xte, y3, folds, seed, args.gpu, 500)
        oof_sum += oof_s; test_sum += test_s
    coarse_oof = oof_sum / len(args.seeds)
    coarse_test = test_sum / len(args.seeds)
    coarse_oof = coarse_oof / np.clip(coarse_oof.sum(axis=1, keepdims=True), 1e-12, None)
    coarse_test = coarse_test / np.clip(coarse_test.sum(axis=1, keepdims=True), 1e-12, None)
    print(f"Coarse v6 MS OOF F1: {f1_score(y3, coarse_oof.argmax(1), average='macro'):.4f}",
          flush=True)

    # ─── Stage 2a ───
    print(f"\n── Stage 2a: fine_walk v6 (seeds={args.seeds}, LGBM+XGB) ──", flush=True)
    walk_mask = (y6 == 1) | (y6 == 2)
    y_bin = np.zeros_like(y6); y_bin[y6 == 2] = 1
    pieces_oof, pieces_test = [], []
    for seed in args.seeds:
        oof_l, test_l = fine_walk_lgbm_seed(X, Xte, y_bin, walk_mask, folds, seed, args.gpu, 400)
        pieces_oof.append(oof_l); pieces_test.append(test_l)
        oof_x, test_x = fine_walk_xgb_seed(X, Xte, y_bin, walk_mask, folds, seed, args.gpu, 400)
        pieces_oof.append(oof_x); pieces_test.append(test_x)
    p_l2_oof = np.mean(pieces_oof, axis=0)
    p_l2_test = np.mean(pieces_test, axis=0)
    walk_idx = np.where(walk_mask)[0]
    preds = (p_l2_oof[walk_idx] >= 0.5).astype(int)
    print(f"Fine_walk v6 MS OOF F1 on walking: "
          f"{f1_score(y_bin[walk_idx], preds, average='macro'):.4f}", flush=True)
    print(classification_report(y_bin[walk_idx], preds, target_names=["L1", "L2"], digits=4))
    fw_oof_2col = np.stack([1 - p_l2_oof, p_l2_oof], axis=1).astype(np.float32)
    fw_test_2col = np.stack([1 - p_l2_test, p_l2_test], axis=1).astype(np.float32)

    # ─── Stage 2b ───
    print(f"\n── Stage 2b: fine_other v6 (seeds={args.seeds}) ──", flush=True)
    other_mask = (y6 == 3) | (y6 == 4) | (y6 == 5)
    y_local = np.zeros_like(y6)
    y_local[y6 == 3] = 0; y_local[y6 == 4] = 1; y_local[y6 == 5] = 2
    oof_sum = np.zeros((len(y_local), 3), dtype=np.float64)
    test_sum = np.zeros((len(Xte), 3), dtype=np.float64)
    for seed in args.seeds:
        oof_s, test_s = fine_other_seed(X, Xte, y_local, other_mask, folds, seed, args.gpu, 400)
        oof_sum += oof_s; test_sum += test_s
    other_oof = oof_sum / len(args.seeds)
    other_test = test_sum / len(args.seeds)
    other_oof = other_oof / np.clip(other_oof.sum(axis=1, keepdims=True), 1e-12, None)
    other_test = other_test / np.clip(other_test.sum(axis=1, keepdims=True), 1e-12, None)
    other_idx = np.where(other_mask)[0]
    print(f"Fine_other v6 MS OOF F1: "
          f"{f1_score(y_local[other_idx], other_oof[other_idx].argmax(1), average='macro'):.4f}",
          flush=True)

    # ─── Compose + blend ───
    p1_oof = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_oof.npy").astype(np.float64)
    p1_test = np.load(ROOT / "oof" / "lgbm_combo_combo_full_v2_test_probs.npy").astype(np.float64)
    p2_oof = compose_pipeline2(coarse_oof, fw_oof_2col, other_oof)
    p2_test = compose_pipeline2(coarse_test, fw_test_2col, other_test)

    p2_f1 = float(f1_score(y6, p2_oof.argmax(1), average="macro"))
    p2_pc = f1_score(y6, p2_oof.argmax(1), average=None)
    print(f"\nP2 v6 raw OOF F1: {p2_f1:.4f}  (refs: v1=0.7408, v2=0.7436, v3=0.7409, "
          f"v4=0.7458, v5=0.7366)", flush=True)
    print(f"P2 v6 per-class F1: {[round(float(x), 4) for x in p2_pc]}", flush=True)
    print(classification_report(y6, p2_oof.argmax(1), target_names=CLASS_NAMES, digits=4))

    alpha_auto, _ = tune_alpha(p1_oof, p2_oof, y6)
    print(f"\nOOF-tuned α: {alpha_auto:.3f}  (refs: v4 LB-peak α=0.88)", flush=True)

    print("\n=== Variant 1: pure P2 v6 ===")
    f1_p2only, pc_p2only = make_submission(
        p1_oof, p2_oof, p1_test, p2_test, 0.00, y6, groups, test_ids,
        name="sub_hier_v6_p2_only_cal_thresh",
    )
    print("\n=== Variant 2: α auto-tuned on OOF ===")
    f1_auto, pc_auto = make_submission(
        p1_oof, p2_oof, p1_test, p2_test, alpha_auto, y6, groups, test_ids,
        name="sub_hier_v6_blend_cal_thresh",
    )
    print("\n=== Variant 3: α=0.88 (LB-validated on v4) ===")
    f1_088, pc_088 = make_submission(
        p1_oof, p2_oof, p1_test, p2_test, 0.88, y6, groups, test_ids,
        name="sub_hier_v6_a088_cal_thresh",
    )

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  α=0.00:    OOF cal+thresh = {f1_p2only:.4f}  L2 = {pc_p2only[2]:.3f}")
    print(f"  α={alpha_auto:.3f}:    OOF cal+thresh = {f1_auto:.4f}  L2 = {pc_auto[2]:.3f}")
    print(f"  α=0.88:    OOF cal+thresh = {f1_088:.4f}  L2 = {pc_088[2]:.3f}")
    print()
    print("  Reference (current best LB): v4 hier_blend (α=0.88) → LB 0.8114  OOF 0.7856")
    print("  PRIMARY (if v6 OOF >= v4 0.7856): sub_hier_v6_a088_cal_thresh.csv")

    np.save(ROOT / "oof" / "hier_v6_pipeline2_oof.npy", p2_oof.astype(np.float32))
    np.save(ROOT / "oof" / "hier_v6_pipeline2_test_probs.npy", p2_test.astype(np.float32))

    sidecar = {
        "feature_blocks": block_names,
        "eo_mask_size": int(eo_mask.sum()),
        "eo_mask_total": int(len(eo_mask)),
        "p2_v6_raw_f1": p2_f1,
        "alpha_auto": alpha_auto,
        "variants": {
            "sub_hier_v6_p2_only_cal_thresh": {"alpha": 0.00, "oof_f1": f1_p2only,
                                                "per_class_f1": [float(x) for x in pc_p2only]},
            "sub_hier_v6_blend_cal_thresh":   {"alpha": alpha_auto, "oof_f1": f1_auto,
                                                "per_class_f1": [float(x) for x in pc_auto]},
            "sub_hier_v6_a088_cal_thresh":    {"alpha": 0.88, "oof_f1": f1_088,
                                                "per_class_f1": [float(x) for x in pc_088]},
        },
    }
    (ROOT / "oof" / "hier_v6_meta.json").write_text(json.dumps(sidecar, indent=2))

    log_path = ROOT / "submissions" / "log.md"
    with open(log_path, "a", encoding="utf-8") as f:
        for name, alpha, f1, pc in [
            ("sub_hier_v6_p2_only_cal_thresh", 0.00, f1_p2only, pc_p2only),
            ("sub_hier_v6_blend_cal_thresh", alpha_auto, f1_auto, pc_auto),
            ("sub_hier_v6_a088_cal_thresh", 0.88, f1_088, pc_088),
        ]:
            f.write(f"| {date.today().isoformat()} | {name} | "
                    f"hier v6 (EO-selected feats: {int(eo_mask.sum())}/{len(eo_mask)}) "
                    f"α={alpha:.3f} cal+thresh | "
                    f"{f1:.4f} (OOF) | _pending_ | _pending_ | "
                    f"per-class F1 {[round(float(x), 4) for x in pc]} |\n")
    print(f"\nLogged to {log_path}", flush=True)
    print("\n=== train_hier_v6_eo_selected.py done ===", flush=True)


if __name__ == "__main__":
    main()
