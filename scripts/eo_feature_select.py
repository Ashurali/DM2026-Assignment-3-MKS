"""Run Equilibrium Optimizer over the hand-engineered feature stack to
discover a parsimonious subset that (a) maximises macro-F1 under LGBM
group-aware 3-fold CV, and (b) penalises mask size.

Search space — engineered (271) + catch22 (132) = 403 binary dims.

WHY not search over the full ~800-col stack: cnn_emb/transformer_emb/base
model OOFs are already-curated learned features. An LGBM-on-subset fitness
function can't easily exploit their value when they're dropped (the OOFs
need each other to be useful). Keeping them always-on and searching only
the hand-engineered subset is where the spurious-correlation cleanup
should live.

Fitness:
  L(S) = α · (1 − macro_F1_cv(LGBM_full_stack_with_S)) + β · |S| / D
  α = 0.99, β = 0.01  (per Topuz & Kaya 2025)
  Uses 3-fold GroupKFold for speed; we'll validate the winning mask with
  5-fold afterwards.

Inner LGBM:
  multiclass with num_leaves=31, n_estimators=200, learning_rate=0.05.
  Class-weighted via class_weight='balanced'-style weights.

Saves:
  oof/eo_selected_mask.npy           bool (403,) for engineered+catch22 cols
  oof/eo_selected_feature_names.txt  one per line
  oof/eo_history.json                per-iteration diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

print("=== eo_feature_select.py starting ===", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.utils.equilibrium_optimizer import EOConfig, equilibrium_optimizer

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--particles", type=int, default=15)
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--alpha", type=float, default=0.99,
                   help="Weight on (1 − F1) in the fitness.")
    p.add_argument("--beta", type=float, default=0.01,
                   help="Weight on (|S|/D) in the fitness.")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--inner-folds", type=int, default=3,
                   help="GroupKFold folds for the inner LGBM CV fitness.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-features", type=int, default=20)
    return p.parse_args()


def load_features() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load the engineered + catch22 stack we'll search over, plus the
    'always-on' learned features and base-model OOFs that we hand to LGBM
    unconditionally.

    Returns
    -------
    X_search       : (N, D_search)   engineered + catch22 — the search target
    X_always       : (N, D_always)   cnn_emb + transformer_emb + base OOFs
    search_names   : feature column names (length D_search)
    y, groups, test_unused_idx_marker
    """
    eng_tr = pd.read_parquet(ROOT / "data" / "feat_train_none.parquet")
    c22_tr = pd.read_parquet(ROOT / "data" / "feat_catch22_train.parquet")

    eng_cols = [c for c in eng_tr.columns if c != "file_id"]
    c22_cols = [c for c in c22_tr.columns if c != "file_id"]

    X_eng = eng_tr[eng_cols].values.astype(np.float32)
    X_c22 = c22_tr[c22_cols].values.astype(np.float32)
    X_search = np.concatenate([X_eng, X_c22], axis=1)
    search_names = list(eng_cols) + list(c22_cols)

    # Always-on blocks
    cnn_emb = np.load(ROOT / "oof" / "cnn_bilstm_v1_emb_train.npy").astype(np.float32)
    tx_emb = np.load(ROOT / "oof" / "transformer_v1_emb_train.npy").astype(np.float32)
    oof_xgb = np.load(ROOT / "oof" / "xgb_v1_oof.npy").astype(np.float32)
    oof_cat = np.load(ROOT / "oof" / "cat_v1_oof.npy").astype(np.float32)
    oof_mr = np.load(ROOT / "oof" / "minirocket_v1_oof.npy").astype(np.float32)
    X_always = np.concatenate([cnn_emb, tx_emb, oof_xgb, oof_cat, oof_mr], axis=1)

    meta = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    y = meta["label"].values.astype(np.int64)
    groups = meta["user_id"].values

    print(f"Loaded: X_search {X_search.shape}  X_always {X_always.shape}", flush=True)
    return X_search, X_always, search_names, y, groups


def make_fitness(
    X_search: np.ndarray,
    X_always: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    args,
):
    """Build the wrapper fitness function for EO.

    Inner LGBM is the same multiclass config we use everywhere. Class
    weights are inverse-frequency normalised to mean 1, applied per-sample.
    """
    D_search = X_search.shape[1]
    folds = list(GroupKFold(n_splits=n_folds).split(np.zeros(len(y)), groups=groups))

    # Per-sample weights (inverse-frequency; sum to len(y))
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    inv = len(y) / (N_CLASSES * counts)
    sample_w = inv[y].astype(np.float32)

    params = {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": args.lr,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "deterministic": True,
        "seed": args.seed,
    }

    def fitness(mask: np.ndarray) -> float:
        s_size = int(mask.sum())
        # Build per-fold OOF preds
        Xs = X_search[:, mask]
        X = np.concatenate([Xs, X_always], axis=1) if X_always.shape[1] > 0 else Xs

        oof_pred = np.zeros(len(y), dtype=np.int64)
        for tr_idx, va_idx in folds:
            ds_tr = lgb.Dataset(X[tr_idx], y[tr_idx], weight=sample_w[tr_idx])
            m = lgb.train(params, ds_tr, num_boost_round=args.n_estimators)
            probs = m.predict(X[va_idx])
            oof_pred[va_idx] = probs.argmax(axis=1)

        f1 = f1_score(y, oof_pred, average="macro")
        D_eff = D_search  # only penalise search-block size
        loss = args.alpha * (1.0 - f1) + args.beta * (s_size / D_eff)
        return float(loss)

    return fitness, D_search


def main():
    args = parse_args()
    X_search, X_always, search_names, y, groups = load_features()

    fitness_fn, D = make_fitness(X_search, X_always, y, groups, args.inner_folds, args)

    # Quick sanity: baseline (all features on) fitness
    print("\nBaseline (all engineered + catch22 ON):", flush=True)
    all_on = np.ones(D, dtype=bool)
    t0 = time.time()
    baseline_loss = fitness_fn(all_on)
    print(f"  baseline fitness: {baseline_loss:.5f}  "
          f"(F1 ≈ {1 - (baseline_loss - args.beta) / args.alpha:.4f})  "
          f"eval_time={time.time() - t0:.1f}s", flush=True)

    # Run EO
    cfg = EOConfig(
        n_particles=args.particles,
        n_iterations=args.iterations,
        dim=D,
        seed=args.seed,
        verbose=True,
        min_features=args.min_features,
    )
    print(f"\nRunning EO: particles={cfg.n_particles}  iter={cfg.n_iterations}  D={D}", flush=True)
    result = equilibrium_optimizer(fitness_fn, cfg, init_mask=all_on)

    best_mask = result["best_mask"]
    best_fit = result["best_fitness"]
    n_kept = int(best_mask.sum())

    print("\n" + "=" * 60, flush=True)
    print(f"EO done in {result['elapsed_seconds']:.0f}s with "
          f"{result['n_evaluations']} fitness evaluations.", flush=True)
    print(f"Baseline fitness: {baseline_loss:.5f}", flush=True)
    print(f"Best fitness:     {best_fit:.5f}", flush=True)
    print(f"Δ fitness:        {best_fit - baseline_loss:+.5f}", flush=True)
    print(f"Features kept:    {n_kept}/{D}  ({100 * n_kept / D:.1f}%)", flush=True)

    # Estimate F1 components
    baseline_f1 = 1 - (baseline_loss - args.beta * 1.0) / args.alpha
    best_f1_est = 1 - (best_fit - args.beta * (n_kept / D)) / args.alpha
    print(f"Baseline F1 est:  {baseline_f1:.4f}", flush=True)
    print(f"Best F1 est:      {best_f1_est:.4f}", flush=True)
    print(f"Δ F1 est:         {best_f1_est - baseline_f1:+.4f}", flush=True)

    # Save mask + names
    np.save(ROOT / "oof" / "eo_selected_mask.npy", best_mask)
    kept_names = [n for n, on in zip(search_names, best_mask) if on]
    (ROOT / "oof" / "eo_selected_feature_names.txt").write_text("\n".join(kept_names))
    history_json = {
        "baseline_loss": float(baseline_loss),
        "baseline_f1_est": float(baseline_f1),
        "best_loss": float(best_fit),
        "best_f1_est": float(best_f1_est),
        "n_kept": int(n_kept),
        "D_search": int(D),
        "history": [{k: (float(v) if not isinstance(v, int) else int(v)) for k, v in h.items()}
                    for h in result["history"]],
        "config": {
            "particles": cfg.n_particles,
            "iterations": cfg.n_iterations,
            "alpha": args.alpha,
            "beta": args.beta,
            "inner_folds": args.inner_folds,
            "lgbm_params": {"n_estimators": args.n_estimators, "num_leaves": args.num_leaves,
                            "lr": args.lr},
        },
        "n_evaluations": int(result["n_evaluations"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }
    (ROOT / "oof" / "eo_history.json").write_text(json.dumps(history_json, indent=2))
    print(f"\nSaved oof/eo_selected_mask.npy  ({n_kept} features kept)", flush=True)
    print(f"Saved oof/eo_selected_feature_names.txt", flush=True)
    print(f"Saved oof/eo_history.json", flush=True)
    print("\n=== eo_feature_select.py done ===", flush=True)


if __name__ == "__main__":
    main()
