"""Equilibrium Optimizer (EO) for binary feature selection.

Reference: Faramarzi, Heidarinejad, Stephens, Mirjalili (2020),
"Equilibrium Optimizer: A novel optimization algorithm." Knowledge-Based
Systems 191, 105190.

Applied to feature selection per Topuz & Kaya (2025), "EO-LGBM-HAR".

Concept
-------
The optimisation views the feature mask as a *concentration* in a control
volume. Each particle is a continuous vector in [0, 1]^D; we threshold at
0.5 to obtain a binary mask. Updates blend three terms:

  1. equilibrium term  — pull toward one of the best-known points
  2. exploration term  — concentration difference scaled by an exponential
                         turnover-rate factor F
  3. generation term   — a small "produced mass" component G that helps
                         exploit good regions

The four best historical particles plus their mean form the "equilibrium
pool". Each iteration each particle picks one pool member at random.

Fitness function (caller-supplied):
  fitness(mask: np.ndarray[bool]) -> float
  where lower is better (we MINIMISE). Typical form:
      α · (1 - accuracy) + β · (|S| / D)
  with α=0.99, β=0.01 (from Topuz & Kaya 2025).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EOConfig:
    n_particles: int = 15
    n_iterations: int = 25
    dim: int = 0                # set by caller (= number of features)
    a1: float = 2.0             # exploration coefficient
    a2: float = 1.0             # exploitation coefficient
    GP: float = 0.5             # generation probability
    binarisation_threshold: float = 0.5
    seed: int = 42
    verbose: bool = True
    min_features: int = 5       # safety: never accept fewer than this
    # If the entire pool is identical, bump diversity with random restart
    restart_diversity_tol: float = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _to_binary(c: np.ndarray, threshold: float) -> np.ndarray:
    return (c >= threshold).astype(bool)


def equilibrium_optimizer(
    fitness_fn: Callable[[np.ndarray], float],
    cfg: EOConfig,
    init_mask: Optional[np.ndarray] = None,
) -> dict:
    """Run EO over a binary feature mask of length cfg.dim.

    Parameters
    ----------
    fitness_fn : callable(mask) -> float, lower is better.
    cfg        : EOConfig.
    init_mask  : optional bool array used to seed one of the particles
                 (e.g. "all features on" as a baseline).

    Returns
    -------
    dict with keys:
      best_mask        : 1-D bool array of length cfg.dim
      best_fitness     : float
      history          : list of dicts (per iteration)
      n_evaluations    : int
      elapsed_seconds  : float
    """
    rng = np.random.default_rng(cfg.seed)
    D = cfg.dim

    # ── Initialise population (continuous concentrations in [0, 1]) ──
    C = rng.uniform(0, 1, size=(cfg.n_particles, D))
    if init_mask is not None:
        # Replace particle 0's concentration with a slightly noised version of
        # init_mask, so EO has a strong starting reference.
        C[0] = init_mask.astype(float) + rng.normal(0, 0.05, size=D)
        C[0] = np.clip(C[0], 0, 1)

    # Evaluate
    fitness = np.empty(cfg.n_particles, dtype=np.float64)
    for i in range(cfg.n_particles):
        mask = _to_binary(C[i], cfg.binarisation_threshold)
        if mask.sum() < cfg.min_features:
            # ensure min features by activating top-k by concentration
            top_idx = np.argsort(-C[i])[: cfg.min_features]
            mask = np.zeros(D, dtype=bool); mask[top_idx] = True
        fitness[i] = fitness_fn(mask)
        if cfg.verbose:
            print(f"  init particle {i}: |S|={int(mask.sum())}  "
                  f"fitness={fitness[i]:.5f}", flush=True)

    # Equilibrium pool: 4 best particles + their average  → 5 candidates
    def build_pool(C: np.ndarray, fit: np.ndarray):
        order = np.argsort(fit)
        c_eq = C[order[:4]]
        c_mean = c_eq.mean(axis=0, keepdims=True)
        return np.concatenate([c_eq, c_mean], axis=0)  # (5, D)

    pool = build_pool(C, fitness)
    best_idx_in_pop = int(np.argmin(fitness))
    best_C = C[best_idx_in_pop].copy()
    best_fit = float(fitness[best_idx_in_pop])
    best_mask = _to_binary(best_C, cfg.binarisation_threshold)

    history = []
    n_evals = cfg.n_particles
    t0 = time.time()

    for t in range(cfg.n_iterations):
        # Update rule constants (Faramarzi 2020 eqs)
        # t_norm decreases from 1 to 0 over iterations; controls turnover rate.
        t_norm = (1.0 - (t + 1) / cfg.n_iterations) ** (cfg.a2 * (t + 1) / cfg.n_iterations)

        new_C = np.empty_like(C)
        for i in range(cfg.n_particles):
            # Pick a random equilibrium candidate from the pool
            c_eq = pool[rng.integers(0, pool.shape[0])]

            # Per-dimension λ ~ U(0,1), random vector r ~ U(0,1)
            lam = rng.uniform(0, 1, size=D)
            r = rng.uniform(0, 1, size=D)

            # F factor (turnover) — Faramarzi eq. 11:
            F = cfg.a1 * np.sign(r - 0.5) * (np.exp(-lam * t_norm) - 1.0)

            # GCP (generation control probability) — eq. 15:
            r1 = rng.uniform(0, 1, size=D)
            r2 = rng.uniform(0, 1, size=D)
            GCP = np.where(r2 >= cfg.GP, 0.0, 0.5 * r1)  # 0 or 0.5*r1

            # G0 (initial generation), G (generation rate) — eqs. 14, 13:
            G0 = GCP * (c_eq - lam * C[i])
            G = G0 * F

            # Update — eq. 16:
            V = 1.0  # volume = 1 (per convention)
            new_C[i] = c_eq + (C[i] - c_eq) * F + (G / (lam * V)) * (1.0 - F)
            new_C[i] = np.clip(new_C[i], 0.0, 1.0)

        # Evaluate new population
        new_fit = np.empty(cfg.n_particles)
        for i in range(cfg.n_particles):
            mask = _to_binary(new_C[i], cfg.binarisation_threshold)
            if mask.sum() < cfg.min_features:
                top_idx = np.argsort(-new_C[i])[: cfg.min_features]
                mask = np.zeros(D, dtype=bool); mask[top_idx] = True
            new_fit[i] = fitness_fn(mask)
            n_evals += 1

        # Memory-of-best: each particle keeps the better of (old, new)
        improved = new_fit < fitness
        C[improved] = new_C[improved]
        fitness[improved] = new_fit[improved]

        # Update best
        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_fit:
            best_fit = float(fitness[cur_best_idx])
            best_C = C[cur_best_idx].copy()
            best_mask = _to_binary(best_C, cfg.binarisation_threshold)
            if best_mask.sum() < cfg.min_features:
                top_idx = np.argsort(-best_C)[: cfg.min_features]
                best_mask = np.zeros(D, dtype=bool); best_mask[top_idx] = True

        # Rebuild pool
        pool = build_pool(C, fitness)

        # Diversity guard — if all of pool is essentially the same point,
        # add random kicks to a fraction of the population
        pool_var = pool.std(axis=0).mean()
        if pool_var < cfg.restart_diversity_tol:
            n_kick = cfg.n_particles // 3
            kick_idx = rng.choice(cfg.n_particles, size=n_kick, replace=False)
            for i in kick_idx:
                if i == cur_best_idx:
                    continue
                C[i] = rng.uniform(0, 1, size=D)
                mask = _to_binary(C[i], cfg.binarisation_threshold)
                if mask.sum() < cfg.min_features:
                    top_idx = np.argsort(-C[i])[: cfg.min_features]
                    mask = np.zeros(D, dtype=bool); mask[top_idx] = True
                fitness[i] = fitness_fn(mask)
                n_evals += 1

        # Log
        sizes = [int(_to_binary(C[i], cfg.binarisation_threshold).sum())
                 for i in range(cfg.n_particles)]
        rec = {
            "iter": t,
            "best_fitness": best_fit,
            "best_mask_size": int(best_mask.sum()),
            "pop_min_fit": float(fitness.min()),
            "pop_mean_fit": float(fitness.mean()),
            "pop_mean_size": float(np.mean(sizes)),
            "pool_diversity": float(pool_var),
        }
        history.append(rec)
        if cfg.verbose:
            elapsed = time.time() - t0
            print(f"  EO iter {t + 1:3d}/{cfg.n_iterations}  "
                  f"best_fit={best_fit:.5f}  |S*|={int(best_mask.sum())}  "
                  f"pop_mean_fit={fitness.mean():.5f}  "
                  f"pool_div={pool_var:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return {
        "best_mask": best_mask,
        "best_fitness": best_fit,
        "history": history,
        "n_evaluations": n_evals,
        "elapsed_seconds": time.time() - t0,
    }
