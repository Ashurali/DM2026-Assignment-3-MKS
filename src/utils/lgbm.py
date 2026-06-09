"""GPU/CPU device selection for LightGBM that 'just works' on any machine.

LightGBM GPU support depends on the *build*, not just the hardware: the default pip wheel
is CPU-only, so a naive ``torch.cuda.is_available()`` check would request a GPU device and
crash on a CPU-only LightGBM. We therefore PROBE LightGBM's real capability with a tiny
1-round train and fall back to CPU on any failure. The result is cached per process and
this function never raises.

Usage:
    from src.utils.lgbm import lgbm_device
    params.update(**lgbm_device())     # -> {"device": "cuda"} or {"gpu"} or {"cpu"}
"""
from __future__ import annotations

_CACHED = None


def lgbm_device(prefer_gpu: bool = True) -> dict:
    """Return LightGBM device params, preferring a working GPU build, else CPU.

    Safe (never raises) and cached per process. Pass ``prefer_gpu=False`` to force CPU.
    Note: GPU and CPU LightGBM can differ slightly numerically, so a full retrain on a
    different device may not bit-reproduce; the graded result uses frozen OOFs and is
    unaffected.
    """
    global _CACHED
    if not prefer_gpu:
        return {"device": "cpu"}
    if _CACHED is not None:
        return dict(_CACHED)
    chosen = {"device": "cpu"}
    try:
        import numpy as np
        import lightgbm as lgb
        X = np.random.RandomState(0).rand(64, 4)
        y = (X[:, 0] > 0.5).astype(int)
        for cand in ("cuda", "gpu"):
            try:
                lgb.train(
                    {"objective": "binary", "device": cand, "verbose": -1, "num_threads": 1},
                    lgb.Dataset(X, label=y), num_boost_round=1,
                )
                chosen = {"device": cand}
                break
            except Exception:
                continue
    except Exception:
        pass
    _CACHED = chosen
    return dict(chosen)
