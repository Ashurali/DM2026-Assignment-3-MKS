"""Test-time augmentation for the saved DL models.

For each saved final-model checkpoint, predict on K augmented variants of
each test sequence and average the softmax probabilities. The augmentations
used here are deliberately MILD (vs training-time aug) because we want to
estimate the model's prediction at the true input plus a small neighborhood,
not generate new training signal.

Augmentations applied (mild):
- Original sequence (always included)
- 2× Gaussian jitter (σ_mean=0.01, σ_std=0.005)
- 2× tiny time shift (±2 samples, circular)
- 2× magnitude scale (×0.97, ×1.03)
NOTE: rotation deliberately NOT used at test-time — the test set's gravity
orientation is the true one, rotating it gives off-distribution predictions.

Usage:
    python scripts/tta_inference.py --runs cnn_bilstm_v1 cnn_bilstm_v2 cnn_bilstm_v3 transformer_v1
    # or with custom K augmentations:
    python scripts/tta_inference.py --runs cnn_bilstm_v1 --n-aug 9

Outputs:
    oof/<run>_tta_test_probs.npy   — TTA-averaged test probabilities
    oof/<run>_tta_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    print("ERROR: PyTorch not installed.", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 6


def load_test_seq() -> tuple[np.ndarray, np.ndarray]:
    cache_x_test = ROOT / "data" / "seq_test.npy"
    cache_test_ids = ROOT / "data" / "seq_test_ids.npy"
    if not cache_x_test.exists():
        raise SystemExit("data/seq_test.npy not found — run the DL training script first to build the cache.")
    return np.load(cache_x_test), np.load(cache_test_ids)


def mild_augmentations(rng: np.random.Generator, n_aug: int = 6) -> list[callable]:
    """Return a list of test-time aug functions, each (X) -> X_modified.

    The first transform is always the identity (the real input). Subsequent
    transforms apply mild perturbations to estimate the local prediction
    neighborhood.
    """
    transforms: list[callable] = [lambda X: X]  # always include original

    # Jitter (additive Gaussian on means; smaller σ on stds to keep them
    # near non-negative)
    def jitter(seed):
        local_rng = np.random.default_rng(seed)
        def fn(X):
            out = X.copy()
            out[:, :3] = out[:, :3] + local_rng.normal(0, 0.01, out[:, :3].shape).astype(np.float32)
            out[:, 3:] = np.clip(out[:, 3:] + local_rng.normal(0, 0.005, out[:, 3:].shape).astype(np.float32), 0.0, None)
            return out
        return fn

    transforms.append(jitter(seed=1001))
    transforms.append(jitter(seed=1002))

    # Tiny time shift (circular)
    def shift(k):
        def fn(X):
            return np.roll(X, k, axis=2)
        return fn
    transforms.append(shift(2))
    transforms.append(shift(-2))

    # Mild magnitude scale
    transforms.append(lambda X: X * 0.97)
    transforms.append(lambda X: X * 1.03)

    return transforms[:n_aug]


def load_model(run_name: str, device: torch.device):
    """Load the final-model checkpoint for a given run name. Auto-detects
    architecture from name prefix."""
    if run_name.startswith("cnn_bilstm_"):
        from src.models.cnn_bilstm import CNNBiLSTM
        model = CNNBiLSTM(n_classes=N_CLASSES)
    elif run_name.startswith("transformer_"):
        from src.models.transformer import TransformerHAR
        model = TransformerHAR(n_classes=N_CLASSES)
    else:
        raise SystemExit(f"Unrecognized run name '{run_name}' — must start with cnn_bilstm_ or transformer_")

    ckpt_path = ROOT / "checkpoints" / run_name / "final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Final checkpoint not found at {ckpt_path}.")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model


def tta_predict(model, X: np.ndarray, transforms: list, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Average softmax probs across all transforms."""
    accumulated = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    for ti, transform in enumerate(transforms):
        Xt = transform(X)
        Xt_t = torch.from_numpy(np.ascontiguousarray(Xt)).float()
        probs_chunks = []
        with torch.no_grad():
            for i in range(0, len(Xt_t), batch_size):
                xb = Xt_t[i:i + batch_size].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
                probs_chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        accumulated += np.concatenate(probs_chunks)
        print(f"    aug {ti+1}/{len(transforms)} done")
    return (accumulated / len(transforms)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="Run names to TTA (e.g. cnn_bilstm_v1 transformer_v1)")
    p.add_argument("--n-aug", type=int, default=7)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Xte, test_ids = load_test_seq()
    print(f"Test set: {Xte.shape}")
    rng = np.random.default_rng(42)
    transforms = mild_augmentations(rng, n_aug=args.n_aug)
    print(f"TTA augmentations: {len(transforms)} variants per sample\n")

    for run in args.runs:
        print(f"=== TTA: {run} ===")
        model = load_model(run, device)
        tta_probs = tta_predict(model, Xte, transforms, device, batch_size=args.batch)

        out_path = ROOT / "oof" / f"{run}_tta_test_probs.npy"
        np.save(out_path, tta_probs)
        print(f"Saved {out_path}")

        # Sidecar
        meta = {
            "run": run,
            "n_aug": len(transforms),
            "test_pred_dist": {int(c): int((tta_probs.argmax(axis=1) == c).sum()) for c in range(N_CLASSES)},
        }
        meta_path = ROOT / "oof" / f"{run}_tta_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved {meta_path}\n")

    print("Done. To use these in blending, pass the TTA test_probs to scripts/blend.py")
    print("(Note: blend.py uses {run}_test_probs.npy by default. Replace those with TTA versions for final blend, or extend blend.py to accept --tta flag.)")


if __name__ == "__main__":
    main()
