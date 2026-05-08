"""Extract DL embeddings using per-fold checkpoints (NOT the final.pt) to
eliminate train-test feature distribution shift.

Why: combo_full + DL embeddings (from final.pt) had a NEGATIVE LB gap on May 8
(combo_full RAW: OOF 0.7687 → LB 0.7568). Diagnosis: the final.pt embeddings
on train memorize the train samples, while at test they generalize. The combo
LGBM learns wrong calibration.

Fix: extract per-fold embeddings using the same PROTOCOL as OOF predictions —
- For training sample i in fold k's val set, use checkpoints/<run>/fold_k_best.pt
  (a model that did NOT see sample i during training).
- For test samples, average embeddings from all 5 fold models.

Both train and test embeddings now come from "the model hasn't seen this sample"
distributions → matched. Combo's stacked-feature transfer should be honest.

Usage:
    python -m src.models.extract_perfold_embeddings --run cnn_bilstm_v1
    python -m src.models.extract_perfold_embeddings --run transformer_v1

Outputs (overwriting the previous final.pt-based embeddings):
    oof/<run>_emb_train.npy   — per-fold OOF embedding for each train sample
    oof/<run>_emb_test.npy    — averaged-across-folds embedding for each test sample
    oof/<run>_emb_meta.json   — protocol metadata
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

from src.utils.cv import make_folds
from src.models.cnn_bilstm import SeqDataset

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run name (e.g. cnn_bilstm_v1, transformer_v1).")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    return p.parse_args()


def build_model(run_name: str):
    """Instantiate the model class and identify the penultimate layer hook target."""
    if run_name.startswith("cnn_bilstm_"):
        from src.models.cnn_bilstm import CNNBiLSTM
        return CNNBiLSTM(n_classes=N_CLASSES), "fc"
    elif run_name.startswith("transformer_"):
        from src.models.transformer import TransformerHAR
        return TransformerHAR(n_classes=N_CLASSES), "fc"
    elif run_name.startswith("inception_time_"):
        from src.models.inception_time import InceptionTime
        return InceptionTime(in_channels=6, n_classes=N_CLASSES), "fc"
    else:
        raise SystemExit(f"Unknown run prefix in '{run_name}'")


def extract_embeddings(model, X: np.ndarray, device, batch_size: int, target_layer: str = "fc") -> np.ndarray:
    """Forward all samples through the model, capturing the input to the final FC layer."""
    captured: list[torch.Tensor] = []

    def hook(_module, inp, _out):
        captured.append(inp[0].detach().cpu())

    handle = getattr(model, target_layer).register_forward_hook(hook)

    ds = SeqDataset(X, training=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    captured.clear()
    model.eval()
    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _ = model(xb)
    handle.remove()
    return torch.cat(captured, dim=0).float().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Run: {args.run}  Device: {device}")

    seq_train = np.load(ROOT / "data" / "seq_train.npy")
    seq_test = np.load(ROOT / "data" / "seq_test.npy")
    print(f"seq_train {seq_train.shape}, seq_test {seq_test.shape}")

    meta_train = pd.read_parquet(ROOT / "data" / "meta_train.parquet")
    groups = meta_train["user_id"].values
    folds = make_folds(groups, n_splits=5)

    ckpt_dir = ROOT / "checkpoints" / args.run
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint dir {ckpt_dir} not found")

    # Verify all 5 fold checkpoints exist
    fold_ckpts = [ckpt_dir / f"fold_{k}_best.pt" for k in range(5)]
    for p in fold_ckpts:
        if not p.exists():
            raise SystemExit(f"Missing fold checkpoint: {p}")

    # Determine embedding dim from the first model's fc layer
    model, target = build_model(args.run)
    emb_dim = model.fc.in_features
    print(f"Embedding dim: {emb_dim}")

    # ── Training: per-fold OOF embeddings ──
    emb_train = np.zeros((len(seq_train), emb_dim), dtype=np.float32)
    print("\nExtracting per-fold OOF embeddings on train...")
    for k, (tr_idx, va_idx) in enumerate(folds):
        print(f"  Fold {k}: loading {fold_ckpts[k].name} → embedding {len(va_idx)} val samples")
        model, target = build_model(args.run)
        ck = torch.load(fold_ckpts[k], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        emb_va = extract_embeddings(model, seq_train[va_idx], device, args.batch, target)
        emb_train[va_idx] = emb_va
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Test: average across all 5 fold models ──
    print("\nExtracting test embeddings (averaging across 5 fold models)...")
    emb_test_sum = np.zeros((len(seq_test), emb_dim), dtype=np.float32)
    for k in range(5):
        print(f"  Fold {k}: contributing to test average")
        model, target = build_model(args.run)
        ck = torch.load(fold_ckpts[k], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        emb_te = extract_embeddings(model, seq_test, device, args.batch, target)
        emb_test_sum += emb_te
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emb_test = (emb_test_sum / 5.0).astype(np.float32)

    # Save (overwriting any previous final.pt-based embeddings — those were the buggy ones)
    out_train = ROOT / "oof" / f"{args.run}_emb_train.npy"
    out_test = ROOT / "oof" / f"{args.run}_emb_test.npy"
    np.save(out_train, emb_train)
    np.save(out_test, emb_test)
    print(f"\nSaved {out_train}  shape={emb_train.shape}")
    print(f"Saved {out_test}  shape={emb_test.shape}")

    meta_path = ROOT / "oof" / f"{args.run}_emb_meta.json"
    meta_path.write_text(json.dumps({
        "run": args.run,
        "protocol": "per-fold OOF for train; mean of 5 fold-models for test",
        "emb_dim": int(emb_dim),
        "n_train": int(len(seq_train)),
        "n_test": int(len(seq_test)),
        "rationale": (
            "Replaces the previous final.pt-based embeddings. The previous "
            "approach had train embeddings memorized by the final model "
            "(seen all of train) while test embeddings represent generalization, "
            "causing a train-test feature distribution shift that the combo "
            "LGBM learned the wrong calibration for (LB 0.7568 RAW). "
            "Per-fold extraction matches the protocol of OOF predictions — "
            "for sample i in fold k's val set, use a model that did NOT "
            "train on i. For test, average the 5 fold-model embeddings so "
            "test embeddings have similar 'unseen-sample' character."
        ),
    }, indent=2))
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
