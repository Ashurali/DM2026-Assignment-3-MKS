"""CNN-BiGRU backbone for HAR — feature extractor for Evidential Alignment.

Mirrors `cnn_bilstm.CNNBiLSTM` but:
  1. swaps the BiLSTM for a **BiGRU** (the user's "autoregressive GRU" direction);
  2. exposes `forward_features()` so the Evidential Alignment trainer
     (`train_gru_evidential.py`) can FREEZE the backbone and retrain only the
     last layer — the paper's last-layer-only second-order-risk-min +
     calibration (Ye, Zheng, Zhang, KDD 2025).

It deliberately reuses `SeqDataset`, the augmentations, `AttentionPool1d`, and
`worker_init_fn` from `cnn_bilstm` so the data pipeline and disjoint-user CV are
byte-for-byte identical to the proven sequence model — only the recurrent cell
and the evidential head differ.

Architecture (input (B, 6, 300)):
    BatchNorm1d
    → 2×[Conv1d(k=5)+ReLU] + MaxPool        300 → 150
    → 2×[Conv1d(k=5)+ReLU] + MaxPool        150 → 75
    → Conv1d(k=3)+ReLU
    → BiGRU(hidden=128)                      (B, 75, 256)
    → AttentionPool over time                (B, 256)  ← `forward_features`
    → Dropout → Linear(256→6)                (B, 6)    ← ERM-phase classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Re-use the validated pieces from the CNN-BiLSTM module so the data path,
# augmentations and disjoint-user folds are identical.
from src.models.cnn_bilstm import (  # noqa: F401  (re-exported for the trainer)
    AttentionPool1d,
    SeqDataset,
    worker_init_fn,
)


class CNNBiGRU(nn.Module):
    """CNN front-end + BiGRU + attention pooling.

    `forward_features(x) -> (B, emb_dim)` returns the pooled embedding (the
    frozen representation used by Evidential Alignment). `forward(x)` adds the
    ERM-phase linear classifier on top.
    """

    def __init__(
        self,
        n_classes: int = 6,
        in_channels: int = 6,
        gru_hidden: int = 128,
        dropout: float = 0.3,
        n_gru_layers: int = 1,
    ):
        super().__init__()
        self.emb_dim = gru_hidden * 2
        self.bn = nn.BatchNorm1d(in_channels)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 300 -> 150
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 150 -> 75
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.attn = AttentionPool1d(gru_hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embedding (B, emb_dim) — the backbone representation."""
        x = self.bn(x)
        x = self.conv_block1(x)            # (B, 64, 150)
        x = self.conv_block2(x)            # (B, 128, 75)
        x = self.conv_block3(x)            # (B, 128, 75)
        x = x.permute(0, 2, 1)             # (B, 75, 128)
        x, _ = self.gru(x)                 # (B, 75, 256)
        return self.attn(x)                # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
