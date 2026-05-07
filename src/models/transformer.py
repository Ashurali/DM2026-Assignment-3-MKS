"""Phase-6 Transformer encoder for raw 6×300 sequences.

Architecture per PROJECT_PLAN.md §Phase 6:
    Input (B, 6, 300)
    → Linear projection (6 → d_model=128)
    → Sinusoidal positional encoding
    → Prepend learnable [CLS] token
    → 4× TransformerEncoderLayer (d_model=128, nhead=4, ff=256, dropout=0.1)
    → Take [CLS] output → LayerNorm → Linear(128→6)

Trained alongside CNN-BiLSTM with similar augmentations (rotation,
jitter, scaling, time-warp), but with stronger mixup (α=0.4 — Transformers
benefit from more aggressive regularization). Lower lr (3e-4) and longer
warmup (5 epochs).

Reuses SeqDataset / augment_sample / worker_init_fn from cnn_bilstm.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal PE (Vaswani 2017). Returns (seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerHAR(nn.Module):
    def __init__(
        self,
        n_classes: int = 6,
        in_channels: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 300,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        # +1 for the CLS token slot
        self.register_buffer("pe", sinusoidal_positional_encoding(seq_len + 1, d_model), persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 300) → (B, 300, 6) → (B, 300, d_model)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        # Prepend learnable CLS token: (B, 301, d_model)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # Add positional encoding
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        # Encode
        x = self.encoder(x)
        # Take [CLS] (position 0)
        x = self.ln(x[:, 0])
        return self.fc(x)
