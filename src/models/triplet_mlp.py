"""Small MLP that maps existing CNN-BiLSTM embeddings (256-d) to a new
L1↔L2-discriminating embedding space (default 64-d).

Trained with triplet loss + hard negative mining:
  anchor   = L2 sample
  positive = another L2 sample
  negative = closest L1 sample to anchor (hard negative)
  loss     = max(0, ||emb(a)-emb(p)||² - ||emb(a)-emb(n)||² + margin)

L2-normalised output so distances are angular.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMLP(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, emb_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)
