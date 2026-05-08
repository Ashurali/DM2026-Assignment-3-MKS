"""DANN-equipped CNN-BiLSTM for cross-subject generalization (Tier B.4).

Domain-Adversarial Neural Network (Ganin et al., JMLR 2016) modification:
the encoder learns features that are useful for activity classification
but uninformative about which user produced them. Achieved by adding a
domain classifier head whose gradient is reversed before flowing back
into the encoder.

The reasoning for our problem:
- Train/test users are disjoint. The model can pick up subject-specific
  shortcuts that don't transfer (we observed this in the t-SNE — files
  cluster by user, not just by class).
- A domain classifier trained to predict user_id will succeed if the
  encoder's features are subject-discriminative.
- The Gradient Reversal Layer (GRL) in front of the domain classifier
  makes the encoder *adversary* of subject-prediction → encoder must
  learn subject-invariant features that still classify activities.
- λ (the GRL multiplier) ramps from 0 → λ_max over training so the early
  feature learning isn't disrupted by adversarial pressure.

References:
- Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016.
- For HAR specifically, e.g., DATTA (arXiv 2411.13284).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function


# ─── Gradient Reversal Layer ────────────────────────────────────────────────
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


# ─── DANN-wrapped CNN-BiLSTM ────────────────────────────────────────────────
class CNNBiLSTM_DANN(nn.Module):
    """CNN-BiLSTM encoder + class head + (separate) domain head.

    Forward returns (class_logits, domain_logits, embedding). The domain head
    receives the embedding through a Gradient Reversal Layer (parameter
    `current_lambda` is updated externally over training).

    `n_domains` = number of distinct subjects in the training fold (= unique
    user_ids in the train slice).
    """

    def __init__(
        self,
        n_classes: int = 6,
        n_domains: int = 60,
        in_channels: int = 6,
        lstm_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Same backbone as CNNBiLSTM
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        # Attention pooling
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)

        emb_dim = lstm_hidden * 2  # 256

        # Class head
        self.fc = nn.Linear(emb_dim, n_classes)

        # Domain head: GRL → 2-layer MLP → user_id logits
        self.domain_head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_domains),
        )

        # GRL multiplier — set externally at each training step via
        # set_lambda(); default 0 so domain head doesn't affect encoder
        # until the trainer enables it.
        self.current_lambda = 0.0

    def set_lambda(self, lambda_: float) -> None:
        self.current_lambda = float(lambda_)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding (B, 2*lstm_hidden)."""
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.lstm(x)
        a = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        emb = torch.sum(x * a.unsqueeze(-1), dim=1)
        emb = self.dropout(emb)
        return emb

    def forward(self, x: torch.Tensor):
        emb = self.encode(x)
        class_logits = self.fc(emb)
        # Domain head sees a gradient-reversed view of the embedding
        domain_input = grad_reverse(emb, self.current_lambda)
        domain_logits = self.domain_head(domain_input)
        return class_logits, domain_logits, emb


def lambda_schedule(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Standard DANN λ ramp: 2/(1+exp(-γ·p)) − 1, scaled by max_lambda.

    progress ∈ [0, 1] (0 at start, 1 at end of training).
    Common gamma=10 starts λ near 0 and ramps to ~max_lambda by end.
    """
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
