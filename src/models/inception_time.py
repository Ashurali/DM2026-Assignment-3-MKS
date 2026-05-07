"""InceptionTime (Fawaz et al. 2020) for raw 6×300 sequences.

Multi-scale parallel-branch architecture purpose-built for time-series
classification. Often beats CNN/LSTM on the UCR/UEA benchmarks; different
inductive bias from CNN-BiLSTM (multi-scale parallel) and Transformer
(global attention) → real ensemble decorrelation.

Architecture:
    InceptionModule:
      - Bottleneck Conv1d(1×1, in→32) reducing channels
      - Three parallel Conv1d with kernels [11, 21, 41] (odd for clean padding)
      - MaxPool branch + Conv1d(1×1)
      - Concat all 4 branches → 4 × n_filters channels
      - BatchNorm + ReLU

    InceptionTime:
      - 6 inception modules
      - Residual connection every 3 modules (skip input → out + ReLU)
      - GlobalAveragePool1D
      - Linear classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (11, 21, 41),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            bt_ch = bottleneck_channels
        else:
            bt_ch = in_channels

        # Conv branches with same-size output via odd-kernel padding=(k-1)//2.
        self.branches = nn.ModuleList([
            nn.Conv1d(bt_ch, n_filters, kernel_size=k, padding=(k - 1) // 2, bias=False)
            for k in kernel_sizes
        ])

        # Pool branch: MaxPool(3) → Conv1×1
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )

        out_ch = (len(kernel_sizes) + 1) * n_filters
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bottleneck:
            x_bt = self.bottleneck(x)
        else:
            x_bt = x
        branches = [b(x_bt) for b in self.branches]
        branches.append(self.pool_branch(x))
        out = torch.cat(branches, dim=1)
        return self.act(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        n_classes: int = 6,
        n_filters: int = 32,
        depth: int = 6,
        residual_every: int = 3,
        kernel_sizes: tuple[int, int, int] = (11, 21, 41),
        bottleneck_channels: int = 32,
    ):
        super().__init__()
        out_ch = (len(kernel_sizes) + 1) * n_filters  # 4 × n_filters
        self.modules_list = nn.ModuleList()
        self.residual_every = residual_every

        cur_in = in_channels
        for i in range(depth):
            mod = InceptionModule(
                in_channels=cur_in,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True,
            )
            self.modules_list.append(mod)
            cur_in = out_ch  # all subsequent modules see out_ch

        # Residual projections — at every `residual_every` modules, project the
        # block-input channels into the block-output channels for the skip add.
        self.res_projections = nn.ModuleList()
        n_residual_blocks = depth // residual_every
        for blk in range(n_residual_blocks):
            in_ch = in_channels if blk == 0 else out_ch
            self.res_projections.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            ))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual_input = x
        for i, mod in enumerate(self.modules_list):
            x = mod(x)
            # After every `residual_every` modules, add a residual from the
            # block start.
            if (i + 1) % self.residual_every == 0:
                blk = (i + 1) // self.residual_every - 1
                res = self.res_projections[blk](residual_input)
                x = F.relu(x + res, inplace=True)
                residual_input = x  # next block's input

        x = self.gap(x).squeeze(-1)  # (B, out_ch)
        return self.fc(x)
