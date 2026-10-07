from __future__ import annotations

import torch
import torch.nn as nn


class ConvHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        d_model: int,
        hidden: int = 256,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.2,
        out_len: int = 4,
        channels_last: bool = True,
    ):
        super().__init__()
        self.channels_last = channels_last
        layers = []
        in_ch = d_model
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, hidden, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(out_len)
        self.proj = nn.Linear(hidden * out_len, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b, c, l, d = x.shape
            x = x.reshape(b, c * l, d)
        if self.channels_last:
            x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)
