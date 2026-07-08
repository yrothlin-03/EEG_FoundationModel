import torch
import torch.nn as nn


class ConvProbe(nn.Module):
    def __init__(self, num_classes, z_channels, hidden=256, kernel_size=3,
                 n_layers=2, dropout=0.0, out_len=4):
        super().__init__()
        layers = []
        in_ch = z_channels
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
        self.head = nn.Linear(hidden * out_len, num_classes)

    def features(self, z):
        h = self.conv(z)
        h = self.pool(h)
        return torch.flatten(h, 1)

    def forward(self, z):
        return self.head(self.features(z))
