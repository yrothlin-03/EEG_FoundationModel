import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, num_classes, z_channels, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        if pooling == "none":
            # flatten -> feature count is z_channels * seq_len, unknown until forward
            self.norm = nn.LazyBatchNorm1d()
            self.head = nn.LazyLinear(num_classes)
        else:
            in_features = z_channels * 2 if pooling == "mean_std" else z_channels
            self.norm = nn.BatchNorm1d(in_features)
            self.head = nn.Linear(in_features, num_classes)

    def pool(self, z):
        if self.pooling == "mean":
            return z.mean(dim=-1)
        if self.pooling == "max":
            return z.amax(dim=-1)
        if self.pooling == "mean_std":
            return torch.cat([z.mean(dim=-1), z.std(dim=-1)], dim=-1)
        if self.pooling == "none":
            return z.flatten(start_dim=1)
        raise ValueError(f"unknown pooling: {self.pooling!r}")

    def features(self, z):
        return self.norm(self.pool(z))

    def forward(self, z):
        return self.head(self.features(z))
