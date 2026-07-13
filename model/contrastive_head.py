import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, z):
        z = z.transpose(1, 2)
        weights = F.softmax(self.score(z), dim=1)
        return (weights * z).sum(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, z_channels, clip_dim, hidden_dim=512):
        super().__init__()
        self.pool = AttentionPool1d(z_channels)
        self.net = nn.Sequential(
            nn.BatchNorm1d(z_channels, track_running_stats=False),
            nn.Linear(z_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim),
            nn.BatchNorm1d(clip_dim, affine=False, track_running_stats=False),
        )

    def forward(self, z):
        return self.net(self.pool(z))
