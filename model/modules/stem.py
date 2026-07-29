import torch
import torch.nn as nn
import torch.nn.functional as F
from .gabor_filter_bank import GaborFilterbank


class Stem(nn.Module):
    def __init__(self, C, dim=128, n_bands=32, K=65, fs=200, pool=20):
        super().__init__()
        self.fb = GaborFilterbank(n_bands, K, fs)
        self.pool = pool
        self.band_norm = nn.LayerNorm(n_bands)
        self.proj = nn.Linear(n_bands, dim, bias=False)
        self.channel_emb = nn.Embedding(C, dim)
        self.presence_emb = nn.Embedding(3, dim)
        nn.init.normal_(self.channel_emb.weight, std=0.02)
        nn.init.zeros_(self.presence_emb.weight)

    def forward(self, x, presence):
        B, C, T = x.shape
        h = self.fb(x)
        h = torch.log1p(h)
        h = F.avg_pool1d(h.reshape(B * C, -1, T), self.pool, self.pool)
        h = h.reshape(B, C, -1, h.shape[-1]).permute(0, 1, 3, 2)
        h = self.proj(self.band_norm(h))
        h = h + self.channel_emb.weight[None, :, None, :]
        h = h + self.presence_emb(presence)[:, :, None, :]
        return h