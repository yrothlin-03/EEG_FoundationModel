import math
import torch
import torch.nn as nn


class DistanceBias(nn.Module):
    def __init__(self, pos, n_heads, n_bins=16, geodesic=True):
        super().__init__()
        if geodesic:
            p = pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            d = (p @ p.T).clamp(-1.0, 1.0).acos()
        else:
            d = torch.cdist(pos, pos)
        bins = (d / d.max().clamp(min=1e-8) * (n_bins - 1)).round().long()
        self.register_buffer("bins", bins)
        self.table = nn.Parameter(torch.zeros(n_heads, n_bins))

    def forward(self):
        return self.table[:, self.bins]


class BeamformerAttention(nn.Module):
    def __init__(self, dim, n_heads=8, n_bands=None, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        stat_dim = dim * 2 if n_bands is None else dim + n_bands
        self.stat_norm = nn.LayerNorm(stat_dim)
        self.to_q = nn.Linear(stat_dim, dim, bias=False)
        self.to_k = nn.Linear(stat_dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def _stats(self, x):
        mu = x.mean(dim=2)
        xc = x - mu.unsqueeze(2)
        logpow = xc.pow(2).mean(dim=2).clamp(min=1e-8).log()
        return torch.cat([mu, logpow], dim=-1)

    def forward(self, x, dist_bias=None, key_padding_mask=None):
        B, C, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        s = self.stat_norm(self._stats(x))
        q = self.to_q(s).view(B, C, H, hd).transpose(1, 2)
        k = self.to_k(s).view(B, C, H, hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        scores = (q @ k.transpose(-1, -2)) * self.scale
        if dist_bias is not None:
            scores = scores + dist_bias.unsqueeze(0)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        attn = self.drop(scores.softmax(dim=-1))

        v = self.to_v(x).view(B, C, T, H, hd).permute(0, 3, 1, 2, 4)
        out = torch.einsum("bhij,bhjtd->bhitd", attn, v)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, C, T, D)
        return self.proj(out), attn