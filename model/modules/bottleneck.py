import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, dim, n_heads=8, z_dim=None, dropout=0.0, n_queries=4):
        super().__init__()
        z_dim = z_dim or dim
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, z_dim, bias=False) if z_dim != dim else nn.Identity()

        self.n_heads = n_heads
        self.n_queries = n_queries
        self.head_dim = z_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Parameter(
            torch.randn(1, n_heads, n_queries, self.head_dim) * 0.02
        )
        self.to_kv = nn.Linear(z_dim, 2 * z_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.merge_norm = nn.RMSNorm(n_queries * z_dim)
        self.merge = nn.Linear(n_queries * z_dim, z_dim, bias=False)
        self.out = nn.Linear(z_dim, z_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        B, C, N, D = x.shape
        z = self.proj(self.norm(x))
        Dz = z.shape[-1]
        H, hd = self.n_heads, self.head_dim

        flat = z.reshape(B, C * N, Dz)
        k, v = self.to_kv(flat).chunk(2, dim=-1)
        k = self.k_norm(k.view(B, C * N, H, hd).transpose(1, 2))
        v = v.view(B, C * N, H, hd).transpose(1, 2)

        scores = (self.query * self.scale) @ k.transpose(-1, -2)
        if key_padding_mask is not None:
            m = key_padding_mask.unsqueeze(-1).expand(B, C, N).reshape(B, 1, 1, C * N)
            scores = scores.masked_fill(m, float("-inf"))
        attn = self.drop(scores.softmax(dim=-1))

        pooled = (attn @ v).transpose(1, 2).reshape(B, self.n_queries * Dz)
        pooled = self.merge(self.merge_norm(pooled))
        return z, self.out(pooled)