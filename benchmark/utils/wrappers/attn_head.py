import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAttentionPool(nn.Module):
    def __init__(self, dim, n_classes=None, n_heads=8, n_queries=1, dropout=0.1):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.n_queries = n_queries
        self.scale = self.head_dim ** -0.5
        self.query = nn.Parameter(torch.randn(1, n_heads, n_queries, self.head_dim) * 0.02)
        self.norm = nn.RMSNorm(dim)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.drop = nn.Dropout(dropout)
        self.head = (
            nn.Sequential(
                nn.RMSNorm(dim * n_queries),
                nn.Dropout(dropout),
                nn.Linear(dim * n_queries, n_classes),
            )
            if n_classes is not None
            else nn.Identity()
        )

    def forward(self, z, mask=None, return_attn=False):
        *lead, D = z.shape
        B, spatial = lead[0], lead[1:]
        S = int(torch.tensor(spatial).prod().item()) if spatial else 1
        H, hd, Q = self.n_heads, self.head_dim, self.n_queries

        h = self.norm(z).reshape(B, S, D)
        k, v = self.to_kv(h).chunk(2, dim=-1)
        k = self.k_norm(k.view(B, S, H, hd).transpose(1, 2))
        v = v.view(B, S, H, hd).transpose(1, 2)

        scores = (self.query * self.scale) @ k.transpose(-1, -2)
        if mask is not None:
            m = mask
            while m.dim() < len(lead):
                m = m.unsqueeze(-1)
            m = m.expand(*lead).reshape(B, 1, 1, S)
            scores = scores.masked_fill(~m, float("-inf"))
        attn = self.drop(scores.softmax(-1))

        pooled = (attn @ v).transpose(1, 2).reshape(B, Q * D)
        out = self.head(pooled)
        if return_attn:
            return out, attn.reshape(B, H, Q, *spatial) if spatial else attn
        return out



class AttnHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        d_model: int,
        n_heads: int = 8,
        n_queries: int = 1,
        dropout: float = 0.1,
        channels_last: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.channels_last = channels_last
        self.pool = TaskAttentionPool(
            d_model,
            n_classes=n_classes,
            n_heads=n_heads,
            n_queries=n_queries,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask=None, return_attn: bool = False):
        if not self.channels_last:
            x = x.movedim(1, -1)
        return self.pool(x, mask=mask, return_attn=return_attn)