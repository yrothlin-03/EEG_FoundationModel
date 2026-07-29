import torch
import torch.nn as nn
from .utils import Norm


try:
    from mamba_ssm import Mamba, Mamba2
except ImportError:
    raise(ImportError("Mamba is not installed. Please install mamba-ssm to use MambaBlock."))


class BiMamba2Temporal(nn.Module):
    def __init__(self, dim, d_state=128, d_conv=4, expand=2, headdim=64, merge="add", dropout=0.0):
        super().__init__()
        assert merge in ("add", "concat")
        assert (dim * expand) % headdim == 0, f"d_inner={dim * expand} doit être divisible par headdim={headdim}"

        self.merge = merge
        self.norm = nn.RMSNorm(dim)

        cfg = dict(d_model=dim, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim)
        self.fwd = Mamba2(**cfg)
        self.bwd = Mamba2(**cfg)

        self.out = nn.Linear(2 * dim, dim, bias=False) if merge == "concat" else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape
        h = self.norm(x).reshape(B * C, N, D)

        yf = self.fwd(h)
        yb = self.bwd(h.flip(1)).flip(1)

        y = yf + yb if self.merge == "add" else self.out(torch.cat([yf, yb], dim=-1))
        return self.drop(y.reshape(B, C, N, D))