import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encoder import EncoderBlock
from .beamformer_attention import DistanceBias


class UpSample(nn.Module):
    def __init__(self, dim_in, dim_out, factor=2):
        super().__init__()
        self.factor = factor
        self.norm = nn.RMSNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_out * factor, bias=False)

    def forward(self, x):
        B, C, N, D = x.shape
        h = self.proj(self.norm(x))
        return h.reshape(B, C, N * self.factor, -1)


class PatchHead(nn.Module):
    def __init__(self, dim, patch=20, hidden=64, kernel=9, groups=8):
        super().__init__()
        assert kernel % 2 == 1
        assert hidden % groups == 0
        self.patch, self.hidden = patch, hidden

        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, hidden * patch, bias=False)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.out = nn.Conv1d(hidden, 1, kernel, padding=kernel // 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, N, D = x.shape
        h = self.proj(self.norm(x)).reshape(B, C, N, self.hidden, self.patch)
        h = h.permute(0, 1, 3, 2, 4).reshape(B * C, self.hidden, N * self.patch)
        y = self.out(self.conv(h)).reshape(B, C, N * self.patch)
        return torch.sign(y) * torch.expm1(y.abs().clamp(max=12.0))


class GaussianLatent(nn.Module):
    def __init__(self, dim, z_dim=None, logvar_range=(-8.0, 8.0)):
        super().__init__()
        self.z_dim = z_dim or dim
        self.logvar_range = logvar_range
        self.to_stats = nn.Linear(dim, 2 * self.z_dim, bias=False)

    def forward(self, x, sample=None):
        mu, logvar = self.to_stats(x).chunk(2, dim=-1)
        logvar = logvar.clamp(*self.logvar_range)
        if sample is None:
            sample = self.training
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return z, mu, logvar, kl


class Decoder(nn.Module):
    def __init__(
        self,
        pos,
        dims=(128, 192, 256, 384),
        depths=(2, 2, 3, 3),
        spatial=("conv", "conv", "beamformer", "beamformer"),
        n_heads=8,
        pool=20,
        z_dim=None,
        drop_path=0.0,
        dropout=0.0,
        n_dist_bins=16,
        patch_hidden=64,
        patch_kernel=9,
        grad_checkpointing=False,
    ):
        super().__init__()
        assert len(dims) == len(depths) == len(spatial)
        C = pos.shape[0]
        dims = tuple(reversed(dims))
        depths = tuple(reversed(depths))
        spatial = tuple(reversed(spatial))

        self.n_channels, self.dims, self.pool = C, dims, pool
        self.factor = 2 ** (len(dims) - 1)
        self.grad_checkpointing = grad_checkpointing

        z_dim = z_dim or dims[0]
        self.inp = nn.Linear(z_dim, dims[0], bias=False) if z_dim != dims[0] else nn.Identity()
        self.dist_bias = DistanceBias(pos, n_heads, n_bins=n_dist_bins)

        dpr = torch.linspace(drop_path, 0, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()

        i = 0
        for lvl, (d, n, sp) in enumerate(zip(dims, depths, spatial)):
            self.stages.append(nn.ModuleList([
                EncoderBlock(C, d, n_heads, spatial=sp,
                             drop_path=dpr[i + j], dropout=dropout)
                for j in range(n)
            ]))
            i += n
            if lvl < len(dims) - 1:
                self.ups.append(UpSample(d, dims[lvl + 1]))

        self.head = PatchHead(dims[-1], patch=pool, hidden=patch_hidden,
                              kernel=patch_kernel)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def out_length(self, n_tokens):
        return n_tokens * self.factor * self.pool

    def forward(self, z, presence=None, length=None):
        B, C, N, _ = z.shape
        assert C == self.n_channels, f"attendu {self.n_channels} canaux, reçu {C}"
        if presence is None:
            presence = torch.ones(B, C, dtype=torch.long, device=z.device)

        h = self.inp(z)
        bias = self.dist_bias()
        kpm = (presence == 0)

        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()

        for lvl, stage in enumerate(self.stages):
            for blk in stage:
                if use_ckpt:
                    h = checkpoint(blk, h, bias, kpm, use_reentrant=False)
                else:
                    h = blk(h, bias, kpm)
            if lvl < len(self.ups):
                h = self.ups[lvl](h)

        y = self.head(h)
        if length is not None and y.shape[-1] != length:
            y = (y[..., :length] if y.shape[-1] > length
                 else F.pad(y, (0, length - y.shape[-1])))
        return y
