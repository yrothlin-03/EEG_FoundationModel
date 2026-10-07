import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .utils import ConformerConv, SwiGLU, DropPath, SpatialConv, DownSample
from .beamformer_attention import BeamformerAttention, DistanceBias
from .mamba_layer import BiMamba2Temporal
from .stem import Stem
from .tokenizer import PatchStem
from .bottleneck import Bottleneck



class EncoderBlock(nn.Module):
    def __init__(self, n_channels, dim, n_heads, spatial="beamformer",
                 drop_path=0.15, ls_init=1e-4, dropout=0.0):
        super().__init__()
        assert spatial in ("conv", "beamformer")
        self.spatial_mode = spatial

        self.norm_s = nn.RMSNorm(dim)
        if spatial == "beamformer":
            self.attn = BeamformerAttention(dim, n_heads, dropout=dropout)
        self.spatial_conv = SpatialConv(n_channels, dim)

        self.norm_t = nn.RMSNorm(dim)
        self.temporal = BiMamba2Temporal(dim, d_state=128, headdim=64, dropout=dropout)

        self.norm_c = nn.RMSNorm(dim)
        self.conv_mod = ConformerConv(dim, kernel_size=15, dropout=dropout)

        self.norm_f = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

        self.ls_s = nn.Parameter(ls_init * torch.ones(dim))
        self.ls_t = nn.Parameter(ls_init * torch.ones(dim))
        self.ls_c = nn.Parameter(ls_init * torch.ones(dim))
        self.ls_f = nn.Parameter(ls_init * torch.ones(dim))

        self.dp = DropPath(drop_path)
        self.last_attn = None

    def forward(self, x, dist_bias=None, kpm=None):
        h = self.norm_s(x)
        s = self.spatial_conv(h, kpm)
        if self.spatial_mode == "beamformer":
            a, attn = self.attn(h, dist_bias, kpm)
            self.last_attn = attn.detach() if not self.training else None
            s = s + a
        x = x + self.dp(self.ls_s * s)

        x = x + self.dp(self.ls_t * self.temporal(self.norm_t(x)))
        x = x + self.dp(self.ls_c * self.conv_mod(self.norm_c(x)))
        return x + self.dp(self.ls_f * self.ffn(self.norm_f(x)))





class Encoder(nn.Module):
    def __init__(
        self,
        pos,
        dims=(128, 192, 256, 384),
        depths=(2, 2, 3, 3),
        spatial=("conv", "conv", "beamformer", "beamformer"),
        n_heads=8,
        n_bands=32,
        kernel_size=65,
        fs=200,
        pool=20,
        z_dim=None,
        drop_path=0.15,
        dropout=0.0,
        n_dist_bins=16,
        n_global_queries=4,
        grad_checkpointing=False,
        stem="gabor",
        patch_hidden=64,
        patch_kernel=9,
    ):
        super().__init__()
        assert len(dims) == len(depths) == len(spatial)
        assert stem in ("gabor", "patch")
        C = pos.shape[0]
        self.n_channels, self.dims, self.pool = C, dims, pool
        self.grad_checkpointing = grad_checkpointing
        self.stem_mode = stem

        if stem == "gabor":
            self.stem = Stem(C, dim=dims[0], n_bands=n_bands,
                             K=kernel_size, fs=fs, pool=pool)
        else:
            self.stem = PatchStem(C, dim=dims[0], patch=pool,
                                  hidden=patch_hidden, kernel=patch_kernel)
                                  
        self.dist_bias = DistanceBias(pos, n_heads, n_bins=n_dist_bins)

        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        i = 0
        for lvl, (d, n, sp) in enumerate(zip(dims, depths, spatial)):
            self.stages.append(nn.ModuleList([
                EncoderBlock(C, d, n_heads, spatial=sp,
                             drop_path=dpr[i + j], dropout=dropout)
                for j in range(n)
            ]))
            i += n
            if lvl < len(dims) - 1:
                self.downs.append(DownSample(d, dims[lvl + 1]))

        self.bottleneck = Bottleneck(
            dims[-1], n_heads, z_dim=z_dim, dropout=dropout,
            n_queries=n_global_queries,
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, presence=None, return_hidden=False):
        B, C, T = x.shape
        assert C == self.n_channels, f"attendu {self.n_channels} canaux, reçu {C}"
        if presence is None:
            presence = torch.ones(B, C, dtype=torch.long, device=x.device)

        h = self.stem(x, presence)
        bias = self.dist_bias()
        kpm = (presence == 0)

        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()

        hidden = []
        for lvl, stage in enumerate(self.stages):
            for blk in stage:
                if use_ckpt:
                    h = checkpoint(blk, h, bias, kpm, use_reentrant=False)
                else:
                    h = blk(h, bias, kpm)
            if return_hidden:
                hidden.append(h)
            if lvl < len(self.downs):
                h = self.downs[lvl](h)

        z, z_glob = self.bottleneck(h, kpm)
        out = {"z": z, "z_glob": z_glob}
        if return_hidden:
            out["hidden"] = hidden
        return out

    def no_weight_decay(self):
        skip = set()
        for n, _ in self.named_parameters():
            if any(k in n for k in ("dist_bias.table", "log_fc", "log_sigma",
                                    "ls_s", "ls_t", "ls_c", "ls_f",
                                    "_emb", "query", "norm")):
                skip.add(n)
        return skip

    @torch.no_grad()
    def diagnostics(self):
        out = {"dist_bias": self.dist_bias.table.detach().cpu()}
        if self.stem_mode == "gabor":
            out["gabor_hz"] = self.stem.fb.frequencies().cpu()
        return out