import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import Encoder
from .modules.stem import Stem
from .jepa_head import JEPAPredictor
from .spectral_head import SpectralHead, MultiResSpectralHead
from .physio_head import PhysioHead


class EEGFM(nn.Module):
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
        window=4000,
        z_dim=None,
        drop_path=0.15,
        dropout=0.0,
        n_dist_bins=16,
        predictor_depth=2,
        predictor_ratio=0.5,
        n_fft_list=(256, 512, 1024),
        hop_div=4,
        f_max=45.0,
        n_physio_feats=10,
        use_jepa=True,
        use_spectral=True,
        use_physio=True,
        detach_target=True,
        use_ema_target=True,
    ):
        super().__init__()
        self.fs = fs
        self.pool = pool
        self.window = window
        self.n_channels = pos.shape[0]
        self.detach_target = detach_target
        self.use_ema_target = use_ema_target

        self.encoder = Encoder(
            pos,
            dims=dims,
            depths=depths,
            spatial=spatial,
            n_heads=n_heads,
            n_bands=n_bands,
            kernel_size=kernel_size,
            fs=fs,
            pool=pool,
            z_dim=z_dim,
            drop_path=drop_path,
            dropout=dropout,
            n_dist_bins=n_dist_bins,
        )
        self.z_dim = z_dim or dims[-1]
        self.n_tokens = self._token_count(window, depths)

        self.target_encoder = None
        if use_ema_target:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.target_encoder.eval()

        self.jepa_predictor = (
            JEPAPredictor(
                self.n_channels,
                self.z_dim,
                pred_dim=int(self.z_dim * predictor_ratio),
                depth=predictor_depth,
                n_heads=n_heads,
            )
            if use_jepa
            else None
        )

        self.spectral_head = (
            MultiResSpectralHead(
                self.z_dim,
                n_fft_list=n_fft_list,
                hop_div=hop_div,
                T=window,
                fs=fs,
                f_max=f_max,
            )
            if use_spectral
            else None
        )

        self.physio_head = (
            PhysioHead(self.z_dim, n_feats=n_physio_feats) if use_physio else None
        )

    def train(self, mode=True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum):
        if self.target_encoder is None:
            return
        for pt, ps in zip(self.target_encoder.parameters(),
                          self.encoder.parameters()):
            pt.lerp_(ps.detach(), 1.0 - momentum)
        for bt, bs in zip(self.target_encoder.buffers(),
                          self.encoder.buffers()):
            bt.copy_(bs)

    def _token_count(self, T, depths):
        n = T // self.pool
        for _ in range(len(depths) - 1):
            n //= 2
        return n

    def token_stride(self):
        return self.window // self.n_tokens

    def _default_presence(self, x):
        return torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)

    def expand_mask(self, mask):
        return mask.repeat_interleave(self.token_stride(), dim=-1)[..., : self.window]

    def apply_mask(self, x, mask_tokens):
        m = self.expand_mask(mask_tokens)
        if m.shape[-1] < x.shape[-1]:
            m = F.pad(m, (0, x.shape[-1] - m.shape[-1]), value=False)
        return x.masked_fill(m, 0.0), m

    def _context_presence(self, presence, mask_tokens):
        fully_masked = mask_tokens.all(dim=-1) & (presence != 0)
        return presence.masked_fill(fully_masked, 0)

    def encode(self, x, presence=None, return_hidden=False):
        if presence is None:
            presence = self._default_presence(x)
        return self.encoder(x, presence, return_hidden=return_hidden)

    def forward_context(self, x, presence, mask_tokens):
        x_masked, mask_samples = self.apply_mask(x, mask_tokens)
        presence_ctx = self._context_presence(presence, mask_tokens)
        out = self.encoder(x_masked, presence_ctx)
        out["mask_samples"] = mask_samples
        out["presence_ctx"] = presence_ctx
        return out

    def forward_target(self, x, presence):
        if self.target_encoder is not None:
            with torch.no_grad():
                out = self.target_encoder(x, presence)
            return {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}
        if self.detach_target:
            with torch.no_grad():
                out = self.encoder(x, presence)
            return {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}
        return self.encoder(x, presence)

    def forward_batched(self, x, presence, mask_tokens):
        if self.target_encoder is not None or self.detach_target:
            return (
                self.forward_context(x, presence, mask_tokens),
                self.forward_target(x, presence),
            )
        x_masked, mask_samples = self.apply_mask(x, mask_tokens)
        presence_ctx = self._context_presence(presence, mask_tokens)

        xx = torch.cat([x_masked, x], dim=0)
        pp = torch.cat([presence_ctx, presence], dim=0)
        out = self.encoder(xx, pp)

        B = x.shape[0]
        ctx = {k: v[:B] for k, v in out.items() if torch.is_tensor(v)}
        tgt = {k: v[B:] for k, v in out.items() if torch.is_tensor(v)}
        ctx["mask_samples"] = mask_samples
        ctx["presence_ctx"] = presence_ctx
        return ctx, tgt

    def forward(self, x, presence=None, mask=None, batched=True):
        if presence is None:
            presence = self._default_presence(x)
        valid = presence != 0

        if mask is None:
            mask = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.n_tokens,
                dtype=torch.bool,
                device=x.device,
            )
        assert mask.shape[-1] == self.n_tokens, (
            f"mask attendu à la résolution token ({self.n_tokens}), "
            f"reçu {mask.shape[-1]}"
        )
        mask = mask & valid.unsqueeze(-1)

        if batched:
            ctx, tgt = self.forward_batched(x, presence, mask)
        else:
            ctx = self.forward_context(x, presence, mask)
            tgt = self.forward_target(x, presence)

        out = {
            "z": ctx["z"],
            "z_glob": ctx["z_glob"],
            "z_target": tgt["z"],
            "z_glob_target": tgt["z_glob"],
            "mask": mask,
            "valid": valid,
        }

        if self.jepa_predictor is not None:
            out["jepa_pred"] = self.jepa_predictor(ctx["z"], mask)
        if self.spectral_head is not None:
            out["spectral_pred"] = self.spectral_head(ctx["z"])
        if self.physio_head is not None:
            out["physio_pred"] = self.physio_head(ctx["z"])

        return out

    @torch.no_grad()
    def embed(self, x, presence=None, level="global", hop=None, reduce="mean_max_std"):
        B, C, T = x.shape
        win = self.window
        hop = hop or win // 2
        if presence is None:
            presence = self._default_presence(x)

        if T < win:
            x = F.pad(x, (0, win - T))
            T = win

        starts = list(range(0, max(T - win, 0) + 1, hop))
        if starts[-1] + win < T:
            starts.append(T - win)

        chunks = []
        for s in starts:
            out = self.encoder(x[..., s : s + win], presence)
            chunks.append(out["z_glob"] if level == "global" else out["z"])
        z = torch.stack(chunks, dim=1)

        if level != "global":
            return z
        if reduce == "mean":
            return z.mean(1)
        if reduce == "max":
            return z.amax(1)
        if reduce == "mean_max_std":
            return torch.cat([z.mean(1), z.amax(1), z.std(1, unbiased=False)], dim=-1)
        raise ValueError(f"reduce inconnu: {reduce}")

    def strip_heads(self):
        self.jepa_predictor = None
        self.spectral_head = None
        self.physio_head = None
        return self

    def no_weight_decay(self):
        keys = (
            "mask_token",
            "dist_bias.table",
            "log_fc",
            "log_sigma",
            "_emb",
            "query",
            "norm",
            "ls_s",
            "ls_t",
            "ls_c",
            "ls_f",
        )
        return {n for n, _ in self.named_parameters() if any(k in n for k in keys)}

    def param_groups(self, weight_decay=0.05):
        skip = self.no_weight_decay()
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if n in skip or p.ndim <= 1 else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def diagnostics(self):
        d = self.encoder.diagnostics()
        d["n_tokens"] = self.n_tokens
        d["token_stride"] = self.token_stride()
        return d