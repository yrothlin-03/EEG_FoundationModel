import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    def __init__(self, beta=1.0, normalize_targets=True):
        super().__init__()
        self.beta = beta
        self.normalize_targets = normalize_targets

    def forward(self, pred, target, mask, valid=None):
        m = mask
        if valid is not None:
            m = m & valid.unsqueeze(-1)
        m = m.unsqueeze(-1).to(pred.dtype)
        denom = m.sum()
        if denom == 0:
            return pred.sum() * 0.0
        if self.normalize_targets:
            target = F.layer_norm(target, (target.shape[-1],))
            pred = F.layer_norm(pred, (pred.shape[-1],))
        l = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        return (l * m).sum() / denom / pred.shape[-1]


class SpectralLoss(nn.Module):
    def __init__(self, n_fft_list=(256, 512, 1024), hop_div=4, fs=200, f_max=45.0):
        super().__init__()
        self.n_fft_list = list(n_fft_list)
        self.hops = [n // hop_div for n in self.n_fft_list]
        self.fs = fs
        self.keep = [int(f_max / (fs / n)) + 1 for n in self.n_fft_list]
        for i, n in enumerate(self.n_fft_list):
            self.register_buffer(f"win_{i}", torch.hann_window(n), persistent=False)

    @torch.no_grad()
    def targets(self, x):
        B, C, T = x.shape
        xf = x.reshape(B * C, T)
        out = []
        for i, (n_fft, hop, k) in enumerate(zip(self.n_fft_list, self.hops, self.keep)):
            w = getattr(self, f"win_{i}").to(xf.dtype)
            S = torch.stft(xf, n_fft=n_fft, hop_length=hop, window=w,
                           return_complex=True, center=True)
            mag = S.abs()[:, :k].transpose(1, 2)
            out.append(torch.log1p(mag).reshape(B, C, mag.shape[1], k))
        return out

    def forward(self, preds, targets, valid=None):
        assert len(preds) == len(targets), (
            f"{len(preds)} resolutions predites contre {len(targets)} cibles : "
            f"la tete spectrale et la loss n'ont pas le meme n_fft_list"
        )
        total = preds[0].sum() * 0.0
        for p, t in zip(preds, targets):
            n = min(p.shape[2], t.shape[2])
            l = (p[:, :, :n] - t[:, :, :n]).abs()
            if valid is None:
                total = total + l.mean()
            else:
                m = valid[:, :, None, None].to(l.dtype)
                total = total + (l * m).sum() / (m.sum() * l.shape[-1] * n).clamp(min=1)
        return total / len(preds)


class PhysioLoss(nn.Module):
    BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

    def __init__(self, fs=200, n_fft=512, n_feats=10, momentum=0.01, beta=1.0):
        super().__init__()
        self.fs, self.n_fft, self.beta = fs, n_fft, beta
        self.momentum = momentum
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("run_mu", torch.zeros(n_feats))
        self.register_buffer("run_var", torch.ones(n_feats))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def targets(self, x, eps=1e-8):
        B, C, T = x.shape
        xf = x.reshape(B * C, T)
        w = self.win.to(xf.dtype)
        S = torch.stft(xf, n_fft=self.n_fft, hop_length=self.n_fft // 4,
                       window=w, return_complex=True, center=True)
        psd = S.abs().pow(2).mean(-1)
        freqs = torch.linspace(0, self.fs / 2, psd.shape[1], device=x.device)

        bp = torch.stack([
            psd[:, (freqs >= lo) & (freqs < hi)].mean(-1).clamp(min=eps).log()
            for lo, hi in self.BANDS
        ], -1)

        fit = (freqs >= 2.0) & (freqs <= 40.0)
        lf = freqs[fit].clamp(min=eps).log()
        lp = psd[:, fit].clamp(min=eps).log()
        lfc = lf - lf.mean()
        slope = (lfc * (lp - lp.mean(-1, keepdim=True))).sum(-1) / lfc.pow(2).sum()
        offset = lp.mean(-1)

        d1 = torch.diff(xf, dim=-1)
        d2 = torch.diff(d1, dim=-1)
        v0, v1, v2 = xf.var(-1) + eps, d1.var(-1) + eps, d2.var(-1) + eps
        mob = (v1 / v0).sqrt()
        comp = (v2 / v1).sqrt() / mob.clamp(min=eps)

        feats = torch.cat([bp, slope[:, None], offset[:, None], v0.log()[:, None],
                           mob.clamp(min=eps).log()[:, None],
                           comp.clamp(min=eps).log()[:, None]], -1)
        return feats.reshape(B, C, -1)

    @torch.no_grad()
    def _update_stats(self, t, valid):
        flat = t.reshape(-1, t.shape[-1])
        if valid is not None:
            flat = flat[valid.reshape(-1)]
        if flat.numel() == 0:
            return
        mu, var = flat.mean(0), flat.var(0, unbiased=False)
        if not self.initialized:
            self.run_mu.copy_(mu)
            self.run_var.copy_(var)
            self.initialized.fill_(True)
        else:
            self.run_mu.mul_(1 - self.momentum).add_(self.momentum * mu)
            self.run_var.mul_(1 - self.momentum).add_(self.momentum * var)

    def forward(self, pred, target, valid=None):
        if self.training:
            self._update_stats(target, valid)
        t = (target - self.run_mu) / self.run_var.clamp(min=1e-4).sqrt()
        l = F.smooth_l1_loss(pred, t, beta=self.beta, reduction="none")
        if valid is None:
            return l.mean()
        m = valid[:, :, None].to(l.dtype)
        return (l * m).sum() / (m.sum() * l.shape[-1]).clamp(min=1)


class SIGRegLoss(nn.Module):
    def __init__(self, n_slices=128, n_knots=17, t_max=4.0, max_samples=8192):
        super().__init__()
        self.n_slices, self.max_samples = n_slices, max_samples
        self.register_buffer("t", torch.linspace(-t_max, t_max, n_knots))
        self.register_buffer("target_cf", (-0.5 * torch.linspace(-t_max, t_max, n_knots).pow(2)).exp())

    def forward(self, z, valid=None):
        if z.dim() == 4 and valid is not None:
            z = z[valid]
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] > self.max_samples:
            idx = torch.randperm(z.shape[0], device=z.device)[:self.max_samples]
            z = z[idx]
        if z.shape[0] < 32:
            return z.sum() * 0.0

        N, D = z.shape
        u = F.normalize(torch.randn(D, self.n_slices, device=z.device, dtype=z.dtype), dim=0)
        s = (z - z.mean(0, keepdim=True)) @ u
        s = s / s.std().clamp(min=1e-6)

        arg = s.unsqueeze(-1) * self.t
        re, im = arg.cos().mean(0), arg.sin().mean(0)
        diff = (re - self.target_cf).pow(2) + im.pow(2)
        return torch.trapz(diff * self.target_cf, self.t, dim=-1).mean()






class PretrainingLoss(nn.Module):
    def __init__(self, weights=None, adaptive=True, warmup_steps=2000,
                 jepa_warmup_frac=0.05, total_steps=100_000,
                 sigreg_token_weight=0.1, ema_momentum=0.01,
                 max_weight_ratio=4.0, **kw):
        super().__init__()
        self.jepa = JEPALoss(**kw.get("jepa", {}))
        self.spectral = SpectralLoss(**kw.get("spectral", {}))
        self.physio = PhysioLoss(**kw.get("physio", {}))
        self.sigreg = SIGRegLoss(**kw.get("sigreg", {}))

        w = {"jepa": 1.0, "spectral": 0.5, "physio": 0.1, "sigreg": 0.1}
        w.update(weights or {})
        self.base_w = w
        self.keys = ["jepa", "spectral", "physio", "sigreg"]

        self.adaptive = adaptive
        self.ema_momentum = ema_momentum
        self.sigreg_token_weight = sigreg_token_weight
        self.jepa_warmup = int(jepa_warmup_frac * total_steps)
        self.warmup_steps = warmup_steps
        self.max_weight_ratio = max_weight_ratio

        self.register_buffer("ema", torch.ones(len(self.keys)))
        self.register_buffer("ema_ref", torch.ones(len(self.keys)))
        self.register_buffer("ema_init", torch.zeros(len(self.keys), dtype=torch.bool))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def _jepa_ramp(self):
        s = int(self.step.item())
        if s >= self.jepa_warmup:
            return 1.0
        return 0.5 * (1 - math.cos(math.pi * s / max(self.jepa_warmup, 1)))

    @torch.no_grad()
    def _scale(self, i, value):
        if not self.adaptive:
            return 1.0
        v = value.detach().abs().clamp(min=1e-8)
        if not self.ema_init[i]:
            if not self.training:
                return (1.0 / v).item()
            self.ema[i] = v
            self.ema_ref[i] = v
            self.ema_init[i] = True
        elif self.training:
            self.ema[i].mul_(1 - self.ema_momentum).add_(self.ema_momentum * v)
        r = max(1.0, self.max_weight_ratio)
        ref = self.ema_ref[i].clamp(min=1e-8)
        denom = self.ema[i].clamp(min=ref / r, max=ref * r)
        return (1.0 / denom.clamp(min=1e-8)).item()

    def forward(self, outputs, batch):
        valid = batch.get("valid")
        raw = {}

        if "jepa_pred" in outputs:
            raw["jepa"] = self.jepa(outputs["jepa_pred"], outputs["z_target"],
                                    batch["mask"], valid)

        if "spectral_pred" in outputs:
            spec_t = self.spectral.targets(batch["x_clean"])
            raw["spectral"] = self.spectral(outputs["spectral_pred"], spec_t, valid)

        if "physio_pred" in outputs:
            phys_t = self.physio.targets(batch["x_clean"])
            raw["physio"] = self.physio(outputs["physio_pred"], phys_t, valid)

        sig = self.sigreg(outputs["z_glob"])
        if self.sigreg_token_weight > 0:
            sig = sig + self.sigreg_token_weight * self.sigreg(outputs["z"], valid)
        raw["sigreg"] = sig

        total = 0.0
        logs = {}
        for i, k in enumerate(self.keys):
            if k not in raw:
                continue
            v = raw[k]
            w = self.base_w[k] * self._scale(i, v)
            if k == "jepa":
                w = w * self._jepa_ramp()
            total = total + w * v
            logs[f"loss/{k}"] = v.detach()
            logs[f"weight/{k}"] = float(w)

        if self.training:
            self.step += 1
        logs["loss/total"] = total.detach()
        return total, logs

    @torch.no_grad()
    def grad_norms(self, model):
        out = {}
        for name, mod in [("encoder", model.encoder)]:
            g = [p.grad.norm() ** 2 for p in mod.parameters() if p.grad is not None]
            out[f"gradnorm/{name}"] = torch.stack(g).sum().sqrt().item() if g else 0.0
        return out

    