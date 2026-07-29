import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


class PhysioHead(nn.Module):
    def __init__(self, dim, n_feats=10, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.RMSNorm(dim), nn.Linear(dim, hidden, bias=False),
            nn.SiLU(), nn.Linear(hidden, n_feats),
        )

    def forward(self, z):
        return self.net(z.mean(dim=2))


@torch.no_grad()
def physio_targets(x, fs=200, n_fft=512, eps=1e-8):
    B, C, T = x.shape
    xf = x.reshape(B * C, T)
    w = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    S = torch.stft(xf, n_fft=n_fft, hop_length=n_fft // 4, window=w,
                   return_complex=True, center=True)
    psd = S.abs().pow(2).mean(-1)
    freqs = torch.linspace(0, fs / 2, psd.shape[1], device=x.device)

    bp = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp.append(psd[:, m].mean(-1).clamp(min=eps).log())
    bp = torch.stack(bp, -1)

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
    comp = (v2 / v1).sqrt() / mob

    feats = torch.cat([bp, slope[:, None], offset[:, None],
                       v0.log()[:, None], mob.log()[:, None],
                       comp.log()[:, None]], -1)
    return feats.reshape(B, C, -1)


def physio_loss(pred, target, valid=None, running_stats=None):
    if running_stats is not None:
        mu, sd = running_stats
        target = (target - mu) / sd.clamp(min=1e-3)
    l = F.smooth_l1_loss(pred, target, beta=1.0, reduction="none")
    if valid is None:
        return l.mean()
    m = valid[:, :, None].to(l.dtype)
    return (l * m).sum() / m.sum().clamp(min=1) / l.shape[-1]