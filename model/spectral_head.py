import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralHead(nn.Module):
    def __init__(self, dim, n_freqs, n_frames, hidden=None, n_layers=2):
        super().__init__()
        hidden = hidden or dim
        self.n_frames = n_frames
        self.norm = nn.RMSNorm(dim)
        self.refine = nn.Sequential(*[
            layer for _ in range(n_layers) for layer in (
                nn.Conv1d(dim if _ == 0 else hidden, hidden, 5, padding=2),
                nn.SiLU(),
            )
        ])
        self.out = nn.Linear(hidden, n_freqs)

    def forward(self, z):
        B, C, N, D = z.shape
        h = self.norm(z).reshape(B * C, N, D).transpose(1, 2)
        h = F.interpolate(h, size=self.n_frames, mode="linear", align_corners=False)
        h = self.refine(h).transpose(1, 2)
        return self.out(h).reshape(B, C, self.n_frames, -1)


class MultiResSpectralHead(nn.Module):
    def __init__(self, dim, n_fft_list=(256, 512, 1024), hop_div=4,
                 T=4000, fs=200, f_max=45.0):
        super().__init__()
        self.n_fft_list = list(n_fft_list)
        self.hop_div = hop_div
        self.hops = [n // hop_div for n in self.n_fft_list]
        self.fs = fs
        self.f_max = f_max
        self.keep, self.heads = [], nn.ModuleList()
        for n_fft, hop in zip(self.n_fft_list, self.hops):
            k = int(f_max / (fs / n_fft)) + 1
            frames = T // hop + 1
            self.keep.append(k)
            self.heads.append(SpectralHead(dim, k, frames))
        self.register_buffer("_win_cache", torch.empty(0), persistent=False)

    @torch.no_grad()
    def targets(self, x):
        B, C, T = x.shape
        out = []
        for n_fft, hop, k in zip(self.n_fft_list, self.hops, self.keep):
            w = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            S = torch.stft(x.reshape(B * C, T), n_fft=n_fft, hop_length=hop,
                           window=w, return_complex=True, center=True)
            mag = S.abs()[:, :k].transpose(1, 2)
            out.append(torch.log1p(mag).reshape(B, C, mag.shape[1], k))
        return out

    def forward(self, z):
        return [head(z) for head in self.heads]


def spectral_loss(preds, targets, valid=None):
    total = 0.0
    for p, t in zip(preds, targets):
        n = min(p.shape[2], t.shape[2])
        p, t = p[:, :, :n], t[:, :, :n]
        l = (p - t).abs()
        if valid is not None:
            m = valid[:, :, None, None].to(l.dtype)
            total = total + (l * m).sum() / m.sum().clamp(min=1) / l.shape[-1] / n
        else:
            total = total + l.mean()
    return total / len(preds)