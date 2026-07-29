import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class GaborFilterbank(nn.Module):
    def __init__(self, n_filters=32, K=65, fs=200, f_min=0.5, f_max=45.0,
                 n_cycles=5.0, learn_sigma=True):
        super().__init__()
        assert K % 2 == 1
        self.fs, self.K = fs, K
        f0 = torch.logspace(math.log10(f_min), math.log10(f_max), n_filters)
        self.log_fc = nn.Parameter(f0.log())
        sigma0 = f0 / n_cycles
        self.log_sigma = nn.Parameter(sigma0.log(), requires_grad=learn_sigma)
        self.register_buffer("t", (torch.arange(K) - K // 2).float() / fs)

    def kernels(self):
        fc = self.log_fc.exp().clamp(0.1, self.fs / 2 - 1)
        sigma = self.log_sigma.exp().clamp(min=1e-3)
        env = torch.exp(-0.5 * (self.t[None] * sigma[:, None]) ** 2)
        phase = 2 * math.pi * fc[:, None] * self.t[None]
        real = env * phase.cos()
        imag = env * phase.sin()
        real = real - real.mean(-1, keepdim=True)
        norm = torch.sqrt(real.pow(2).sum(-1, keepdim=True)
                          + imag.pow(2).sum(-1, keepdim=True)).clamp(min=1e-8)
        return real / norm, imag / norm

    def forward(self, x):
        B, C, T = x.shape
        real, imag = self.kernels()
        w = torch.cat([real, imag], 0).unsqueeze(1)
        y = F.conv1d(x.reshape(B * C, 1, T), w, padding=self.K // 2)
        yr, yi = y.chunk(2, dim=1)
        mag = (yr.pow(2) + yi.pow(2)).clamp(min=1e-8).sqrt()
        return mag.reshape(B, C, -1, T)

    @torch.no_grad()
    def frequencies(self):
        return self.log_fc.exp().sort().values