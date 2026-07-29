
import torch
import torch.nn as nn


def round_ste(z):
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    def __init__(self, levels, dim=None):
        super().__init__()
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32))
        basis = torch.cat([
            torch.ones(1, dtype=torch.int32),
            torch.tensor(levels[:-1], dtype=torch.int32)
        ]).cumprod(dim=0)
        self.register_buffer("_basis", basis)

        self.codebook_dim = len(levels)
        self.codebook_size = int(torch.tensor(levels).prod().item())
        dim = dim if dim is not None else self.codebook_dim

        self.project_in = (
            nn.Linear(dim, self.codebook_dim)
            if dim != self.codebook_dim else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.codebook_dim, dim)
            if dim != self.codebook_dim else nn.Identity()
        )

    def bound(self, z, eps=1e-3):
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, z_hat):
        half_width = self._levels // 2
        return (z_hat * half_width) + half_width

    def _scale_and_shift_inverse(self, z_hat):
        half_width = self._levels // 2
        return (z_hat - half_width) / half_width

    def codes_to_indices(self, z_hat):
        z_hat = self._scale_and_shift(z_hat)
        return (z_hat * self._basis).sum(dim=-1).to(torch.int64)

    def indices_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes)
        return self.project_out(codes)

    def forward(self, z):
        z = self.project_in(z)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        out = self.project_out(codes)
        return out, indices

    @torch.no_grad()
    def utilization(self, indices):
        counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
        p = counts / counts.sum()
        used = (counts > 0).float().mean()
        entropy = -(p[p > 0] * p[p > 0].log()).sum()
        return {
            "used_frac": used.item(),
            "perplexity": entropy.exp().item(),
            "max_perplexity": float(self.codebook_size),
        }