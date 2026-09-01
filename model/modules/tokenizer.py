import torch
import torch.nn as nn


class PatchStem(nn.Module):
    def __init__(self, C, dim=128, patch=20, hidden=64, kernel=9, groups=8):
        super().__init__()
        assert kernel % 2 == 1
        assert hidden % groups == 0
        self.patch, self.hidden = patch, hidden

        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(hidden * patch)
        self.proj = nn.Linear(hidden * patch, dim, bias=False)
        self.channel_emb = nn.Embedding(C, dim)
        self.presence_emb = nn.Embedding(3, dim)
        nn.init.normal_(self.channel_emb.weight, std=0.02)
        nn.init.zeros_(self.presence_emb.weight)

    def forward(self, x, presence):
        B, C, T = x.shape
        N = T // self.patch
        x = torch.sign(x) * torch.log1p(x.abs())
        h = self.conv(x.reshape(B * C, 1, T))[..., : N * self.patch]
        h = h.reshape(B, C, self.hidden, N, self.patch)
        h = h.permute(0, 1, 3, 2, 4).reshape(B, C, N, self.hidden * self.patch)
        h = self.proj(self.norm(h))
        h = h + self.channel_emb.weight[None, :, None, :]
        return h + self.presence_emb(presence)[:, :, None, :]
