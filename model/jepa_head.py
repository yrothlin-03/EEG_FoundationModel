import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import EncoderBlock


class JEPAPredictor(nn.Module):
    def __init__(self, n_channels, dim, pred_dim=None, depth=2, n_heads=8):
        super().__init__()
        pred_dim = pred_dim or dim // 2
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.proj_in = nn.Linear(dim, pred_dim, bias=False)
        self.blocks = nn.ModuleList([
            EncoderBlock(n_channels, pred_dim, n_heads,
                         spatial="conv", drop_path=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, dim, bias=False)

    def forward(self, z_ctx, mask):
        h = self.proj_in(z_ctx)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        for blk in self.blocks:
            h = blk(h)
        return self.proj_out(self.norm(h))


def jepa_loss(pred, target, mask, beta=1.0):
    m = mask.unsqueeze(-1)
    if m.sum() == 0:
        return pred.sum() * 0.0
    t = F.layer_norm(target, (target.shape[-1],))
    p = F.layer_norm(pred, (pred.shape[-1],))
    l = F.smooth_l1_loss(p, t, beta=beta, reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1) / pred.shape[-1]