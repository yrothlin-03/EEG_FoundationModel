import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal




class Norm(nn.Module):
    def __init__(self, channels, kind: Literal['group', 'layer', 'batch'] = 'group'):
        super().__init__()
        if kind == 'group':
            self.norm = nn.GroupNorm(num_groups=math.gcd(8, channels), num_channels=channels)
        elif kind == 'layer':
            self.norm = nn.LayerNorm(normalized_shape=channels)
        elif kind == 'batch':
            self.norm = nn.BatchNorm1d(num_features=channels)
        else:
            raise ValueError(f"Unsupported normalization kind: {kind}")
        
    
    def forward(self, x):
        if isinstance(self.norm, nn.LayerNorm):
            x = x.permute(0, 2, 1)  
            x = self.norm(x)
            x = x.permute(0, 2, 1) 
        else:
            x = self.norm(x)
        return x


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=8 / 3, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult)
        hidden = (hidden + 63) // 64 * 64
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(a) * b))


class ConformerConv(nn.Module):
    def __init__(self, dim, kernel_size=15, expand=2, dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        inner = dim * expand
        self.pw1 = nn.Linear(dim, 2 * inner, bias=False)
        self.dw = nn.Conv1d(inner, inner, kernel_size,
                            padding=kernel_size // 2, groups=inner, bias=False)
        self.norm = nn.GroupNorm(1, inner)
        self.pw2 = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape
        a, b = self.pw1(x).chunk(2, dim=-1)
        h = (a * torch.sigmoid(b)).reshape(B * C, N, -1).transpose(1, 2)
        h = F.silu(self.norm(self.dw(h)))
        h = h.transpose(1, 2).reshape(B, C, N, -1)
        return self.drop(self.pw2(h))



class SpatialConv(nn.Module):
    def __init__(self, C, dim, hidden_mult=1):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mix = nn.Conv2d(C, C * hidden_mult, 1, bias=False)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(C * hidden_mult, C, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, key_padding_mask=None):
        h = self.norm(x)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask[:, :, None, None], 0.0)
        return self.out(self.act(self.mix(h)))

    


class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, factor=2):
        super().__init__()
        self.factor = factor
        self.norm = nn.RMSNorm(dim_in * factor)
        self.proj = nn.Linear(dim_in * factor, dim_out, bias=False)

    def forward(self, x):
        B, C, N, D = x.shape
        N2 = (N // self.factor) * self.factor
        x = x[:, :, :N2].reshape(B, C, N2 // self.factor, D * self.factor)
        return self.proj(self.norm(x))