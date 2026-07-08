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








class UpSampleBis(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                stride=2,
            ),
            Norm(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        B,C,T = x.shape
        if T % 2 != 0:
            raise ValueError(f"Input tensor length {T} is not divisible by 2 for UpSample.")
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = Norm(in_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            Norm(in_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        B,C,T = x.shape
        if T%2 != 0:
            x = F.pad(x, (0, 1), mode="replicate")
        return self.block(x)