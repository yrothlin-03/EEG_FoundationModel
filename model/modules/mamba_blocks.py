import torch
import torch.nn as nn
from .utils import Norm


try:
    from mamba_ssm import Mamba, Mamba2
except ImportError:
    raise(ImportError("Mamba is not installed. Please install mamba-ssm to use MambaBlock."))


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, headdim: int, dropout: float = 0.0, expand: int = 2, bidirectional: bool = False):
        super().__init__()

        self.bidirectional = bidirectional
        common = dict(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand)

        self.mamba_forward = Mamba2(**common)
        self.mamba_backward = Mamba2(**common) if bidirectional else None
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        out = self.mamba_forward(x)
        if self.bidirectional:
            x_b = torch.flip(x, dims=[1])
            x_b = self.mamba_backward(x_b).flip(dims=[1])
            out = (out + x_b) / 2
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        return out


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = Norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, residual=True, kernel_size=15):
        super().__init__()
        self.residual = residual
        step = (out_channels + in_channels) // 2

        self.conv_block = nn.Sequential(
            SeparableConvBlock(in_channels, step, kernel_size=kernel_size, dropout=dropout),
            SeparableConvBlock(step, out_channels, kernel_size=kernel_size, dropout=dropout),
        )
        self.mamba_layer = MambaLayer(d_model=out_channels, d_state=out_channels // 2, headdim=out_channels // 4, dropout=dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)
        out = self.mamba_layer(out)
        if self.residual:
            x = self.shortcut(x)
            return x + out
        return out