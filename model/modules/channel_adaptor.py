import torch
import torch.nn as nn

from .utils import Norm
from .criss_cross_attention import CCA_Module


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2

        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = Norm(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ChannelAdaptor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_layers: int, kernel_size: int = 15, use_cca: bool = False):
        super().__init__()

        layers = []
        current = in_channels
        step = out_channels // n_layers

        for i in range(n_layers):
            next_ch = step * (i + 1) if i < n_layers - 1 else out_channels
            dilation = 2 ** i
            layers.append(SeparableBlock(current, next_ch, kernel_size=kernel_size, dilation=dilation))
            current = next_ch

        
        self.model = nn.Sequential(*layers)
        self.cca = CCA_Module(embed_dim=out_channels) if use_cca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.cca(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, T = 16, 64, 1000
    x = torch.randn(B, C, T).to(device)
    in_channels = x.shape[1]
    out_channels = 128
    n_layers = 4
    use_cca = False
    adaptor = ChannelAdaptor(in_channels, out_channels, n_layers, use_cca=use_cca).to(device)
    y = adaptor(x)
    print(y.shape)
    assert y.shape == (B, out_channels, T)
    n_params = sum(p.numel() for p in adaptor.parameters())
    print(f"Params: {n_params}")
    print("OK")