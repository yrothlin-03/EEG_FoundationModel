import torch.nn as nn
import torch.nn.functional as F
from .mamba_blocks import MambaBlock
from .utils import Norm, DownSample




class Encoder(nn.Module):
    def __init__(self,
        in_channels,
        z_channels,
        ch=64,
        ch_mult=[1, 2, 4],
        dropout=0.0,
        n_blocks=2,
    ):
        super().__init__()

        self.num_stages = len(ch_mult)
        self.n_blocks = n_blocks

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i in range(self.num_stages):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i]
            block_out = ch * in_ch_mult[i + 1]
            for _ in range(self.n_blocks):
                block.append(MambaBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
            stage = nn.Module()
            stage.block = block
            if i != self.num_stages - 1:
                stage.downsample = DownSample(block_in)
            self.down.append(stage)

        self.mid = MambaBlock(block_in, block_in, dropout=dropout)
        self.norm = Norm(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, kernel_size=3, padding=1)


    def forward(self, x):
        h = x

        for i in range(self.num_stages):
            for block in self.down[i].block:
                h = block(h)
            if i != self.num_stages - 1:
                h = self.down[i].downsample(h)

        h = self.mid(h)
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
