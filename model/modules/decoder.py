import torch.nn as nn
import torch.nn.functional as F
from .blocks import build_block
from .utils import Norm, UpSample



class Decoder(nn.Module):
    def __init__(self,
        out_channels,
        z_channels,
        ch=64,
        ch_mult=[1, 2, 4],
        dropout=0.0,
        n_blocks=2,
        block_type="mamba",
    ):
        super().__init__()

        self.num_stages = len(ch_mult)
        self.n_blocks = n_blocks

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[-1]

        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        self.mid = build_block(block_type, block_in, block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_stages)):
            block = nn.ModuleList()
            block_out = ch * in_ch_mult[i]
            for _ in range(self.n_blocks):
                block.append(build_block(block_type, block_in, block_out, dropout=dropout))
                block_in = block_out

            stage = nn.Module()
            stage.block = block
            if i != 0:
                stage.upsample = UpSample(block_in)
            self.up.insert(0, stage)

        self.norm = Norm(block_in)
        self.conv_out = nn.Conv1d(block_in, out_channels, kernel_size=3, padding=1)


    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)

        for i in reversed(range(self.num_stages)):
            for block in self.up[i].block:
                h = block(h)
            if i != 0:
                h = self.up[i].upsample(h)

        h = self.norm(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
