import torch.nn as nn
import torch.nn.functional as F
from .mamba_blocks import MambaBlock
from .utils import Norm, UpSample




class Decoder(nn.Module):
    def __init__(self,
        out_channels,
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
        block_in = ch * ch_mult[-1]

        # entrée : latent -> canaux du goulot
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        # goulot
        self.mid = MambaBlock(block_in, block_in, dropout=dropout)

        # niveaux montants (miroir de l'encodeur)
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_stages)):
            block = nn.ModuleList()
            block_out = ch * in_ch_mult[i]
            for _ in range(self.n_blocks):
                block.append(MambaBlock(block_in, block_out, dropout=dropout))
                block_in = block_out

            stage = nn.Module()
            stage.block = block
            if i != 0:
                stage.upsample = UpSample(block_in)
            self.up.insert(0, stage)

        # sortie
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
