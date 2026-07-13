from .mamba_blocks import MambaBlock
from .resblock import ResnetBlock


def build_block(block_type, in_channels, out_channels, dropout=0.0):
    if block_type == "mamba":
        return MambaBlock(in_channels, out_channels, dropout=dropout)
    if block_type == "resnet":
        return ResnetBlock(in_channels, out_channels, dropout=dropout)
    raise ValueError(f"Unknown block_type: {block_type!r} (expected 'mamba' or 'resnet')")
