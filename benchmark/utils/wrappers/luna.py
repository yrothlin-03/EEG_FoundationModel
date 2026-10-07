from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from .base import DownstreamWrapper
from .attn_head import AttnHead
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "BioFoundation"

TARGET_CHS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
]


def _import_luna():
    import timm

    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "modules" or name.startswith("modules."):
            del sys.modules[name]

    saved_path = list(sys.path)
    sys.path[:] = [
        p for p in sys.path
        if p == repo_root or not (Path(p, "models").is_dir() or Path(p, "modules").is_dir())
    ]
    try:
        from models.LUNA import LUNA
    finally:
        sys.path[:] = saved_path

    return LUNA


def _load_positions(electrode_names: Sequence[str]) -> torch.Tensor:
    import mne
    import numpy as np

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = {k.upper(): v for k, v in montage.get_positions()["ch_pos"].items()}
    coords = np.array([ch_pos[name.upper()] for name in electrode_names])
    return torch.from_numpy(coords).float()


class LUNAClassifier(nn.Module):
    def __init__(self, n_patches: int, n_classes: int, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b s d -> b (s d)"),
            nn.Linear(n_patches * d_model, 4 * d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearWithConstraint(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_norm: float | None = None):
        super().__init__()
        self.max_norm = max_norm
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.linear.weight.data = torch.renorm(
                self.linear.weight.data, p=2, dim=0, maxnorm=float(self.max_norm)
            )
        return self.linear(x)


class LUNAProbeHead(nn.Module):
    def __init__(
        self,
        n_patches: int,
        n_classes: int,
        d_model: int = 256,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.probe = LinearWithConstraint(d_model, hidden_dim, max_norm=probe_max_norm)
        self.classifier = LinearWithConstraint(n_patches * hidden_dim, n_classes, max_norm=classifier_max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        return self.classifier(x)


class LUNAOfficialHead(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 256, num_heads: int = 2, dropout: float = 0.15):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, n_classes),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        query = self.query.expand(b, -1, -1)
        out, _ = self.attn(query=query, key=x, value=x)
        out = out[:, 0, :]
        return self.mlp(out)


class LUNAWrapper(DownstreamWrapper):
    patch_size = 40
    num_queries = 4
    embed_dim = 64
    num_heads = 2
    depth = 8
    model_channels = len(TARGET_CHS)

    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1):
        self.ckpt_path = ckpt_path
        self.dropout = dropout

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if n_channels != self.model_channels:
            raise ValueError(
                f"LUNAWrapper requires channel_mode='mapped' ({self.model_channels} channels: {TARGET_CHS}), "
                f"got n_channels={n_channels}"
            )
        if time_length % self.patch_size != 0:
            raise ValueError(f"time_length must be a multiple of {self.patch_size}")
        self.n_channels = n_channels
        self.n_patches = time_length // self.patch_size

        LUNA = _import_luna()
        self.backbone = LUNA(
            patch_size=self.patch_size,
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            drop_path=0.0,
            num_classes=0,
        )
        self.d_model = self.embed_dim * self.num_queries

        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)

        del self.backbone.decoder_head
        del self.backbone.channel_emb

        self.pos = _load_positions(TARGET_CHS).unsqueeze(0)
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        if head == "all_patch_reps":
            self.classifier = LUNAClassifier(self.n_patches, n_classes, self.d_model, dropout=kwargs["dropout"])
        elif head == "probe":
            self.classifier = LUNAProbeHead(self.n_patches, n_classes, self.d_model, **kwargs)
        elif head == "official":
            self.classifier = LUNAOfficialHead(n_classes, self.d_model, self.num_heads)
        elif head == "conv":
            self.classifier = ConvHead(n_classes, d_model=self.d_model, dropout=kwargs["dropout"])
        elif head == "attn":
            self.classifier = AttnHead(
                n_classes,
                d_model=self.d_model,
                n_heads=attn_heads,
                n_queries=attn_queries,
                dropout=kwargs["dropout"],
            )
        else:
            raise ValueError(f"Unknown head {head!r}")
        return self.classifier

    def load_checkpoint(self, ckpt_path: str) -> dict:
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        n_total = len(self.backbone.state_dict())
        n_loaded = n_total - len(missing)
        print(f"[LUNA] loaded {n_loaded}/{n_total} keys from {ckpt_path}")
        if missing:
            print(f"[LUNA] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[LUNA] unexpected keys ({len(unexpected)}): {unexpected}")
        return state_dict

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        pos = self.pos.to(x.device).expand(b, -1, -1)
        x, _ = self.backbone.prepare_tokens(x, pos, mask=None)
        x, _ = self.backbone.cross_attn(x)
        x = rearrange(x, "(B t) Q D -> B t (Q D)", B=b)
        for blk in self.backbone.blocks:
            x = blk(x)
        return self.backbone.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, len(TARGET_CHS), 800).to(device)

    wrapper = LUNAWrapper()
    wrapper.build_backbone(n_channels=len(TARGET_CHS), time_length=800)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)

    out = wrapper.forward(x)
    print(out.shape)
    feats = wrapper.forward_features(x)
    print(feats.shape)
