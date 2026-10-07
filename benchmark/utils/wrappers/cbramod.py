from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base import DownstreamWrapper
from .attn_head import AttnHead
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "CBraMod"


def _import_cbramod():
    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    from models.cbramod import CBraMod

    return CBraMod


class CBraModClassifier(nn.Module):
    def __init__(self, n_channels: int, n_patches: int, n_classes: int, d_model: int = 200, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(n_channels * n_patches * d_model, 4 * d_model),
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


class CBraModProbeHead(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_patches: int,
        n_classes: int,
        d_model: int = 200,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ):
        super().__init__()
        self.rearrange = Rearrange("b c s d -> b (c s) d")
        self.dropout = nn.Dropout(dropout)
        self.probe = LinearWithConstraint(d_model, hidden_dim, max_norm=probe_max_norm)
        self.classifier = LinearWithConstraint(
            n_channels * n_patches * hidden_dim, n_classes, max_norm=classifier_max_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearrange(x)
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        return self.classifier(x)


class CBraModWrapper(DownstreamWrapper):
    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1):
        self.ckpt_path = ckpt_path
        self.dropout = dropout

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if time_length % self.sampling_rate != 0:
            raise ValueError(f"time_length must be a multiple of {self.sampling_rate}")
        self.n_channels = n_channels
        self.n_patches = time_length // self.sampling_rate

        CBraMod = _import_cbramod()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=self.n_patches,
            n_layer=12, nhead=8,
        )
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        self.backbone.proj_out = nn.Identity()
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        if head == "all_patch_reps":
            self.classifier = CBraModClassifier(self.n_channels, self.n_patches, n_classes, dropout=kwargs["dropout"])
        elif head == "probe":
            self.classifier = CBraModProbeHead(self.n_channels, self.n_patches, n_classes, **kwargs)
        elif head == "official":
            self.classifier = CBraModClassifier(self.n_channels, self.n_patches, n_classes, dropout=kwargs["dropout"])
        elif head == "conv":
            self.classifier = ConvHead(n_classes, d_model=200, dropout=kwargs["dropout"])
        elif head == "attn":
            self.classifier = AttnHead(
                n_classes,
                d_model=200,
                n_heads=attn_heads,
                n_queries=attn_queries,
                dropout=kwargs["dropout"],
            )
        else:
            raise ValueError(f"Unknown head {head!r}")
        return self.classifier

    def load_checkpoint(self, ckpt_path: str) -> dict:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        n_total = len(self.backbone.state_dict())
        n_loaded = n_total - len(missing)
        print(f"[CBraMod] loaded {n_loaded}/{n_total} keys from {ckpt_path}")
        if missing:
            print(f"[CBraMod] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[CBraMod] unexpected keys ({len(unexpected)}): {unexpected}")
        return state_dict

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.sampling_rate != 0:
            raise ValueError(f"time_length must be a multiple of {self.sampling_rate}")
        return x.view(b, c, t // self.sampling_rate, self.sampling_rate)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 22, 2000).to(device)

    wrapper = CBraModWrapper()
    wrapper.build_backbone(n_channels=22, time_length=2000)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)

    x = wrapper.preprocess(x)
    out = wrapper.forward(x)
    print(out.shape)
    feats = wrapper.forward_features(x)
    print(feats.shape)
