from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .attn_head import AttnHead
from .base import DownstreamWrapper
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "CSBrain"

TARGET_CHS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
]

BRAIN_REGIONS = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]

TOPOLOGY = {
    0: ["FP1", "F7", "F3", "FZ", "F4", "F8", "FP2"],
    1: ["P3", "PZ", "P4"],
    2: ["P7", "T7", "T8", "P8"],
    3: ["O1", "O2"],
    4: ["C3", "CZ", "C4"],
}

SORTED_INDICES = [
    index
    for region in sorted(TOPOLOGY)
    for channel in TOPOLOGY[region]
    for index, name in enumerate(TARGET_CHS)
    if name == channel
]


def _import_csbrain():
    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    from models.CSBrain import CSBrain

    return CSBrain


class CSBrainClassifier(nn.Module):
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
    def __init__(self, in_features: int, out_features: int, max_norm: float | None = None):
        super().__init__()
        self.max_norm = max_norm
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.linear.weight.data = torch.renorm(self.linear.weight.data, p=2, dim=0, maxnorm=float(self.max_norm))
        return self.linear(x)


class CSBrainProbeHead(nn.Module):
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
        self.probe = LinearWithConstraint(d_model, hidden_dim, probe_max_norm)
        self.classifier = LinearWithConstraint(n_channels * n_patches * hidden_dim, n_classes, classifier_max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.probe(self.dropout(self.rearrange(x)))
        return self.classifier(x.flatten(1))


class CSBrainWrapper(DownstreamWrapper):
    model_channels = len(TARGET_CHS)

    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1):
        self.ckpt_path = ckpt_path
        self.dropout = dropout

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if n_channels != self.model_channels:
            raise ValueError(
                f"CSBrainWrapper requires channel_mode='mapped' ({self.model_channels} channels: {TARGET_CHS}), "
                f"got n_channels={n_channels}"
            )
        if time_length % self.sampling_rate != 0:
            raise ValueError(f"time_length must be a multiple of {self.sampling_rate}")
        self.n_channels = n_channels
        self.n_patches = time_length // self.sampling_rate
        CSBrain = _import_csbrain()
        self.backbone = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=self.n_patches,
            n_layer=12,
            nhead=8,
            brain_regions=BRAIN_REGIONS,
            sorted_indices=SORTED_INDICES,
        )
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        self.backbone.proj_out = nn.Identity()
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        if head in {"all_patch_reps", "official"}:
            self.classifier = CSBrainClassifier(self.n_channels, self.n_patches, n_classes, dropout=kwargs["dropout"])
        elif head == "probe":
            self.classifier = CSBrainProbeHead(self.n_channels, self.n_patches, n_classes, **kwargs)
        elif head == "conv":
            self.classifier = ConvHead(n_classes, d_model=200, dropout=kwargs["dropout"])
        elif head == "attn":
            self.classifier = AttnHead(n_classes, d_model=200, n_heads=attn_heads, n_queries=attn_queries, dropout=kwargs["dropout"])
        else:
            raise ValueError(f"Unknown head {head!r}")
        return self.classifier

    def load_checkpoint(self, ckpt_path: str) -> dict:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model_state = self.backbone.state_dict()
        matching = {key: value for key, value in state_dict.items() if key in model_state and value.shape == model_state[key].shape}
        model_state.update(matching)
        self.backbone.load_state_dict(model_state)
        print(f"[CSBrain] loaded {len(matching)}/{len(model_state)} keys from {ckpt_path}")
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
