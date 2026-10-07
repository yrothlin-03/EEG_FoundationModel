from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base import DownstreamWrapper
from .attn_head import AttnHead
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "reve-base"

TARGET_CHS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
]


def _load_positions(electrode_names: Sequence[str]) -> torch.Tensor:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    with torch.no_grad():
        positions = model(list(electrode_names)).float().cpu()
    return positions


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out * self.weight


class ReveClassifier(nn.Module):
    def __init__(self, n_channels: int, n_patches: int, n_classes: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c h d -> b (c h d)"),
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


class ReveProbeHead(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_patches: int,
        n_classes: int,
        d_model: int = 512,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ):
        super().__init__()
        self.rearrange = Rearrange("b c h d -> b (c h) d")
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


class ReveOfficialHead(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, d = x.shape
        x = x.reshape(b, c * h, d)
        query = self.query.expand(b, -1, -1)
        scores = torch.matmul(query, x.transpose(-1, -2)) / (d ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, x).squeeze(1)
        return self.linear(self.dropout(self.norm(context)))


class ReveWrapper(DownstreamWrapper):
    model_channels = len(TARGET_CHS)

    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1):
        self.ckpt_path = ckpt_path
        self.dropout = dropout

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if ch_names is None:
            if n_channels != self.model_channels:
                raise ValueError(
                    f"ReveWrapper requires {n_channels} channel names when using a raw montage"
                )
            ch_names = TARGET_CHS
        if len(ch_names) != n_channels:
            raise ValueError(f"Expected {n_channels} channel names, got {len(ch_names)}")
        self.n_channels = n_channels

        if self.ckpt_path is not None:
            self.backbone = self.load_checkpoint(self.ckpt_path)
        else:
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(str(REPO_ROOT), trust_remote_code=True)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)

        config = self.backbone.config
        self.patch_size = config.patch_size
        self.overlap_size = config.patch_overlap
        self.embed_dim = config.embed_dim
        self.n_patches = (time_length - self.patch_size) // (self.patch_size - self.overlap_size) + 1

        self.pos = _load_positions(ch_names).unsqueeze(0)
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        if head == "all_patch_reps":
            self.classifier = ReveClassifier(self.n_channels, self.n_patches, n_classes, self.embed_dim, dropout=kwargs["dropout"])
        elif head == "probe":
            self.classifier = ReveProbeHead(self.n_channels, self.n_patches, n_classes, self.embed_dim, **kwargs)
        elif head == "official":
            self.classifier = ReveOfficialHead(n_classes, self.embed_dim, dropout=kwargs["dropout"])
        elif head == "conv":
            self.classifier = ConvHead(n_classes, d_model=self.embed_dim, dropout=kwargs["dropout"])
        elif head == "attn":
            self.classifier = AttnHead(
                n_classes,
                d_model=self.embed_dim,
                n_heads=attn_heads,
                n_queries=attn_queries,
                dropout=kwargs["dropout"],
            )
        else:
            raise ValueError(f"Unknown head {head!r}")
        return self.classifier

    def load_checkpoint(self, ckpt_path: str):
        from transformers import AutoModel

        model_dir = str(Path(ckpt_path).parent)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        print(f"[REVE] loaded pretrained weights from {model_dir}")
        return model

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        pos = self.pos.to(x.device).expand(b, -1, -1)
        return self.backbone(x, pos, return_output=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, len(TARGET_CHS), 800).to(device)

    wrapper = ReveWrapper()
    wrapper.build_backbone(n_channels=len(TARGET_CHS), time_length=800)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)

    out = wrapper.forward(x)
    print(out.shape)
    feats = wrapper.forward_features(x)
    print(feats.shape)
