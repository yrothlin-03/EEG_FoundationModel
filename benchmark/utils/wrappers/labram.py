from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base import DownstreamWrapper
from .attn_head import AttnHead
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "LaBraM"


def _import_labram():
    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    for name in list(sys.modules):
        if name == "modeling_finetune" or name.startswith("modeling_finetune."):
            del sys.modules[name]
    from modeling_finetune import NeuralTransformer

    return NeuralTransformer


class LaBraMClassifier(nn.Module):
    def __init__(self, n_tokens: int, n_classes: int, d_model: int = 200, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b n d -> b (n d)"),
            nn.Linear(n_tokens * d_model, 4 * d_model),
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


class LaBraMProbeHead(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        n_classes: int,
        d_model: int = 200,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.probe = LinearWithConstraint(d_model, hidden_dim, max_norm=probe_max_norm)
        self.classifier = LinearWithConstraint(n_tokens * hidden_dim, n_classes, max_norm=classifier_max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        return self.classifier(x)


class LaBraMOfficialHead(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 200):
        super().__init__()
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.mean(1))


class LaBraMWrapper(DownstreamWrapper):
    patch_size = 200
    embed_dim = 200
    depth = 12
    num_heads = 10
    out_chans = 8
    model_channels = 128

    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1, chan_conv_max_norm: float = 1.0):
        self.ckpt_path = ckpt_path
        self.dropout = dropout
        self.chan_conv_max_norm = chan_conv_max_norm

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if time_length % self.sampling_rate != 0:
            raise ValueError(f"time_length must be a multiple of {self.sampling_rate}")
        self.n_channels = n_channels
        self.n_patches = time_length // self.patch_size

        NeuralTransformer = _import_labram()
        self.backbone = NeuralTransformer(
            EEG_size=self.patch_size,
            patch_size=self.patch_size,
            out_chans=self.out_chans,
            num_classes=0,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_mean_pooling=True,
        )
        self.build_chan_conv(n_channels, self.model_channels, max_norm=self.chan_conv_max_norm)
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        n_tokens = self.model_channels * self.n_patches
        if head == "all_patch_reps":
            self.classifier = LaBraMClassifier(n_tokens, n_classes, self.embed_dim, dropout=kwargs["dropout"])
        elif head == "probe":
            self.classifier = LaBraMProbeHead(n_tokens, n_classes, self.embed_dim, **kwargs)
        elif head == "official":
            self.classifier = LaBraMOfficialHead(n_classes, self.embed_dim)
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

    def load_checkpoint(self, ckpt_path: str) -> dict:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt.get("module", ckpt))
        if any(k.startswith("student.") for k in state_dict):
            state_dict = {k[len("student."):]: v for k, v in state_dict.items() if k.startswith("student.")}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        n_total = len(self.backbone.state_dict())
        n_loaded = n_total - len(missing)
        print(f"[LaBraM] loaded {n_loaded}/{n_total} keys from {ckpt_path}")
        if missing:
            print(f"[LaBraM] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[LaBraM] unexpected keys ({len(unexpected)}): {unexpected}")
        return state_dict

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.sampling_rate != 0:
            raise ValueError(f"time_length must be a multiple of {self.sampling_rate}")
        return x.view(b, c, t // self.sampling_rate, self.sampling_rate)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a, t = x.shape
        x = self.chan_conv(x.reshape(b, c, a * t)).reshape(b, self.model_channels, a, t)
        return self.backbone.forward_features(x, return_patch_tokens=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 22, 800).to(device)

    wrapper = LaBraMWrapper()
    wrapper.build_backbone(n_channels=22, time_length=800)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)
    wrapper.chan_conv.to(device)

    x = wrapper.preprocess(x)
    out = wrapper.forward(x)
    print(out.shape)
    feats = wrapper.forward_features(x)
    print(feats.shape)
