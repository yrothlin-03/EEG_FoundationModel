from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


class EEGVAEProbeHead(nn.Module):

    def __init__(
        self,
        task: str = "classification",
        n_classes: int = 2,
        output_dim: int = 1,
        pooling: str = "mean_max",
        n_layer: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        norm: str = "layernorm",
        probe_max_norm: Optional[float] = None,
        classifier_max_norm: Optional[float] = None,
        temporal_kernel_size: int = 5,
        pyramid_bins: int = 4,
        **_,
    ):
        super().__init__()
        self.task = task
        self.pooling = str(pooling).lower()
        self.hidden_dim = int(hidden_dim)
        self.n_layer = int(n_layer)
        self.dropout = float(dropout)
        self.norm = str(norm).lower()
        self.probe_max_norm = probe_max_norm
        self.classifier_max_norm = classifier_max_norm
        self.pyramid_bins = int(pyramid_bins)

        if self.pooling not in {
            "mean",
            "max",
            "mean_max",
            "stats",
            "pyramid",
            "flatten",
            "attn",
            "attention",
        }:
            raise ValueError(f"Unknown EEGVAE pooling: {pooling}")

        out_dim = int(n_classes) if task == "classification" else int(output_dim)
        padding = int(temporal_kernel_size) // 2

        self.input_projection = nn.LazyConv1d(self.hidden_dim, kernel_size=1, bias=False)
        self.input_norm = self._make_conv_norm(self.hidden_dim)
        self.temporal_probe = nn.Sequential(
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=int(temporal_kernel_size),
                padding=padding,
                bias=False,
            ),
            self._make_conv_norm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        if self.pooling in {"attn", "attention"}:
            self.attn = nn.Sequential(nn.LazyLinear(1), nn.Softmax(dim=1))
        else:
            self.attn = None

        self.mlp = self._build_mlp(out_dim)

    def _make_conv_norm(self, channels: int) -> nn.Module:
        if self.norm == "batchnorm":
            return nn.BatchNorm1d(channels)
        if self.norm == "layernorm":
            return nn.GroupNorm(1, channels)
        if self.norm == "none":
            return nn.Identity()
        raise ValueError(f"Unknown norm: {self.norm}")

    def _build_mlp(self, out_dim: int) -> nn.Sequential:
        layers = [nn.Dropout(self.dropout)]

        if self.n_layer <= 1:
            layers.append(nn.LazyLinear(out_dim))
            return nn.Sequential(*layers)

        layers.append(nn.LazyLinear(self.hidden_dim))
        self._add_mlp_norm_activation_dropout(layers)

        for _ in range(self.n_layer - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self._add_mlp_norm_activation_dropout(layers)

        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _add_mlp_norm_activation_dropout(self, layers):
        if self.norm == "batchnorm":
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        elif self.norm == "layernorm":
            layers.append(nn.LayerNorm(self.hidden_dim))
        elif self.norm == "none":
            pass
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.dropout))

    def _to_latent_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(-1)

        if x.ndim == 3:
            return x.contiguous()

        if x.ndim == 4:
            b, c, h, w = x.shape
            return x.reshape(b, c, h * w).contiguous()

        raise ValueError(f"Unsupported EEGVAE feature shape: {tuple(x.shape)}")

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_latent_map(x)
        x = self.input_projection(x)
        x = self.input_norm(x)
        return self.temporal_probe(x)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return x.mean(dim=-1)

        if self.pooling == "max":
            return x.amax(dim=-1)

        if self.pooling == "flatten":
            return x.flatten(1)

        if self.pooling in {"attn", "attention"}:
            tokens = x.transpose(1, 2).contiguous()
            weights = self.attn(tokens)
            return (tokens * weights).sum(dim=1)

        mean = x.mean(dim=-1)
        max_value = x.amax(dim=-1)

        if self.pooling == "mean_max":
            bins = nn.functional.adaptive_avg_pool1d(
                x, min(self.pyramid_bins, x.shape[-1])
            ).flatten(1)
            return torch.cat([mean, max_value, bins], dim=-1)

        std = x.std(dim=-1, unbiased=False)
        if self.pooling == "stats":
            return torch.cat([mean, max_value, std], dim=-1)

        if self.pooling == "pyramid":
            bins4 = nn.functional.adaptive_avg_pool1d(
                x, min(self.pyramid_bins, x.shape[-1])
            ).flatten(1)
            bins8 = nn.functional.adaptive_max_pool1d(
                x, min(2 * self.pyramid_bins, x.shape[-1])
            ).flatten(1)
            return torch.cat([mean, max_value, std, bins4, bins8], dim=-1)

        raise ValueError(f"Unknown EEGVAE pooling: {self.pooling}")

    def _renorm_linear(self, module: nn.Linear, max_norm: Optional[float]):
        if max_norm is None or isinstance(module.weight, UninitializedParameter):
            return

        module.weight.data = torch.renorm(
            module.weight.data,
            p=2,
            dim=0,
            maxnorm=float(max_norm),
        )

    def _apply_max_norm(self):
        linear_layers = [
            module for module in self.mlp if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            return

        for module in linear_layers[:-1]:
            self._renorm_linear(module, self.probe_max_norm)
        self._renorm_linear(linear_layers[-1], self.classifier_max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode_latent(x)
        x = self._pool(x)
        self._apply_max_norm()
        x = self.mlp(x)

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
