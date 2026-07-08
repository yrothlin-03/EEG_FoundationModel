import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGVAEHead(nn.Module):
    def __init__(self, num_classes, z_channels, pooling="asp", n_layer=2,
                 hidden_dim=128, dropout=0.2, norm="layernorm",
                 temporal_kernel_size=5, pyramid_bins=4, seq_len=None):
        super().__init__()
        self.pooling = pooling.lower()
        self.hidden_dim = int(hidden_dim)
        self.pyramid_bins = int(pyramid_bins)
        self.seq_len = seq_len
        padding = int(temporal_kernel_size) // 2

        self.input_projection = nn.Conv1d(z_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.input_norm = self._conv_norm(norm, self.hidden_dim)
        self.temporal_probe = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=int(temporal_kernel_size),
                      padding=padding, bias=False),
            self._conv_norm(norm, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attn = nn.Linear(self.hidden_dim, 1) if self.pooling in {"attn", "asp"} else None
        pooled_dim = self._pooled_dim()

        body = [nn.Dropout(dropout)]
        in_dim = pooled_dim
        if n_layer > 1:
            body += [nn.Linear(pooled_dim, self.hidden_dim), self._mlp_norm(norm, self.hidden_dim),
                     nn.GELU(), nn.Dropout(dropout)]
            for _ in range(n_layer - 2):
                body += [nn.Linear(self.hidden_dim, self.hidden_dim), self._mlp_norm(norm, self.hidden_dim),
                         nn.GELU(), nn.Dropout(dropout)]
            in_dim = self.hidden_dim
        self.mlp = nn.Sequential(*body)
        self.head = nn.Linear(in_dim, num_classes)

    def _pooled_dim(self):
        h, b = self.hidden_dim, self.pyramid_bins
        if self.pooling in {"mean", "max", "attn"}:
            return h
        if self.pooling == "asp":
            return 2 * h
        if self.pooling == "mean_max":
            return 2 * h + h * b
        if self.pooling == "stats":
            return 3 * h
        if self.pooling == "pyramid":
            return 3 * h + h * b + h * 2 * b
        if self.pooling == "flatten":
            if self.seq_len is None:
                raise ValueError("flatten pooling requires config.seq_len (latent length T')")
            return h * int(self.seq_len)
        raise ValueError(f"unknown pooling: {self.pooling!r}")

    @staticmethod
    def _conv_norm(norm, channels):
        norm = norm.lower()
        if norm == "batchnorm":
            return nn.BatchNorm1d(channels)
        if norm == "layernorm":
            return nn.GroupNorm(1, channels)
        return nn.Identity()

    @staticmethod
    def _mlp_norm(norm, channels):
        norm = norm.lower()
        if norm == "batchnorm":
            return nn.BatchNorm1d(channels)
        if norm == "layernorm":
            return nn.LayerNorm(channels)
        return nn.Identity()

    def _pool(self, x):
        if self.pooling == "mean":
            return x.mean(dim=-1)
        if self.pooling == "max":
            return x.amax(dim=-1)
        if self.pooling == "flatten":
            return x.flatten(1)
        if self.pooling == "attn":
            tokens = x.transpose(1, 2)
            weights = torch.softmax(self.attn(tokens), dim=1)
            return (tokens * weights).sum(dim=1)
        if self.pooling == "asp":
            tokens = x.transpose(1, 2)
            weights = torch.softmax(self.attn(tokens), dim=1)
            mean = (tokens * weights).sum(dim=1)
            var = (weights * (tokens - mean.unsqueeze(1)).pow(2)).sum(dim=1)
            std = torch.sqrt(var.clamp_min(1e-8))
            return torch.cat([mean, std], dim=-1)

        mean = x.mean(dim=-1)
        max_value = x.amax(dim=-1)
        if self.pooling == "mean_max":
            bins = F.adaptive_avg_pool1d(x, min(self.pyramid_bins, x.shape[-1])).flatten(1)
            return torch.cat([mean, max_value, bins], dim=-1)

        std = x.std(dim=-1, unbiased=False)
        if self.pooling == "stats":
            return torch.cat([mean, max_value, std], dim=-1)
        if self.pooling == "pyramid":
            bins4 = F.adaptive_avg_pool1d(x, min(self.pyramid_bins, x.shape[-1])).flatten(1)
            bins8 = F.adaptive_max_pool1d(x, min(2 * self.pyramid_bins, x.shape[-1])).flatten(1)
            return torch.cat([mean, max_value, std, bins4, bins8], dim=-1)
        raise ValueError(f"unknown pooling: {self.pooling!r}")

    def features(self, z):
        h = self.input_projection(z)
        h = self.input_norm(h)
        h = self.temporal_probe(h)
        return self._pool(h)

    def forward(self, z):
        return self.head(self.mlp(self.features(z)))
