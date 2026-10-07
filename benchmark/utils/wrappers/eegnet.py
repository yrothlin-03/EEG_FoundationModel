from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from braindecode.models import EEGNet

from .base import DownstreamWrapper


class EEGNetWrapper(DownstreamWrapper):
    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.25):
        self.ckpt_path = ckpt_path
        self.dropout = dropout

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        self.n_channels = n_channels
        self.time_length = time_length
        self.backbone = nn.Identity()
        return self.backbone

    def build_classifier(self, n_classes: int, **kwargs) -> nn.Module:
        self.classifier = EEGNet(
            n_chans=self.n_channels,
            n_outputs=n_classes,
            n_times=self.time_length,
            drop_prob=self.dropout,
        )
        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        return self.classifier

    def load_checkpoint(self, ckpt_path: str) -> dict:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.classifier.load_state_dict(state_dict, strict=False)
        n_total = len(self.classifier.state_dict())
        n_loaded = n_total - len(missing)
        print(f"[EEGNet] loaded {n_loaded}/{n_total} keys from {ckpt_path}")
        if missing:
            print(f"[EEGNet] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[EEGNet] unexpected keys ({len(unexpected)}): {unexpected}")
        return state_dict

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 22, 800).to(device)

    wrapper = EEGNetWrapper()
    wrapper.build_backbone(n_channels=22, time_length=800)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)

    out = wrapper.forward(x)
    print(out.shape)
