import torch
import torch.nn as nn
import abc 
from abc import ABC, abstractmethod
from typing import Sequence


SAMPLING_RATE = 200


class Conv1dWithConstraint(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_norm: float | None = 1.0):
        super().__init__()
        self.max_norm = max_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.conv.weight.data = torch.renorm(self.conv.weight.data, p=2, dim=0, maxnorm=float(self.max_norm))
        return self.conv(x)


class DownstreamWrapper(ABC):
    backbone: nn.Module
    classifier: nn.Module
    sampling_rate: int = SAMPLING_RATE

    @abstractmethod
    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_classifier(self, n_classes: int, **kwargs) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def build_chan_conv(self, n_channels: int, model_channels: int, max_norm: float = 1.0) -> nn.Module:
        self.chan_conv = Conv1dWithConstraint(n_channels, model_channels, max_norm=max_norm)
        return self.chan_conv

    def summary(self) -> None:
        chan_conv = getattr(self, "chan_conv", None)
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_classifier = sum(p.numel() for p in self.classifier.parameters())
        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_trainable += sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f"backbone params: {n_backbone:,}")
        print(f"classifier params: {n_classifier:,}")
        if chan_conv is not None:
            n_chan_conv = sum(p.numel() for p in chan_conv.parameters())
            n_trainable += sum(p.numel() for p in chan_conv.parameters() if p.requires_grad)
            print(f"chan_conv params: {n_chan_conv:,}")
        print(f"trainable params: {n_trainable:,}")
