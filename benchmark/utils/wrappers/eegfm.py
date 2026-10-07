from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import yaml
from einops.layers.torch import Rearrange

from .base import DownstreamWrapper
from .attn_head import AttnHead
from .conv_head import ConvHead

REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "EEG_FoundationModel"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"

TARGET_CHS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
]

DOWNSTREAM_OVERRIDES = dict(
    use_jepa=False,
    use_spectral=False,
    use_physio=False,
    use_ema_target=False,
)


def _load_model_config() -> dict:
    if not MODEL_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"EEGFM model config not found at {MODEL_CONFIG_PATH}")
    with open(MODEL_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return dict(config["model"])


def _load_positions(electrode_names: Sequence[str], montage_name: str = "standard_1020") -> torch.Tensor:
    import mne
    import numpy as np

    ch_pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    missing = [c for c in electrode_names if c not in ch_pos and c.capitalize() not in ch_pos]
    if missing:
        raise KeyError(f"channels missing from montage {montage_name}: {missing}")
    positions = [ch_pos.get(c, ch_pos.get(c.capitalize())) for c in electrode_names]
    return torch.from_numpy(np.stack(positions)).float()


def _import_eegfm():
    repo_root = str(REPO_ROOT)
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    from model import EEGFM

    return EEGFM


class EEGFMClassifier(nn.Module):
    def __init__(self, n_channels: int, n_tokens: int, n_classes: int, d_model: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c n d -> b (c n d)"),
            nn.Linear(n_channels * n_tokens * d_model, 4 * d_model),
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


class EEGFMProbeHead(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_tokens: int,
        n_classes: int,
        d_model: int = 384,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ):
        super().__init__()
        self.rearrange = Rearrange("b c n d -> b (c n) d")
        self.dropout = nn.Dropout(dropout)
        self.probe = LinearWithConstraint(d_model, hidden_dim, max_norm=probe_max_norm)
        self.classifier = LinearWithConstraint(
            n_channels * n_tokens * hidden_dim, n_classes, max_norm=classifier_max_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearrange(x)
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        return self.classifier(x)


class EEGFMOfficialHead(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGFMWrapper(DownstreamWrapper):
    model_channels = len(TARGET_CHS)

    def __init__(self, ckpt_path: str | None = None, dropout: float = 0.1):
        self.ckpt_path = ckpt_path
        self.dropout = dropout
        self.feature_level = "tokens"

    def build_backbone(self, n_channels: int, time_length: int, ch_names: Sequence[str] | None = None, **kwargs) -> nn.Module:
        if n_channels != self.model_channels:
            raise ValueError(
                f"EEGFMWrapper requires channel_mode='mapped' ({self.model_channels} channels: {TARGET_CHS}), "
                f"got n_channels={n_channels}"
            )

        config = _load_model_config()
        config.update(DOWNSTREAM_OVERRIDES)
        config.update(kwargs)
        config["fs"] = self.sampling_rate
        config["window"] = time_length

        EEGFM = _import_eegfm()
        pos = _load_positions(TARGET_CHS)
        eegfm = EEGFM(pos=pos, **config)

        self.n_channels = n_channels
        self.backbone = eegfm.encoder
        self.n_tokens = eegfm.n_tokens
        self.d_model = eegfm.z_dim

        token_stride = config["pool"] * 2 ** (len(config["dims"]) - 1)
        if self.n_tokens < 1:
            raise ValueError(f"time_length={time_length} is too short, minimum is {token_stride}")
        if time_length % token_stride != 0:
            print(
                f"[EEGFM] time_length={time_length} is not a multiple of {token_stride}, "
                f"the tail of each window is truncated to {self.n_tokens * token_stride} samples"
            )

        if self.ckpt_path is not None:
            self.load_checkpoint(self.ckpt_path)
        return self.backbone

    def build_classifier(self, n_classes: int, head: str = "all_patch_reps", **kwargs) -> nn.Module:
        kwargs.setdefault("dropout", self.dropout)
        attn_heads = kwargs.pop("attn_heads", 8)
        attn_queries = kwargs.pop("attn_queries", 1)
        self.feature_level = "global" if head == "official" else "tokens"
        if head == "all_patch_reps":
            self.classifier = EEGFMClassifier(
                self.n_channels, self.n_tokens, n_classes, self.d_model, dropout=kwargs["dropout"]
            )
        elif head == "probe":
            self.classifier = EEGFMProbeHead(
                self.n_channels, self.n_tokens, n_classes, self.d_model, **kwargs
            )
        elif head == "official":
            self.classifier = EEGFMOfficialHead(n_classes, self.d_model, dropout=kwargs["dropout"])
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
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        for prefix in ("encoder.", "target_encoder."):
            encoder_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if encoder_sd:
                break
        else:
            encoder_sd = state_dict

        model_sd = self.backbone.state_dict()
        encoder_sd = {
            k: v for k, v in encoder_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        missing, unexpected = self.backbone.load_state_dict(encoder_sd, strict=False)
        n_total = len(self.backbone.state_dict())
        n_loaded = n_total - len(missing)
        print(f"[EEGFM] loaded {n_loaded}/{n_total} encoder keys from {ckpt_path}")
        if missing:
            print(f"[EEGFM] missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"[EEGFM] unexpected keys ({len(unexpected)}): {unexpected}")
        return state_dict

    def freeze_backbone(self) -> None:
        super().freeze_backbone()
        self.backbone.eval()
        original_train = self.backbone.train

        def _stay_in_eval(mode: bool = True):
            return original_train(False)

        self.backbone.train = _stay_in_eval

    def preprocess(self, x: torch.Tensor, ch_names: Sequence[Sequence[str]] | None = None) -> torch.Tensor:
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out["z_glob"] if self.feature_level == "global" else out["z"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, len(TARGET_CHS), 800).to(device)

    wrapper = EEGFMWrapper()
    wrapper.build_backbone(n_channels=len(TARGET_CHS), time_length=800)
    wrapper.build_classifier(n_classes=4)
    wrapper.backbone.to(device)
    wrapper.classifier.to(device)

    feats = wrapper.forward_features(x)
    print(feats.shape)
    out = wrapper.forward(x)
    print(out.shape)
