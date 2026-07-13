import torch
import torch.nn as nn

from .model import EEGVAE
from .linear_probe import LinearProbe
from .conv_probe import ConvProbe
from .eegvae_head import EEGVAEHead


def build_probe(probe, num_classes, z_channels, probe_config):
    if probe == "conv":
        return ConvProbe(num_classes=num_classes, z_channels=z_channels, **probe_config)
    if probe == "eegvae":
        return EEGVAEHead(num_classes=num_classes, z_channels=z_channels, **probe_config)
    return LinearProbe(num_classes=num_classes, z_channels=z_channels, **probe_config)


class DownstreamModel(nn.Module):
    def __init__(self, backbone_config, num_classes, freeze_backbone=True,
                 freeze_channel_adaptor=True, probe="linear", probe_config=None,
                 input_channels=None, drop_decoder=True):
        super().__init__()
        backbone_config = dict(backbone_config)
        self.pretrained_in_channels = backbone_config["in_channels"]
        if input_channels is not None and input_channels != self.pretrained_in_channels:
            backbone_config["in_channels"] = input_channels
            backbone_config["out_channels"] = (
                backbone_config.get("out_channels") or self.pretrained_in_channels
            )
        backbone = EEGVAE(**backbone_config)
        self.channel_adaptor = backbone.channel_adaptor
        self.autoencoder = backbone.autoencoder
        self.has_decoder = hasattr(self.autoencoder, "decoder")
        if drop_decoder and self.has_decoder:
            del self.autoencoder.decoder
            self.has_decoder = False
        self.freeze_backbone = freeze_backbone
        self.freeze_channel_adaptor = freeze_channel_adaptor

        self.probe = build_probe(
            probe,
            num_classes=num_classes,
            z_channels=backbone_config.get("z_channels", 4),
            probe_config=probe_config or {},
        )

        self._freeze(self.channel_adaptor, freeze_channel_adaptor)
        self._freeze(self.autoencoder, freeze_backbone)

    @staticmethod
    def _freeze(module, frozen):
        if frozen:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_channel_adaptor:
            self.channel_adaptor.eval()
        if self.freeze_backbone:
            self.autoencoder.eval()
        return self

    def encode(self, x):
        if self.freeze_channel_adaptor:
            with torch.no_grad():
                x = self.channel_adaptor(x)
        else:
            x = self.channel_adaptor(x)

        if self.freeze_backbone and self.freeze_channel_adaptor:
            with torch.no_grad():
                return self.autoencoder.encode(x).mode()
        return self.autoencoder.encode(x).mode()

    def load_weights(self, checkpoint_path, map_location="cpu", strict=True):
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        adaptor_sd = {k[len("channel_adaptor."):]: v for k, v in state_dict.items()
                      if k.startswith("channel_adaptor.")}
        autoenc_sd = {k[len("autoencoder."):]: v for k, v in state_dict.items()
                      if k.startswith("autoencoder.")}
        if not self.has_decoder:
            autoenc_sd = {k: v for k, v in autoenc_sd.items() if not k.startswith("decoder.")}
        current = self.channel_adaptor.state_dict()
        skipped = [k for k, v in adaptor_sd.items()
                   if k not in current or current[k].shape != v.shape]
        for k in skipped:
            adaptor_sd.pop(k)
        self.channel_adaptor.load_state_dict(adaptor_sd, strict=strict and not skipped)
        self.autoencoder.load_state_dict(autoenc_sd, strict=strict)
        print(f"loaded backbone from {checkpoint_path} | "
              f"adaptor: {len(adaptor_sd)} | autoencoder: {len(autoenc_sd)} tensors")
        if skipped:
            print(f"  reinitialized adaptor input stem "
                  f"(pretrained in_channels={self.pretrained_in_channels}): {skipped}")

    def extract_features(self, x):
        return self.probe.features(self.encode(x))

    def forward(self, x):
        return self.probe(self.encode(x))

    def print_summary(self):
        def fmt(n):
            n = float(n)
            for unit in ["", "K", "M", "B"]:
                if abs(n) < 1000.0:
                    return f"{int(n)}" if unit == "" else f"{n:.2f}{unit}"
                n /= 1000.0
            return f"{n:.2f}T"

        modules = [
            ("channel_adaptor", self.channel_adaptor),
            ("autoencoder", self.autoencoder),
            ("probe", self.probe),
        ]

        def numel(p):
            # LazyModule params are uninitialized until the first forward
            return 0 if isinstance(p, nn.parameter.UninitializedParameter) else p.numel()

        print("DownstreamModel summary")
        print(f"{'Module':<18}{'Params':>12}{'Trainable':>12}{'Status':>10}")
        print("-" * 52)
        total, total_train = 0, 0
        for name, module in modules:
            n = sum(numel(p) for p in module.parameters())
            n_train = sum(numel(p) for p in module.parameters() if p.requires_grad)
            status = "train" if n_train == n and n > 0 else ("frozen" if n_train == 0 else "partial")
            total += n
            total_train += n_train
            print(f"{name:<18}{fmt(n):>12}{fmt(n_train):>12}{status:>10}")
        print("-" * 52)
        print(f"{'TOTAL':<18}{fmt(total):>12}{fmt(total_train):>12}")
        pct = 100.0 * total_train / total if total else 0.0
        print(f"trainable: {total_train:,}/{total:,} ({pct:.2f}%)")
