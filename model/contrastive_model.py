import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EEGVAE
from .contrastive_head import ProjectionHead


class ContrastiveModel(nn.Module):
    def __init__(self, backbone_config, clip_dim, freeze_backbone=True,
                 freeze_channel_adaptor=True, head_config=None,
                 input_channels=None, normalize=True):
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
        if hasattr(self.autoencoder, "decoder"):
            del self.autoencoder.decoder

        self.freeze_backbone = freeze_backbone
        self.freeze_channel_adaptor = freeze_channel_adaptor
        self.normalize = normalize

        self.head = ProjectionHead(
            z_channels=backbone_config.get("z_channels", 4),
            clip_dim=clip_dim,
            **(head_config or {}),
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

    def forward(self, x):
        features = self.head(self.encode(x))
        if self.normalize:
            features = F.normalize(features, dim=-1)
        return features

    def load_weights(self, checkpoint_path, map_location="cpu", strict=False):
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        adaptor_sd = {k[len("channel_adaptor."):]: v for k, v in state_dict.items()
                      if k.startswith("channel_adaptor.")}
        autoenc_sd = {k[len("autoencoder."):]: v for k, v in state_dict.items()
                      if k.startswith("autoencoder.") and not k.startswith("autoencoder.decoder.")}
        current = self.channel_adaptor.state_dict()
        for k in [k for k, v in adaptor_sd.items()
                  if k not in current or current[k].shape != v.shape]:
            adaptor_sd.pop(k)
        self.channel_adaptor.load_state_dict(adaptor_sd, strict=strict)
        self.autoencoder.load_state_dict(autoenc_sd, strict=strict)
        print(f"loaded backbone from {checkpoint_path} | "
              f"adaptor: {len(adaptor_sd)} | autoencoder: {len(autoenc_sd)} tensors")

