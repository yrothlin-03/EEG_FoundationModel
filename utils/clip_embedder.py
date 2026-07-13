import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPEmbedder(nn.Module):
    def __init__(self, model_name="ViT-L-14", pretrained="laion2b_s32b_b82k",
                 normalize=True, device=None):
        super().__init__()
        self.normalize = normalize
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.embed_dim = self.model.visual.output_dim
        for p in self.model.parameters():
            p.requires_grad = False
        if device is not None:
            self.model.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, images):
        if not torch.is_tensor(images):
            if isinstance(images, (list, tuple)):
                images = torch.stack([self.preprocess(img) for img in images])
            else:
                images = self.preprocess(images).unsqueeze(0)
        images = images.to(self.device)
        features = self.model.encode_image(images)
        if self.normalize:
            features = F.normalize(features, dim=-1)
        return features

    def forward(self, images):
        return self.encode(images)


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    embedder = CLIPEmbedder()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    features = embedder(img)
    print(features.shape)