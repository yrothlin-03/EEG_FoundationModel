import torch.nn as nn

from .distribution import DiagonalGaussianDistribution
from .vector_quantizer import VectorQuantizerEMA, ResidualVectorQuantizer


class VQPosterior:
    loss_name = "commitment_loss"

    def __init__(self, z_quantized, commitment_loss, indices, perplexity):
        self.z_quantized = z_quantized
        self.commitment_loss = commitment_loss
        self.indices = indices
        self.perplexity = perplexity

    def sample(self):
        return self.z_quantized

    def mode(self):
        return self.z_quantized

    def kl(self, other=None, dims=None):
        return self.commitment_loss


class Posterior(nn.Module):

    def __init__(self, z_channels, mode="kl",
                 num_embeddings=512, n_quantizers=1,
                 commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.mode = mode

        if mode == "kl":
            self.proj = nn.Conv1d(z_channels, 2 * z_channels, kernel_size=1)  # doublage
        elif mode == "vq":
            if n_quantizers > 1:
                self.quantize = ResidualVectorQuantizer(
                    num_embeddings, z_channels, n_quantizers, commitment_cost, decay)
            else:
                self.quantize = VectorQuantizerEMA(
                    num_embeddings, z_channels, commitment_cost, decay)
        else:
            raise ValueError(f"mode inconnu : {mode!r}. Attendu 'kl' ou 'vq'.")

    def forward(self, h):
        if self.mode == "kl":
            return DiagonalGaussianDistribution(self.proj(h))
        z_q, commitment_loss, indices, perplexity = self.quantize(h)
        return VQPosterior(z_q, commitment_loss, indices, perplexity)
