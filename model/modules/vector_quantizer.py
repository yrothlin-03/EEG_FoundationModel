import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", embedding.clone())

    def freeze_codebook(self):
        self._codebook_frozen = True

    def forward(self, z):
        B, D, T = z.shape

        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)

        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.T
        )

        indices = dist.argmin(dim=1)
        z_q_flat = self.embedding[indices]

        if self.training and not getattr(self, "_codebook_frozen", False):
            z_flat_fp32 = z_flat.detach().float()

            indices_onehot = F.one_hot(indices, self.num_embeddings).float()
            cluster_size_new = indices_onehot.sum(dim=0)
            embedding_sum_new = indices_onehot.T @ z_flat_fp32

            if torch.isfinite(embedding_sum_new).all():
                self.cluster_size.mul_(self.decay).add_(cluster_size_new, alpha=1.0 - self.decay)
                self.embedding_avg.mul_(self.decay).add_(embedding_sum_new, alpha=1.0 - self.decay)

                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )
                self.embedding.copy_(self.embedding_avg / smoothed.unsqueeze(1))

        commitment_loss = self.commitment_cost * F.mse_loss(
            z_flat.float(), z_q_flat.detach().float()
        )

        z_q_flat = z_flat + (z_q_flat.to(z_flat.dtype) - z_flat).detach()

        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1).contiguous()
        indices = indices.view(B, T)

        avg_probs = F.one_hot(indices, self.num_embeddings).float().view(-1, self.num_embeddings).mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, commitment_loss, indices, perplexity


class ResidualVectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        n_quantizers: int = 8,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay, eps)
            for _ in range(n_quantizers)
        ])

    def freeze_codebook(self):
        for q in self.quantizers:
            q.freeze_codebook()

    def forward(self, z):
        residual = z
        z_q_total = torch.zeros_like(z)
        total_commitment = torch.tensor(0.0, device=z.device, dtype=torch.float32)
        all_indices = []
        all_perplexities = []

        for quantizer in self.quantizers:
            z_q_i, commitment_i, indices_i, perplexity_i = quantizer(residual)
            residual = residual - z_q_i.detach()
            z_q_total = z_q_total + z_q_i
            total_commitment = total_commitment + commitment_i
            all_indices.append(indices_i)
            all_perplexities.append(perplexity_i)

        indices = torch.stack(all_indices, dim=1)
        perplexity = torch.stack(all_perplexities).mean()
        commitment_loss = total_commitment / self.n_quantizers

        return z_q_total, commitment_loss, indices, perplexity
