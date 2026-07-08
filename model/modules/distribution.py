import numpy as np
import torch


class DiagonalGaussianDistribution:
    loss_name = "kl_loss"

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters

        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        self.deterministic = deterministic

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self):
        noise = torch.randn_like(self.mean)
        return self.mean + self.std * noise

    def kl(self, other=None, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.parameters.device)

        if other is None:
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=dims,
            )

        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dims,
        )

    def nll(self, sample, dims=[1, 2]):
        if self.deterministic:
            return torch.zeros(1, device=self.parameters.device)

        logtwopi = np.log(2.0 * np.pi)

        return 0.5 * torch.sum(
            logtwopi
            + self.logvar
            + (sample - self.mean).pow(2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean