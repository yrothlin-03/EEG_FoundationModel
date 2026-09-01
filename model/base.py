import abc
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F

from .projector import Projector

class LeJEPAModel(ABC, nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector()

    def load_checkpoint(self, checkpoint_path):
        pass

    @abstractmethod
    def _encode_features(self, x):
        pass

    @abstractmethod
    def get_latent_dim(self, x):
        with torch.no_grad():
            z = self._encode_features(x)
        if z.ndim == 2:
            B,D = z.shape
            return D
        elif z.ndim == 3:
            B,C,D = z.shape
            return C*D
        elif z.ndim == 4:
            B,C,L,D = z.shape
            return C*L*D
        else:
            raise ValueError(f"Unexpected feature shape: {z.shape}")


    def encode_view(self, x):
        pass

    def forward(self, globals: Sequence[torch.Tensor], locals: Sequence[torch.Tensor]):
        pass

