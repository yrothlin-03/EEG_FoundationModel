import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .posterior import Posterior


class AutoEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels=None,
        z_channels=4,
        ch=64,
        ch_mult=[1, 2, 4],
        n_blocks=2,
        dropout=0.0,
        mode="kl",
        num_embeddings=512,
        n_quantizers=1,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.mode = mode

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            dropout=dropout,
            n_blocks=n_blocks,
        )

        self.posterior = Posterior(
            z_channels=z_channels,
            mode=mode,
            num_embeddings=num_embeddings,
            n_quantizers=n_quantizers,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            dropout=dropout,
            n_blocks=n_blocks,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        h = self.encoder(x)
        return self.posterior(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample_posterior=True, return_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        dec = dec[..., :x.shape[-1]]

        if return_posterior:
            return dec, posterior
        return dec

    def reconstruct(self, x):
        return self.forward(x, sample_posterior=False, return_posterior=False)

    def get_last_layer(self):
        return self.decoder.conv_out.weight
