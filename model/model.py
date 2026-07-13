import torch
import torch.nn as nn

from .modules.channel_adaptor import ChannelAdaptor
from .modules.autoencoder import AutoEncoder



class EEGVAE(nn.Module):

    def __init__(self,
        in_channels,
        out_channels=None,
        channel_adaptor=True,
        adaptor_layers=2,
        use_cca=False,
        z_channels=4,
        ch=64,
        ch_mult=[1, 2, 4],
        n_blocks=2,
        block_type="mamba",
        dropout=0.0,
        mode="kl",
        num_embeddings=512,
        n_quantizers=1,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.mode = mode
        self.use_channel_adaptor = channel_adaptor
        self.use_cca = use_cca
        self.block_type = block_type

        if channel_adaptor:
            self.channel_adaptor = ChannelAdaptor(
                in_channels=in_channels,
                out_channels=ch,
                n_layers=adaptor_layers,
                use_cca=use_cca,
            )
            ae_in_channels = ch
        else:
            self.channel_adaptor = nn.Identity()
            ae_in_channels = in_channels

        self.autoencoder = AutoEncoder(
            in_channels=ae_in_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            n_blocks=n_blocks,
            block_type=block_type,
            dropout=dropout,
            mode=mode,
            num_embeddings=num_embeddings,
            n_quantizers=n_quantizers,
            commitment_cost=commitment_cost,
            decay=decay,
        )

    def encode(self, x):
        x = self.channel_adaptor(x)
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def forward(self, x, sample_posterior=True, return_posterior=True):
        x = self.channel_adaptor(x)
        return self.autoencoder(
            x,
            sample_posterior=sample_posterior,
            return_posterior=return_posterior,
        )

    def reconstruct(self, x):
        return self.forward(x, sample_posterior=False, return_posterior=False)

    def get_last_layer(self):
        return self.autoencoder.get_last_layer()

    @torch.no_grad()
    def summary(self, seq_len=512, batch_size=1, device=None):
        was_training = self.training
        self.eval()

        if device is None:
            device = next(self.parameters()).device
        x = torch.randn(batch_size, self.in_channels, seq_len, device=device)

        def n_params(module):
            return sum(p.numel() for p in module.parameters())

        def fmt(n):
            n = float(n)
            for unit in ["", "K", "M", "B"]:
                if abs(n) < 1000.0:
                    return f"{int(n)}" if unit == "" else f"{n:.2f}{unit}"
                n /= 1000.0
            return f"{n:.2f}T"

        x_ad = self.channel_adaptor(x)

        enc = self.autoencoder.encoder
        h = x_ad
        enc_rows = []
        for i in range(enc.num_stages):
            for block in enc.down[i].block:
                h = block(h)
            if i != enc.num_stages - 1:
                h = enc.down[i].downsample(h)
            enc_rows.append((f"  Enc stage {i}", enc.down[i], tuple(h.shape)))
        h = enc.mid(h)
        enc_rows.append(("  Enc mid", enc.mid, tuple(h.shape)))
        h = enc.norm(h)
        h = nn.functional.silu(h)
        h = enc.conv_out(h)
        enc_rows.append(("  Enc conv_out", enc.conv_out, tuple(h.shape)))

        posterior = self.autoencoder.posterior(h)
        z = posterior.sample()
        dec = self.autoencoder.decoder(z)

        rows = [
            ("ChannelAdaptor", self.channel_adaptor, tuple(x_ad.shape)),
            ("Encoder", self.autoencoder.encoder, tuple(h.shape)),
        ]
        rows += enc_rows
        rows += [
            ("Posterior", self.autoencoder.posterior, tuple(z.shape)),
            ("Decoder", self.autoencoder.decoder, tuple(dec.shape)),
        ]

        print(f"EEGVAE summary  (mode={self.mode}, block_type={self.block_type}, use_cca={self.use_cca})")
        print(f"{'Module':<16}{'Params':>12}   Output shape")
        print("-" * 56)
        print(f"{'input':<16}{'':>12}   {tuple(x.shape)}")
        for name, module, shape in rows:
            print(f"{name:<16}{fmt(n_params(module)):>12}   {shape}")
        print("-" * 56)
        print(f"{'TOTAL':<16}{fmt(n_params(self)):>12}")

        if self.mode == "vq":
            quant = self.autoencoder.posterior.quantize
            if hasattr(quant, "quantizers"):
                n_q = quant.n_quantizers
                base = quant.quantizers[0]
            else:
                n_q = 1
                base = quant
            num_emb, emb_dim = base.num_embeddings, base.embedding_dim
            print("-" * 56)
            print(f"{'Codebook':<16}{'':>12}   {n_q} x {num_emb} codes (dim {emb_dim})")
            print(f"{'':<16}{'':>12}   total codes: {num_emb * n_q}, "
                  f"entries: {fmt(num_emb * emb_dim * n_q)}")

        if was_training:
            self.train()

        return dec, posterior




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B,C,T = 32, 64, 6000
    x = torch.randn(B,C,T).to(device)
    test_config = dict(
        in_channels = 64,
        out_channels = 64,
        use_cca=False,
        z_channels=16,
        ch=64,
        ch_mult=[1, 2, 4],
        n_blocks=2,
        block_type="mamba",
        dropout=0.0,
        mode="vq",
        num_embeddings=512,
        n_quantizers=4,
        commitment_cost=0.25,
        decay=0.99,
    )

    model = EEGVAE(**test_config).to(device)
    model.summary(seq_len=6000, batch_size=2)