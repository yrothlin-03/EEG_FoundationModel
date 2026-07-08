import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class ActNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1))
        self.eps = eps
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2], keepdim=True)
            std = x.std(dim=[0, 2], keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1.0 / (std + self.eps))

    def forward(self, x):
        if self.training and self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)
        return self.scale * (x + self.loc)


class NLayerDiscriminator1D(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        use_actnorm=False,
    ):
        super().__init__()

        norm_layer = ActNorm1d if use_actnorm else nn.BatchNorm1d
        use_bias = norm_layer != nn.BatchNorm1d

        kw = 4
        padw = 1

        sequence = [
            nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [
                nn.Conv1d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        sequence += [
            nn.Conv1d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [
            nn.Conv1d(
                ndf * nf_mult,
                1,
                kernel_size=kw,
                stride=1,
                padding=padw,
            )
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        ndf=64,
        n_layers=3,
        use_actnorm=False,
    ):
        super().__init__()

        self.model = NLayerDiscriminator1D(
            input_nc=in_channels,
            ndf=ndf,
            n_layers=n_layers,
            use_actnorm=use_actnorm,
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)