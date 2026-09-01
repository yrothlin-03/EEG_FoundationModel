import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import Encoder
from .modules.stem import Stem


class EEGFM(nn.Module):
    def __init__(
        self,
        pos,
        dims=(128, 192, 256, 384),
        depths=(2, 2, 3, 3),
        spatial=("conv", "conv", "beamformer", "beamformer"),
        n_heads=8,
        n_bands=32,
        kernel_size=65,
        fs=200,
        pool=20,
        window=6000,
        z_dim=None,
        drop_path=0.15,
        dropout=0.0,
        n_dist_bins=16,
        hop_div=4,
        f_max=45.0,
        n_global_queries=4,
        grad_checkpointing=False,
        stem="gabor",
        patch_hidden=64,
        patch_kernel=9,
    ):
        super().__init__()
        self.fs = fs
        self.pool = pool
        self.window = window
        self.n_channels = pos.shape[0]

        self.encoder = Encoder(
            pos,
            dims=dims,
            depths=depths,
            spatial=spatial,
            n_heads=n_heads,
            n_bands=n_bands,
            kernel_size=kernel_size,
            fs=fs,
            pool=pool,
            z_dim=z_dim,
            drop_path=drop_path,
            dropout=dropout,
            n_dist_bins=n_dist_bins,
            n_global_queries=n_global_queries,
            grad_checkpointing=grad_checkpointing,
            stem=stem,
            patch_hidden=patch_hidden,
            patch_kernel=patch_kernel,
        )
        self.z_dim = z_dim or dims[-1]
        self.n_tokens = self._token_count(window, depths)

    def _token_count(self, T, depths):
        n = T // self.pool
        for _ in range(len(depths) - 1):
            n //= 2
        return n

    def token_stride(self):
        return self.window // self.n_tokens

    def _default_presence(self, x):
        return torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)

    def encode(self, x, presence=None, return_hidden=False):
        if presence is None:
            presence = self._default_presence(x)
        return self.encoder(x, presence, return_hidden=return_hidden)

    def forward(self, x, presence=None, return_hidden=False):
        if presence is None:
            presence = self._default_presence(x)
        out = self.encoder(x, presence, return_hidden=return_hidden)
        out["valid"] = presence != 0
        return out

    @torch.no_grad()
    def embed(self, x, presence=None, level="global", hop=None, reduce="mean_max_std"):
        B, C, T = x.shape
        win = self.window
        hop = hop or win // 2
        if presence is None:
            presence = self._default_presence(x)

        if T < win:
            x = F.pad(x, (0, win - T))
            T = win

        starts = list(range(0, max(T - win, 0) + 1, hop))
        if starts[-1] + win < T:
            starts.append(T - win)

        chunks = []
        for s in starts:
            out = self.encoder(x[..., s : s + win], presence)
            chunks.append(out["z_glob"] if level == "global" else out["z"])
        z = torch.stack(chunks, dim=1)

        if level != "global":
            return z
        if reduce == "mean":
            return z.mean(1)
        if reduce == "max":
            return z.amax(1)
        if reduce == "mean_max_std":
            return torch.cat([z.mean(1), z.amax(1), z.std(1, unbiased=False)], dim=-1)
        raise ValueError(f"reduce inconnu: {reduce}")

    def no_weight_decay(self):
        keys = (
            "dist_bias.table",
            "log_fc",
            "log_sigma",
            "_emb",
            "query",
            "norm",
            "ls_s",
            "ls_t",
            "ls_c",
            "ls_f",
        )
        return {n for n, _ in self.named_parameters() if any(k in n for k in keys)}

    def param_groups(self, weight_decay=0.05):
        skip = self.no_weight_decay()
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if n in skip or p.ndim <= 1 else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def diagnostics(self):
        d = self.encoder.diagnostics()
        d["n_tokens"] = self.n_tokens
        d["token_stride"] = self.token_stride()
        return d


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, C, fs, window = 2, 19, 200, 2000
    pos = torch.randn(C, 3)
    pos = pos / pos.norm(dim=-1, keepdim=True)

    model = EEGFM(
        pos,
        dims=(32, 64, 96, 128),
        depths=(1, 1, 1, 1),
        n_heads=4,
        n_bands=8,
        kernel_size=33,
        fs=fs,
        pool=20,
        window=window,
    ).to(device)
    model.eval()

    x = torch.randn(B, C, window, device=device)
    presence = torch.ones(B, C, dtype=torch.long, device=device)
    presence[1, -3:] = 0

    with torch.no_grad():
        out = model(x, presence)
        out_hidden = model.encode(x, presence, return_hidden=True)
        emb = model.embed(torch.randn(B, C, 3 * window, device=device), presence)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    print(f"n_tokens: {model.n_tokens}  token_stride: {model.token_stride()}")
    print(f"z: {tuple(out['z'].shape)}  z_glob: {tuple(out['z_glob'].shape)}")
    print(f"valid: {tuple(out['valid'].shape)}  sum: {out['valid'].sum().item()}")
    print(f"hidden: {[tuple(h.shape) for h in out_hidden['hidden']]}")
    print(f"embed: {tuple(emb.shape)}")

    groups = model.param_groups()
    print(f"decay: {len(groups[0]['params'])}  no_decay: {len(groups[1]['params'])}")

    assert out["z"].shape == (B, C, model.n_tokens, model.z_dim)
    assert out["z_glob"].shape == (B, model.z_dim)
    assert emb.shape == (B, 3 * model.z_dim)
    assert torch.isfinite(out["z"]).all() and torch.isfinite(out["z_glob"]).all()

    model.train()
    loss = model(x, presence)["z_glob"].square().mean()
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f"backward ok, loss={loss.item():.4f}, params avec grad: {n_grad}/{len(list(model.parameters()))}")

