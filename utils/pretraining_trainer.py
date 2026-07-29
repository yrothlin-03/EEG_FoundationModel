import os
import math
import time
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pretraining_loss import PretrainingLoss


class MaskGenerator:
    def __init__(self, n_tokens, probs=(0.40, 0.25, 0.20, 0.15),
                 ratio_start=0.40, ratio_end=0.75, ramp_steps=20_000,
                 max_channel_frac=0.5, min_visible_tokens=4):
        self.n_tokens = n_tokens
        self.probs = torch.tensor(probs, dtype=torch.float32)
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.ramp_steps = max(1, ramp_steps)
        self.max_channel_frac = max_channel_frac
        self.min_visible_tokens = min_visible_tokens
        self.step = 0

    def current_ratio(self):
        p = min(1.0, self.step / self.ramp_steps)
        return self.ratio_start + p * (self.ratio_end - self.ratio_start)

    def _random_patches(self, B, C, N, r, device):
        n_mask = int(round(N * r))
        n_mask = min(N - self.min_visible_tokens, max(1, n_mask))
        scores = torch.rand(B, C, N, device=device)
        idx = scores.argsort(dim=-1)[..., :n_mask]
        m = torch.zeros(B, C, N, dtype=torch.bool, device=device)
        return m.scatter_(-1, idx, True)

    def _temporal_block(self, B, C, N, r, device):
        L = int(round(N * r))
        L = min(N - self.min_visible_tokens, max(1, L))
        start = torch.randint(0, N - L + 1, (B, 1), device=device)
        ar = torch.arange(N, device=device)[None, :]
        m = (ar >= start) & (ar < start + L)
        return m.unsqueeze(1).expand(B, C, N).clone()

    def _full_channel(self, B, C, N, r, device):
        k = int(round(C * min(r, self.max_channel_frac)))
        k = min(C - 1, max(1, k))
        idx = torch.rand(B, C, device=device).argsort(dim=-1)[:, :k]
        ch = torch.zeros(B, C, dtype=torch.bool, device=device)
        ch.scatter_(-1, idx, True)
        return ch.unsqueeze(-1).expand(B, C, N).clone()

    def _full_timestep(self, B, C, N, r, device):
        n_mask = int(round(N * r))
        n_mask = min(N - self.min_visible_tokens, max(1, n_mask))
        idx = torch.rand(B, N, device=device).argsort(dim=-1)[:, :n_mask]
        t = torch.zeros(B, N, dtype=torch.bool, device=device)
        t.scatter_(-1, idx, True)
        return t.unsqueeze(1).expand(B, C, N).clone()

    def __call__(self, B, C, device, valid=None, advance=True):
        N, r = self.n_tokens, self.current_ratio()
        kind = int(torch.multinomial(self.probs, 1).item())
        fn = (self._random_patches, self._temporal_block,
              self._full_channel, self._full_timestep)[kind]
        mask = fn(B, C, N, r, device)

        if valid is not None:
            mask = mask & valid.unsqueeze(-1)
            all_masked = mask.all(dim=(1, 2))
            if all_masked.any():
                mask[all_masked, :, : self.min_visible_tokens] = False

        if advance:
            self.step += 1
        return mask, kind, r


@torch.no_grad()
def rankme(z, max_rows=8192, eps=1e-7):
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] > max_rows:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_rows]]
    if z.shape[0] < 2:
        return 0.0
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    p = s / (s.sum() + eps)
    return float(torch.exp(-(p * (p + eps).log()).sum()).item())


@torch.no_grad()
def cosine_stats(z, max_rows=2048):
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] > max_rows:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_rows]]
    zc = F.normalize(z - z.mean(0, keepdim=True), dim=-1)
    g = zc @ zc.T
    off = ~torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
    return float(g[off].mean().item()), float(g[off].std().item())


class Trainer:
    HEADS = ("encoder", "jepa_predictor", "spectral_head", "physio_head")

    def __init__(self, model, loaders, pretraining_config):
        self.model = model
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.config = pretraining_config

        t = pretraining_config["training"]
        get = lambda k, d: t[k] if t.get(k) is not None else d

        self.device = next(model.parameters()).device
        self.out_dir = get("out_dir", "outputs/pretraining")
        self.num_epochs = get("num_epochs", 20)
        self.warmup_frac = get("warmup_frac", 0.05)
        self.grad_clip = get("grad_clip", 1.0)
        self.use_amp = get("use_amp", True)
        self.lr = get("lr", 1.5e-4)
        self.weight_decay = get("weight_decay", 0.05)
        self.betas = tuple(get("betas", (0.9, 0.95)))
        self.max_train_step = t.get("max_train_step")
        self.max_val_step = t.get("max_val_step")
        self.log_step = t.get("log_step")
        self.diag_every = get("diag_every", 1)
        self.plot_every = get("plot_every", 1)
        self.channel_dropout = get("channel_dropout", 0.2)
        self.max_channel_drop = get("max_channel_drop", 3)
        self.detach_target_until = get("detach_target_until", None)
        self.ema_momentum_start = get("ema_momentum_start", 0.996)
        self.ema_momentum_end = get("ema_momentum_end", 1.0)

        self.steps_per_epoch = (
            min(len(self.train_loader), self.max_train_step)
            if self.max_train_step else len(self.train_loader)
        )
        self.total_steps = self.num_epochs * self.steps_per_epoch

        mask_cfg = t.get("masking") or {}
        self.masker = MaskGenerator(
            n_tokens=self.model.n_tokens,
            probs=tuple(mask_cfg.get("probs", (0.40, 0.25, 0.20, 0.15))),
            ratio_start=mask_cfg.get("ratio_start", 0.40),
            ratio_end=mask_cfg.get("ratio_end", 0.75),
            ramp_steps=int(mask_cfg.get("ramp_frac", 0.20) * self.total_steps),
        )

        self.criterion = self._build_criterion(t).to(self.device)
        self.opt = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp_dtype = torch.bfloat16

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": [], "rankme": [], "rankme_token": []}

        batch = next(iter(self.train_loader))
        x, presence = self._prepare_batch(batch)
        print(f"x: {tuple(x.shape)} | presence: {tuple(presence.shape)} "
              f"| n_tokens: {self.model.n_tokens} "
              f"| stride: {self.model.token_stride()} "
              f"| params: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
        self._sanity_check(x, presence)

    def _build_criterion(self, t):
        head_kw = {}
        sh = self.model.spectral_head
        if sh is not None:
            head_kw["spectral"] = dict(n_fft_list=sh.n_fft_list, hop_div=sh.hop_div,
                                       fs=sh.fs, f_max=sh.f_max)
        ph = self.model.physio_head
        if ph is not None:
            head_kw["physio"] = dict(fs=self.model.fs, n_feats=ph.net[-1].out_features)

        return PretrainingLoss(
            weights=t.get("loss_weights"),
            adaptive=t.get("adaptive_weights", True),
            jepa_warmup_frac=t.get("jepa_warmup_frac", 0.05),
            total_steps=self.total_steps,
            sigreg_token_weight=t.get("sigreg_token_weight", 0.1),
            max_weight_ratio=t.get("max_weight_ratio", 4.0),
            **head_kw,
        )

    def _build_optimizer(self):
        groups = self.model.param_groups(weight_decay=self.weight_decay)
        return torch.optim.AdamW(groups, lr=self.lr, betas=self.betas, eps=1e-8)

    def _build_scheduler(self):
        warmup = int(self.warmup_frac * self.total_steps)

        def fn(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            p = (step - warmup) / max(1, self.total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

        return torch.optim.lr_scheduler.LambdaLR(self.opt, fn)

    def _prepare_batch(self, batch):
        if isinstance(batch, dict):
            x, presence = batch["x"], batch.get("presence")
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
            presence = next(
                (e for e in batch[1:]
                 if torch.is_tensor(e) and e.shape == x.shape[:2]),
                None,
            )
        else:
            x, presence = batch, None
        x = x.to(self.device, non_blocking=True).float()
        if presence is None:
            presence = torch.ones(x.shape[0], x.shape[1],
                                  dtype=torch.long, device=self.device)
        else:
            presence = presence.to(self.device, non_blocking=True).long()
        return x, presence

    def _apply_channel_dropout(self, presence):
        if not self.model.training or self.channel_dropout <= 0:
            return presence
        B, C = presence.shape
        active = torch.rand(B, device=presence.device) < self.channel_dropout
        if not active.any():
            return presence
        k = int(torch.randint(1, self.max_channel_drop + 1, (1,)).item())
        valid = presence != 0
        scores = torch.rand(B, C, device=presence.device).masked_fill(~valid, -1.0)
        idx = scores.argsort(dim=-1, descending=True)[:, :k]
        drop = torch.zeros(B, C, dtype=torch.bool, device=presence.device)
        drop.scatter_(-1, idx, True)
        drop = drop & active.unsqueeze(-1) & valid
        return presence.masked_fill(drop, 0)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type,
                              dtype=self.amp_dtype, enabled=self.use_amp)

    @staticmethod
    def _fmt(d):
        return " | ".join(f"{k}: {float(v):.4f}" for k, v in d.items())

    def _step(self, x, presence, advance_mask=True):
        presence = self._apply_channel_dropout(presence)
        valid = presence != 0
        mask, kind, ratio = self.masker(
            x.shape[0], x.shape[1], x.device, valid, advance=advance_mask
        )
        with self._autocast():
            out = self.model(x, presence, mask)
            loss, logs = self.criterion(
                out, {"x_clean": x, "mask": out["mask"], "valid": out["valid"]}
            )
        logs["mask/kind"] = float(kind)
        logs["mask/ratio"] = float(ratio)
        return loss, logs, out

    @torch.no_grad()
    def _sanity_check(self, x, presence):
        self.model.eval()
        self.criterion.eval()
        loss, logs, out = self._step(x[:2], presence[:2], advance_mask=False)
        print("sanity | " + self._fmt({k: v for k, v in logs.items()
                                       if k.startswith("loss/")}))
        print(f"  z: {tuple(out['z'].shape)} | z_glob: {tuple(out['z_glob'].shape)}")
        assert torch.isfinite(loss), "loss non finie au premier pas"

    @torch.no_grad()
    def _grad_norms(self):
        out = {}
        for name in self.HEADS:
            mod = getattr(self.model, name, None)
            if mod is None:
                continue
            gs = [p.grad.detach().float().norm() ** 2
                  for p in mod.parameters() if p.grad is not None]
            out[f"gradnorm/{name}"] = (
                float(torch.stack(gs).sum().sqrt().item()) if gs else 0.0
            )
        total = math.sqrt(sum(v ** 2 for v in out.values())) or 1.0
        for k in list(out):
            out[k.replace("gradnorm/", "gradfrac/")] = out[k] / total
        return out

    def _target_momentum(self):
        p = min(1.0, self.global_step / max(1, self.total_steps))
        return self.ema_momentum_end - (
            (self.ema_momentum_end - self.ema_momentum_start)
            * (math.cos(math.pi * p) + 1) / 2
        )

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()
        if self.detach_target_until is not None and self.model.target_encoder is None:
            self.model.detach_target = self.epoch < self.detach_target_until

        running, count = {}, 0
        window, wcount = {}, 0
        last_grads = {}
        for step, batch in enumerate(self.train_loader):
            if self.max_train_step is not None and step >= self.max_train_step:
                break
            x, presence = self._prepare_batch(batch)
            loss, logs, _ = self._step(x, presence)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if step % max(1, self.steps_per_epoch // 4) == 0:
                last_grads = self._grad_norms()
            gnorm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.scheduler.step()
            self.global_step += 1
            m = self._target_momentum()
            self.model.update_target(m)

            logs["grad/total"] = float(gnorm)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + float(v)
                window[k] = window.get(k, 0.0) + float(v)
            count += 1
            wcount += 1

            if self.log_step and (step + 1) % self.log_step == 0:
                avg = {k: v / wcount for k, v in window.items()
                       if k.startswith(("loss/", "grad/"))}
                print(f"  epoch {self.epoch} | step {step+1}/{self.steps_per_epoch} "
                      f"| lr {self.scheduler.get_last_lr()[0]:.2e} "
                      f"| m {m:.4f} | {self._fmt(avg)}")
                window, wcount = {}, 0

        out = {k: v / max(1, count) for k, v in running.items()}
        out.update(last_grads)
        return out

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        self.criterion.eval()
        running, count = {}, 0
        z_pool, zg_pool = [], []

        for step, batch in enumerate(self.val_loader):
            if self.max_val_step is not None and step >= self.max_val_step:
                break
            x, presence = self._prepare_batch(batch)
            _, logs, out = self._step(x, presence, advance_mask=False)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + float(v)
            count += 1
            if len(zg_pool) < 16:
                zg_pool.append(out["z_glob_target"].float().cpu())
                z_pool.append(out["z_target"][:, :, ::4].float().cpu())

        stats = {k: v / max(1, count) for k, v in running.items()}
        if zg_pool:
            zg = torch.cat(zg_pool)
            z = torch.cat(z_pool)
            stats["diag/rankme_glob"] = rankme(zg)
            stats["diag/rankme_token"] = rankme(z)
            mu, sd = cosine_stats(zg)
            stats["diag/cos_mean"] = mu
            stats["diag/cos_std"] = sd
            stats["diag/z_norm"] = float(zg.norm(dim=-1).mean().item())
        return stats

    def _save_checkpoint(self, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "criterion": self.criterion.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "masker_step": self.masker.step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, os.path.join(self.out_dir, f"pretraining_{tag}.pt"))

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.criterion.load_state_dict(ck["criterion"])
        self.opt.load_state_dict(ck["opt"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.masker.step = ck.get("masker_step", 0)
        self.epoch = ck["epoch"] + 1
        self.global_step = ck.get("global_step", 0)
        self.best_val_loss = ck.get("best_val_loss", float("inf"))

    @torch.no_grad()
    def _plot_gabor(self):
        fb = self.model.encoder.stem.fb
        hz = fb.frequencies().cpu().numpy()
        real, _ = fb.kernels()
        H = torch.fft.rfft(real.float().cpu(), n=512).abs().numpy()
        freqs = np.fft.rfftfreq(512, d=1.0 / fb.fs)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].stem(np.arange(len(hz)), hz)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("filtre")
        ax[0].set_ylabel("fc (Hz)")
        ax[0].axhline(50.0, color="r", ls="--", lw=0.8)
        for i in range(H.shape[0]):
            ax[1].plot(freqs, H[i] / (H[i].max() + 1e-8), lw=0.7)
        ax[1].set_xlim(0, 60)
        ax[1].set_xlabel("Hz")
        ax[1].set_ylabel("|H(f)| normalisé")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"gabor_epoch{self.epoch}.png"), dpi=120)
        plt.close(fig)

    @torch.no_grad()
    def _plot_dist_bias(self):
        tables = {k: v for k, v in self.model.diagnostics().items()
                  if k.startswith("dist_bias")}
        if not tables:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, tab in tables.items():
            for h in range(tab.shape[0]):
                ax.plot(tab[h].numpy(), lw=0.8, alpha=0.6)
        ax.set_xlabel("bin de distance")
        ax.set_ylabel("biais")
        ax.axhline(0.0, color="k", lw=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"distbias_epoch{self.epoch}.png"), dpi=120)
        plt.close(fig)

    @torch.no_grad()
    def _plot_spectral(self):
        if self.model.spectral_head is None:
            return
        self.model.eval()
        k = min(10, max(0, len(self.val_loader) - 1))
        batch = next(islice(self.val_loader, k, k + 1))
        x, presence = self._prepare_batch(batch)
        _, _, out = self._step(x, presence, advance_mask=False)
        r = min(1, len(self.criterion.spectral.n_fft_list) - 1)
        tgt = self.criterion.spectral.targets(x)[r][0, 0].float().cpu().numpy()
        pred = out["spectral_pred"][r][0, 0].float().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        vmin, vmax = tgt.min(), tgt.max()
        for a, (img, title) in zip(ax, [(tgt, "cible"), (pred, "prédiction")]):
            a.imshow(img.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            a.set_title(title)
            a.set_xlabel("frame")
        ax[0].set_ylabel("bin fréquentiel")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"spectral_epoch{self.epoch}.png"), dpi=120)
        plt.close(fig)

    def _plot_curves(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.history["train"], label="train")
        ax[0].plot(self.history["val"], label="val")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend()
        ax[1].plot(self.history["rankme"], label="z_glob")
        ax[1].plot(self.history["rankme_token"], label="z tokens")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("RankMe")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "curves.png"), dpi=120)
        plt.close(fig)

    def pretrain(self, start_epoch=0):
        os.makedirs(self.out_dir, exist_ok=True)
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            target_mode = ("ema" if self.model.target_encoder is not None
                           else f"detach={self.model.detach_target}")
            print(f"\n===== epoch {epoch+1}/{self.num_epochs} "
                  f"| mask ratio {self.masker.current_ratio():.2f} "
                  f"| target {target_mode} =====")

            t0 = time.time()
            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()
            dt = time.time() - t0

            self.history["train"].append(train_logs.get("loss/total", 0.0))
            self.history["val"].append(val_logs.get("loss/total", 0.0))
            self.history["rankme"].append(val_logs.get("diag/rankme_glob", 0.0))
            self.history["rankme_token"].append(val_logs.get("diag/rankme_token", 0.0))

            print(f"epoch {epoch} done in {dt:.1f}s "
                  f"| lr {self.scheduler.get_last_lr()[0]:.2e}")
            print("  train | " + self._fmt({k: v for k, v in train_logs.items()
                                            if k.startswith("loss/")}))
            print("  val   | " + self._fmt({k: v for k, v in val_logs.items()
                                            if k.startswith("loss/")}))
            print("  diag  | " + self._fmt({k: v for k, v in val_logs.items()
                                            if k.startswith("diag/")}))
            print("  grads | " + self._fmt({k: v for k, v in train_logs.items()
                                            if k.startswith("gradfrac/")}))
            print("  w     | " + self._fmt({k: v for k, v in train_logs.items()
                                            if k.startswith("weight/")}))

            for k, v in train_logs.items():
                if k.startswith("gradfrac/") and k != "gradfrac/encoder" and v < 0.05:
                    print(f"  [warn] {k} = {v:.3f} < 0.05 : tête quasi inerte")
            rk = val_logs.get("diag/rankme_glob", 0.0)
            if rk and rk < 0.1 * self.model.z_dim:
                print(f"  [warn] RankMe {rk:.1f} << z_dim {self.model.z_dim} "
                      f": collapse dimensionnel probable")

            val_loss = val_logs.get("loss/total", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print(f"  -> best val {val_loss:.4f}, saved pretraining_best.pt")
            self._save_checkpoint("last")

            if self.plot_every and (epoch + 1) % self.plot_every == 0:
                self._plot_gabor()
                self._plot_dist_bias()
                self._plot_spectral()
            self._plot_curves()

        self._save_checkpoint("final")
        print(f"\nDone. best val loss: {self.best_val_loss:.4f}")