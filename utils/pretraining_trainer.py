import os
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pretraining_loss import PretrainingLoss


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


class Trainer:

    def __init__(self, model, loaders, config):
        self.model = model
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.config = config["trainer"]
        self.loss_config = config["loss"]

        self.device = next(model.parameters()).device
        self.out_dir = self.config["out_dir"]
        self.num_epochs = self.config["num_epochs"]
        self.grad_clip = self.config["grad_clip"]
        self.use_amp = self.config["use_amp"]
        self.max_train_step = self.config.get("max_train_step")
        self.max_val_step = self.config.get("max_val_step")
        self.log_step = self.config.get("log_step")
        self.plot_every = self.config.get("plot_every", 1)
        self.rankme_rows = self.config.get("rankme_rows", 2048)
        self.rankme_pool_rows = 0
        self.amp_dtype = torch.bfloat16

        self.criterion = self._build_criterion(self.loss_config).to(self.device)
        self.optimizer = self._build_optimizer(self.config)
        self.scheduler = self._build_scheduler(self.config)

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": [], "alignment": [], "sigreg": [], "rankme": []}

    def _build_criterion(self, config):
        return PretrainingLoss(config)

    def _build_optimizer(self, config):
        return torch.optim.AdamW(
            self.model.param_groups(weight_decay=config["weight_decay"]),
            lr=config["lr"],
            betas=tuple(config["betas"]),
        )

    def _build_scheduler(self, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["T_max"],
            eta_min=config["eta_min"],
        )

    def _prepare_batch(self, batch):
        x = batch[0].to(self.device, non_blocking=True).float()
        presence = batch[1].to(self.device, non_blocking=True).long()
        to_device = lambda views: [v.to(self.device, non_blocking=True).float() for v in views]
        global_views = to_device(batch[2]) if len(batch) > 2 else [x]
        local_views = to_device(batch[3]) if len(batch) > 3 else []
        return x, presence, global_views, local_views

    def _encode_views(self, views, presence, reference=None):
        if not views:
            return reference.new_zeros((reference.shape[0], 0, reference.shape[-1]))
        z = [self.model(v, presence)["z_glob"] for v in views]
        return torch.stack(z, dim=1)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    @staticmethod
    def _fmt(d):
        return " | ".join(f"{k}: {float(v):.4f}" for k, v in d.items())

    def _step(self, batch):
        _, presence, global_views, local_views = self._prepare_batch(batch)
        with self._autocast():
            z_global = self._encode_views(global_views, presence)
            z_local = self._encode_views(local_views, presence, reference=z_global)
            out = self.criterion(z_global, z_local, global_step=self.global_step)
        loss, logs = out if isinstance(out, tuple) else (out, {"total_loss": out.detach()})
        return loss, logs, z_global

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()
        running, count = {}, 0
        window, wcount = {}, 0

        for step, batch in enumerate(self.train_loader):
            if self.max_train_step is not None and step >= self.max_train_step:
                break

            loss, logs, _ = self._step(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            logs = dict(logs)
            logs["grad_norm"] = float(gnorm)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + float(v)
                window[k] = window.get(k, 0.0) + float(v)
            count += 1
            wcount += 1

            if self.log_step and (step + 1) % self.log_step == 0:
                avg = {k: v / wcount for k, v in window.items()}
                print(f"  epoch {self.epoch} | step {step+1} "
                      f"| lr {self.scheduler.get_last_lr()[0]:.2e} | {self._fmt(avg)}")
                window, wcount = {}, 0

        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        self.criterion.eval()
        running, count = {}, 0
        z_pool, pooled_rows = [], 0

        for step, batch in enumerate(self.val_loader):
            if self.max_val_step is not None and step >= self.max_val_step:
                break

            _, logs, z_global = self._step(batch)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + float(v)
            count += 1
            if pooled_rows < self.rankme_rows:
                z_pool.append(z_global[:, 0].float().cpu())
                pooled_rows += z_global.shape[0]

        self.rankme_pool_rows = pooled_rows
        stats = {k: v / max(1, count) for k, v in running.items()}
        if z_pool:
            stats["rankme"] = rankme(torch.cat(z_pool))
        return stats

    def _save_checkpoint(self, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": {"trainer": self.config, "loss": self.loss_config},
        }, os.path.join(self.out_dir, f"pretraining_{tag}.pt"))

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.epoch = ck["epoch"] + 1
        self.global_step = ck.get("global_step", 0)
        self.best_val_loss = ck.get("best_val_loss", float("inf"))
        self.history = ck.get("history", self.history)

    @torch.no_grad()
    def _plot_diagnostics(self):
        diag = self.model.diagnostics()
        table = diag["dist_bias"].numpy()
        hz = diag["gabor_hz"].numpy() if "gabor_hz" in diag else None

        n_ax = 2 if hz is not None else 1
        fig, ax = plt.subplots(1, n_ax, figsize=(6 * n_ax, 4), squeeze=False)
        ax = ax[0]
        if hz is not None:
            ax[0].stem(np.arange(len(hz)), hz)
            ax[0].set_yscale("log")
            ax[0].set_xlabel("filtre")
            ax[0].set_ylabel("fc (Hz)")
        for h in range(table.shape[0]):
            ax[-1].plot(table[h], lw=0.8, alpha=0.7)
        ax[-1].axhline(0.0, color="k", lw=0.5)
        ax[-1].set_xlabel("bin de distance")
        ax[-1].set_ylabel("biais")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"diagnostics_epoch{self.epoch}.png"), dpi=120)
        plt.close(fig)

    @torch.no_grad()
    def _plot_batch(self, n_channels=19):
        batch = next(iter(self.train_loader))
        x, presence, global_views, local_views = self._prepare_batch(batch)

        b = 2
        valid = (presence[b] != 0).nonzero().flatten()
        pool = valid if len(valid) >= n_channels else torch.arange(x.shape[1], device=x.device)
        idx = pool[torch.randperm(len(pool), device=x.device)[:n_channels]].cpu()

        columns = [("x", x, "#2a78d6")]
        if global_views:
            columns.append(("vue globale 0", global_views[0], "#eb6834"))
        if local_views:
            columns.append(("vue locale 0", local_views[0], "#1baf7a"))

        t = np.arange(x.shape[-1]) / self.model.fs
        fig, ax = plt.subplots(
            len(idx), len(columns),
            figsize=(6 * len(columns), 2.2 * len(idx)),
            sharex=True, sharey="row", squeeze=False,
        )

        for row, c in enumerate(idx):
            for col, (name, signal, color) in enumerate(columns):
                ax[row, col].plot(t, signal[b, c].float().cpu().numpy(), lw=0.8, color=color)
                if row == 0:
                    ax[row, col].set_title(name)
            ax[row, 0].set_ylabel(f"canal {int(c)}")
        for col in range(len(columns)):
            ax[-1, col].set_xlabel("temps (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "batch.png"), dpi=120)
        plt.close(fig)

    def _plot_curves(self):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].plot(self.history["train"], label="train")
        ax[0].plot(self.history["val"], label="val")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].legend()
        ax[1].plot(self.history["alignment"], label="alignment")
        ax[1].plot(self.history["sigreg"], label="sigreg")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("composantes (val)")
        ax[1].legend()
        ax[2].plot(self.history["rankme"])
        ax[2].set_xlabel("epoch")
        ax[2].set_ylabel("RankMe")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "curves.png"), dpi=120)
        plt.close(fig)

    def pretrain(self, start_epoch=0):
        os.makedirs(self.out_dir, exist_ok=True)
        self._plot_batch()

        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            print(f"\n===== epoch {epoch+1}/{self.num_epochs} =====")

            t0 = time.time()
            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()
            dt = time.time() - t0

            self.history["train"].append(train_logs.get("total_loss", 0.0))
            self.history["val"].append(val_logs.get("total_loss", 0.0))
            self.history["alignment"].append(val_logs.get("alignment_loss", 0.0))
            self.history["sigreg"].append(val_logs.get("sigreg_loss", 0.0))
            self.history["rankme"].append(val_logs.get("rankme", 0.0))

            print(f"epoch {epoch+1} done in {dt:.1f}s | lr {self.scheduler.get_last_lr()[0]:.2e}")
            print("  train | " + self._fmt(train_logs))
            print("  val   | " + self._fmt(val_logs))

            ceiling = min(max(1, self.rankme_pool_rows - 1), self.model.z_dim)
            rk = val_logs.get("rankme", 0.0)
            if rk and rk < 0.1 * ceiling:
                print(f"  [warn] RankMe {rk:.1f} << plafond {ceiling} "
                      f"({self.rankme_pool_rows} lignes, z_dim {self.model.z_dim}) "
                      f": collapse dimensionnel probable")

            val_loss = val_logs.get("total_loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print(f"  -> best val {val_loss:.4f}, saved pretraining_best.pt")
            self._save_checkpoint("last")

            if self.plot_every and (epoch + 1) % self.plot_every == 0:
                self._plot_diagnostics()
            self._plot_curves()

        self._save_checkpoint("final")
        print(f"\nDone. best val loss: {self.best_val_loss:.4f}")
