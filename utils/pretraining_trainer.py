import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itertools import islice

from .pretraining_loss import PretrainingLoss, generator_loss, discriminator_loss


TARGET_CHS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
              "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ"]


class Trainer:
    def __init__(self, model, discriminator, loaders, pretraining_config):
        self.model = model
        self.disc = discriminator
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.config = pretraining_config

        t = pretraining_config["training"]
        get = lambda k, d: t[k] if t.get(k) is not None else d

        self.device = next(model.parameters()).device

        self.out_dir = get("out_dir", "outputs/pretraining")
        self.num_epochs = get("num_epochs", 20)
        self.warmup_epochs = get("warmup_epochs", 0)
        self.adv_start_epoch = t.get("adv_start_epoch")
        self.grad_clip = get("grad_clip", 1.0)
        self.use_amp = get("use_amp", False)
        self.lr = get("lr", 3e-4)
        self.disc_lr = get("disc_lr", 1e-4)
        self.weight_decay = get("weight_decay", 0.05)
        self.disc_weight_decay = get("disc_weight_decay", 0.0)
        self.disc_update_freq = get("discriminator_update_freq", 1)
        self.gen_update_freq = get("generator_update_freq", 1)
        self.max_train_step = t.get("max_train_step")
        self.max_val_step = t.get("max_val_step")
        self.plot_recon_every = get("plot_recon_every", 0)
        self.log_step = t.get("log_step")

        self.rec_weight = get("recon_loss_weight", 1.0)
        self.kl_weight = get("kl_loss_weight", 1e-6)
        self.adv_weight = get("adv_loss_weight", 0.01)
        self.stft_weight = get("stft_loss_weight", 1.0)
        self.cwt_weight = get("cwt_loss_weight", 0.0)
        self.stft_sizes = get("sizes", [32, 128, 384])

        self.criterion = self._build_criterion()
        self.opt_g = self._build_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
        self.opt_d = torch.optim.Adam(
            self.disc.parameters(), lr=self.disc_lr,
            betas=(0.5, 0.9), weight_decay=self.disc_weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.amp_device = self.device.type
        self.amp_dtype = torch.bfloat16

        self.epoch = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": []}

        batch = next(iter(self.train_loader))
        print(f"train batch shape: {[b.shape for b in batch]}")
        self._plot_figure(batch)

    def _plot_figure(self, batch):
        x_in = batch[0][0].cpu().numpy()
        x_tgt = batch[1][0].cpu().numpy()

        C = x_in.shape[0]
        names = TARGET_CHS[:C] if C <= len(TARGET_CHS) else [f"ch{i}" for i in range(C)]
        n = min(3, C)

        fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i in range(n):
            axes[i].plot(x_tgt[i], color="#222222", lw=0.8, label="x_tgt")
            # axes[i].plot(x_in[i], color="#1f77b4", lw=0.8, label="x_in")
            axes[i].set_title(names[i], fontsize=10)
            axes[i].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (samples)")

        os.makedirs(self.out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "test_plot.png"), dpi=150)
        plt.close(fig)

    def _build_optimizer(self, model, lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if getattr(p, "_no_weight_decay", False) or p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)

    def _build_scheduler(self):
        total = self.num_epochs * len(self.train_loader)
        warmup = self.warmup_epochs * len(self.train_loader)

        def fn(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.opt_g, fn)

    def _build_criterion(self):
        return PretrainingLoss(
            rec_weight=self.rec_weight,
            kl_weight=self.kl_weight,
            stft_weight=self.stft_weight,
            cwt_weight=self.cwt_weight,
            stft_sizes=self.stft_sizes,
        )

    def _use_gan(self):
        return self.adv_start_epoch is not None and self.epoch >= self.adv_start_epoch

    def _prepare_batch(self, batch):
        x_in, x_tgt = batch[0], batch[1]
        x_in = x_in.to(self.device, non_blocking=True).float()
        x_tgt = x_tgt.to(self.device, non_blocking=True).float()
        return x_in, x_tgt

    def _autocast(self):
        return torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp)

    @staticmethod
    def _fmt(d):
        return " | ".join(f"{k}: {float(v):.4f}" for k, v in d.items())

    def _save_checkpoint(self, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        ckpt = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "disc": self.disc.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(ckpt, os.path.join(self.out_dir, f"pretraining_{tag}.pt"))

    @torch.no_grad()
    def _plot_recon_figure(self, epoch):
        self.model.eval()
        k=50
        batch = next(islice(self.val_loader, k, k + 1))
        x_in, x_tgt = self._prepare_batch(batch)
        rec = self.model.reconstruct(x_in)
        tgt = x_tgt[10].cpu().numpy()
        rec = rec[10].float().cpu().numpy()

        C = tgt.shape[0]
        names = TARGET_CHS[:C] if C <= len(TARGET_CHS) else [f"ch{i}" for i in range(C)]
        n = min(3, C)

        fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i in range(n):
            axes[i].plot(tgt[i], color="#222222", lw=0.8, label="original")
            axes[i].plot(rec[i], color="#d62728", lw=0.8, label="reconstruction")
            axes[i].set_title(names[i], fontsize=10)
            axes[i].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (samples)")

        os.makedirs(self.out_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f"recon_epoch{epoch}.png"), dpi=120)
        plt.close(fig)

    def _plot_loss_curves(self):
        os.makedirs(self.out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.history["train"], label="train")
        ax.plot(self.history["val"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "loss_curves.png"), dpi=120)
        plt.close(fig)

    def train_one_epoch(self):
        self.model.train()
        use_gan = self._use_gan()
        if use_gan:
            self.disc.train()

        n_steps = len(self.train_loader)
        if self.max_train_step is not None:
            n_steps = min(n_steps, self.max_train_step)

        running = {}
        count = 0
        for step, batch in enumerate(self.train_loader):
            if self.max_train_step is not None and step >= self.max_train_step:
                break
            x_in, x_tgt = self._prepare_batch(batch)

            with self._autocast():
                pred, posterior = self.model(x_in)
                loss, logs = self.criterion(pred, x_tgt, posterior)
                if use_gan:
                    adv = generator_loss(self.disc(pred))
                    loss = loss + self.adv_weight * adv
                    logs["adv_loss"] = adv.detach()

            if step % self.gen_update_freq == 0:
                self.opt_g.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt_g.step()
                self.scheduler.step()

            if use_gan and step % self.disc_update_freq == 0:
                with self._autocast():
                    logits_real = self.disc(x_tgt)
                    logits_fake = self.disc(pred.detach())
                    d_loss = discriminator_loss(logits_real, logits_fake)
                self.opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                self.opt_d.step()
                logs["disc_loss"] = d_loss.detach()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1

            if self.log_step and (step + 1) % self.log_step == 0:
                lr = self.scheduler.get_last_lr()[0]
                avg_logs = {k: v / count for k, v in running.items()}
                print(f"  epoch {self.epoch} | step {step + 1}/{n_steps} "
                      f"| lr: {lr:.2e} | {self._fmt(avg_logs)}")

        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        running = {}
        count = 0
        for step, batch in enumerate(self.val_loader):
            if self.max_val_step is not None and step >= self.max_val_step:
                break
            x_in, x_tgt = self._prepare_batch(batch)
            with self._autocast():
                pred, posterior = self.model(x_in)
                _, logs = self.criterion(pred, x_tgt, posterior)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1
        return {k: v / max(1, count) for k, v in running.items()}

    def pretrain(self):
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\n===== epoch {epoch+1}/{self.num_epochs} "
                  f"| gan: {self._use_gan()} =====")

            t0 = time.time()
            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()
            dt = time.time() - t0

            self.history["train"].append(train_logs.get("loss", 0.0))
            self.history["val"].append(val_logs.get("loss", 0.0))

            print(f"epoch {epoch} done in {dt:.1f}s | lr: {self.scheduler.get_last_lr()[0]:.2e}")
            print(f"  train | {self._fmt(train_logs)}")
            print(f"  val   | {self._fmt(val_logs)}")

            val_loss = val_logs.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print(f"  -> new best val loss {val_loss:.4f}, saved pretraining_best.pt")

            if self.plot_recon_every and (epoch + 1) % self.plot_recon_every == 0:
                self._plot_recon_figure(epoch)
            self._plot_loss_curves()

        self._save_checkpoint("final")
        print(f"\nDone. Saved pretraining_final.pt | best val loss: {self.best_val_loss:.4f}")
