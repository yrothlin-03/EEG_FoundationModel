import os
import time

import torch
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .downstream_loss import DownstreamLoss
from .metrics import Metrics


class Trainer:
    def __init__(self, model, loaders, config):
        self.model = model
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders.get("test")
        self.config = config

        t = config["training"]
        get = lambda k, d: t[k] if t.get(k) is not None else d

        self.device = next(model.parameters()).device
        self.num_classes = model.probe.head.out_features

        self.out_dir = get("out_dir", "outputs/downstream")
        self.num_epochs = get("num_epochs", 50)
        self.scale_factor = get("scale_factor", 1.0)
        self.grad_clip = get("grad_clip", 1.0)
        self.use_amp = get("use_amp", False)
        self.lr = get("lr", 4e-4)
        self.max_lr = get("max_lr", self.lr)
        self.weight_decay = get("weight_decay", 0.01)
        self.betas = tuple(get("betas", [0.9, 0.999]))
        self.optimizer_name = get("optimizer", "adamw")
        self.scheduler_name = get("scheduler", "none")
        self.log_step = t.get("log_step")
        self.max_train_step = t.get("max_step_train")
        self.max_val_step = t.get("max_step_val")
        self.max_test_step = t.get("max_step_test")

        self.pct_start = get("onecycle_pct_start", 0.3)
        self.div_factor = get("onecycle_div_factor", 25.0)
        self.final_div_factor = get("onecycle_final_div_factor", 1e4)

        self.monitor = get("monitoring_metric", "BACC")
        self.greater_is_better = self.monitor != "loss"
        self.patience = get("early_stopping_patience", 0)
        self.restore_best = get("restore_best_checkpoint", True)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.step_scheduler_per_batch = self.scheduler_name in ("onecycle", "cosine", "step")
        self.amp_device = self.device.type
        self.amp_dtype = torch.bfloat16

        self.epoch = 0
        self.best_metric = -float("inf") if self.greater_is_better else float("inf")
        self.best_state = None
        self.epochs_no_improve = 0
        self.history = {"train": [], "val": []}
        self._print_data_info()

    def _print_data_info(self):
        x, y = next(iter(self.train_loader))
        print(f"train data: {x.shape}")
        mean, std = x.mean().item(), x.std().item()
        x_max, x_min = x.max().item(), x.min().item()
        print(f"train data stats: mean={mean:.4f} | std={std:.4f} | max={x_max:.4f} | min={x_min:.4f}")
        print(f"train labels stats: min={y.min().item()} | max={y.max().item()} | unique={torch.unique(y)}")


    def _build_criterion(self):
        t = self.config["training"]
        return DownstreamLoss(
            loss=t.get("loss", "label_smoothing"),
            label_smoothing=t.get("label_smoothing", 0.0),
            class_weights=t.get("class_weights"),
        ).to(self.device)

    def _build_optimizer(self):
        decay, no_decay = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if getattr(p, "_no_weight_decay", False) or p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if self.optimizer_name == "adam":
            return torch.optim.Adam(groups, lr=self.lr, betas=self.betas, eps=1e-8)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(groups, lr=self.lr, momentum=0.9)
        return torch.optim.AdamW(groups, lr=self.lr, betas=self.betas, eps=1e-8)

    def _build_scheduler(self):
        steps_per_epoch = len(self.train_loader)
        if self.max_train_step is not None:
            steps_per_epoch = min(steps_per_epoch, self.max_train_step)
        total_steps = self.num_epochs * steps_per_epoch

        name = self.scheduler_name
        if name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, total_steps // 3), gamma=0.1)
        if name == "plateau":
            mode = "max" if self.greater_is_better else "min"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=mode, patience=3)
        return None

    def _autocast(self):
        return torch.autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp)

    def _prepare_batch(self, batch):
        x, y = batch[0], batch[1]
        x = x.to(self.device, non_blocking=True).float() * self.scale_factor
        y = y.to(self.device, non_blocking=True).long()
        return x, y

    def _is_better(self, metric):
        if self.greater_is_better:
            return metric > self.best_metric
        return metric < self.best_metric

    def _save_checkpoint(self, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        ckpt = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(ckpt, os.path.join(self.out_dir, f"downstream_{tag}.pt"))

    def _plot_curves(self):
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
        n_steps = len(self.train_loader)
        if self.max_train_step is not None:
            n_steps = min(n_steps, self.max_train_step)

        metrics = Metrics(self.num_classes)
        for step, batch in enumerate(self.train_loader):
            if self.max_train_step is not None and step >= self.max_train_step:
                break
            x, y = self._prepare_batch(batch)

            with self._autocast():
                logits = self.model(x)
                loss, logs = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), self.grad_clip)
            self.optimizer.step()
            if self.step_scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            metrics.update(logits, y, logs["loss"])

            if self.log_step and (step + 1) % self.log_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  epoch {self.epoch} | step {step + 1}/{n_steps} "
                      f"| lr: {lr:.2e} | loss: {metrics.running_loss:.4f}")

        return metrics.compute()

    @torch.no_grad()
    def evaluate(self, loader, max_step=None):
        self.model.eval()
        metrics = Metrics(self.num_classes)
        for step, batch in enumerate(loader):
            if max_step is not None and step >= max_step:
                break
            x, y = self._prepare_batch(batch)
            with self._autocast():
                logits = self.model(x)
                _, logs = self.criterion(logits, y)
            metrics.update(logits, y, logs["loss"])

        return metrics.compute()

    def train(self):
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print(f"\n===== epoch {epoch + 1}/{self.num_epochs} =====")

            t0 = time.time()
            train_logs = self.train_one_epoch()
            val_logs = self.evaluate(self.val_loader, self.max_val_step)
            dt = time.time() - t0

            if self.scheduler is not None and not self.step_scheduler_per_batch:
                self.scheduler.step(val_logs[self.monitor])

            self.history["train"].append(train_logs["loss"])
            self.history["val"].append(val_logs["loss"])

            print(f"epoch {epoch} done in {dt:.1f}s | lr: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  train | {Metrics.format(train_logs)}")
            print(f"  val   | {Metrics.format(val_logs)}")

            metric = val_logs[self.monitor]
            if self._is_better(metric):
                self.best_metric = metric
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_no_improve = 0
                self._save_checkpoint("best")
                print(f"  -> new best {self.monitor} {metric:.4f}, saved downstream_best.pt")
            else:
                self.epochs_no_improve += 1

            self._plot_curves()

            if self.patience and self.epochs_no_improve >= self.patience:
                print(f"  early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        if self.restore_best and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"restored best checkpoint ({self.monitor}: {self.best_metric:.4f})")

        self._save_checkpoint("final")

        if self.test_loader is not None:
            test_logs = self.evaluate(self.test_loader, self.max_test_step)
            print(f"\ntest | {Metrics.format(test_logs)}")

        print(f"\nDone. Saved downstream_final.pt | best val {self.monitor}: {self.best_metric:.4f}")
