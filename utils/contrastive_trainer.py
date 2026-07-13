import os
import time

import torch
from torch.nn.utils import clip_grad_norm_

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constrative_loss import InfoNCELoss, SupConLoss


class Trainer:
    def __init__(self, model, loaders, config, clip_embedder=None):
        self.model = model
        self.clip_embedder = clip_embedder
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders.get("test")
        self.config = config

        t = config["training"]
        get = lambda k, d: t[k] if t.get(k) is not None else d

        self.device = next(model.parameters()).device
        self.loss_name = get("loss", "infonce")
        if self.loss_name == "infonce" and clip_embedder is None:
            raise ValueError("loss 'infonce' requires a clip_embedder")

        self.out_dir = get("out_dir", "outputs/contrastive")
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

        self.monitor = "loss" if self.loss_name == "supcon" else get("monitoring_metric", "top1")
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
        batch = next(iter(self.train_loader))
        shapes = [tuple(b.shape) if torch.is_tensor(b) else type(b).__name__ for b in batch]
        print(f"loss: {self.loss_name} | train batch: {shapes} | device: {self.device}")

    def _build_criterion(self):
        losses = self.config.get("losses", {})
        if self.loss_name == "infonce":
            return InfoNCELoss(**losses.get("infonce", {})).to(self.device)
        if self.loss_name == "supcon":
            return SupConLoss(**losses.get("supcon", {})).to(self.device)
        raise ValueError(f"unknown loss: {self.loss_name}")

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

    @torch.no_grad()
    def _image_features(self, img):
        if img.dim() == 4:
            return self.clip_embedder.encode(img)
        return img.float()

    def _compute(self, batch):
        x = batch[0].to(self.device, non_blocking=True).float() * self.scale_factor
        features = self.model(x)
        if self.loss_name == "infonce":
            e_img = self._image_features(batch[1].to(self.device, non_blocking=True))
            loss = self.criterion(features, e_img)
            logs = {"loss": loss.detach(), "top1": self._retrieval_top1(features, e_img)}
        else:
            labels = batch[1].to(self.device, non_blocking=True).long()
            loss = self.criterion(features, labels)
            logs = {"loss": loss.detach()}
        return loss, logs

    @staticmethod
    @torch.no_grad()
    def _retrieval_top1(eeg, image):
        logits = eeg @ image.t()
        preds = logits.argmax(dim=1)
        targets = torch.arange(eeg.size(0), device=eeg.device)
        return (preds == targets).float().mean()

    def _is_better(self, metric):
        return metric > self.best_metric if self.greater_is_better else metric < self.best_metric

    def _save_checkpoint(self, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        ckpt = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }
        torch.save(ckpt, os.path.join(self.out_dir, f"contrastive_{tag}.pt"))

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

    @staticmethod
    def _fmt(d):
        return " | ".join(f"{k}: {float(v):.4f}" for k, v in d.items())

    def train_one_epoch(self):
        self.model.train()
        n_steps = len(self.train_loader)
        if self.max_train_step is not None:
            n_steps = min(n_steps, self.max_train_step)

        running, count = {}, 0
        for step, batch in enumerate(self.train_loader):
            if self.max_train_step is not None and step >= self.max_train_step:
                break

            with self._autocast():
                loss, logs = self._compute(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), self.grad_clip)
            self.optimizer.step()
            if self.step_scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1

            if self.log_step and (step + 1) % self.log_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                avg = {k: v / count for k, v in running.items()}
                print(f"  epoch {self.epoch} | step {step + 1}/{n_steps} | lr: {lr:.2e} | {self._fmt(avg)}")

        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, loader, max_step=None):
        self.model.eval()
        running, count = {}, 0
        for step, batch in enumerate(loader):
            if max_step is not None and step >= max_step:
                break
            with self._autocast():
                _, logs = self._compute(batch)
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1
        return {k: v / max(1, count) for k, v in running.items()}

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
            print(f"  train | {self._fmt(train_logs)}")
            print(f"  val   | {self._fmt(val_logs)}")

            metric = val_logs.get(self.monitor, val_logs["loss"])
            if self._is_better(metric):
                self.best_metric = metric
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_no_improve = 0
                self._save_checkpoint("best")
                print(f"  -> new best {self.monitor} {metric:.4f}, saved contrastive_best.pt")
            else:
                self.epochs_no_improve += 1

            self._plot_curves()

            if self.patience and self.epochs_no_improve >= self.patience:
                print(f"  early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        if self.restore_best and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"restored best checkpoint ({self.monitor}: {self.best_metric:.4f})")

        self._save_checkpoint("final")

        if self.test_loader is not None:
            test_logs = self.evaluate(self.test_loader, self.max_test_step)
            print(f"\ntest | {self._fmt(test_logs)}")

        print(f"\nDone. Saved contrastive_final.pt | best val {self.monitor}: {self.best_metric:.4f}")
