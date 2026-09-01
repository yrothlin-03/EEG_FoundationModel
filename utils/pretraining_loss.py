import math
from typing import Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce



class AlignmentLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        config = config or {}

        self.detach_center = config.get(
            "detach_center",
            False,
        )
        self.feature_reduction = config.get(
            "feature_reduction",
            "mean",
        )
        self.reduction = config.get(
            "reduction",
            "mean",
        )

    def forward(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
    ) -> torch.Tensor:
        if z_global.ndim != 3:
            raise ValueError(
                "z_global must have shape [B, V, D]."
            )

        if z_local.ndim != 3:
            raise ValueError(
                "z_local must have shape [B, L, D]."
            )


        center = z_global.mean(
            dim=1,
            keepdim=True,
        )

        if self.detach_center:
            center = center.detach()

        all_views = torch.cat(
            [z_global, z_local],
            dim=1,
        )

        squared_error = (
            all_views - center
        ).square()

        if self.feature_reduction == "mean":
            per_view_loss = squared_error.mean(dim=-1)
        else:
            per_view_loss = squared_error.sum(dim=-1)

        if self.reduction == "mean":
            return per_view_loss.mean()

        if self.reduction == "sum":
            return per_view_loss.sum()

        return per_view_loss



class SIGRegLoss(nn.Module):
    def __init__(
        self,
        num_slices: int = 256,
        num_integration_points: int = 17,
        integration_range: tuple[float, float] = (-5.0, 5.0),
        reduction: Literal["mean", "sum", "none"] = "mean",
        base_seed: int = 0,
        sync_ddp: bool = True,
        force_float32: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.num_slices = num_slices
        self.reduction = reduction
        self.base_seed = base_seed
        self.sync_ddp = sync_ddp
        self.force_float32 = force_float32
        self.eps = eps

        integration_points = torch.linspace(
            integration_range[0],
            integration_range[1],
            num_integration_points,
            dtype=torch.float32,
        )

        target_cf = torch.exp(
            -0.5 * integration_points.square()
        )

        self.register_buffer(
            "integration_points",
            integration_points,
            persistent=False,
        )

        self.register_buffer(
            "target_cf",
            target_cf,
            persistent=False,
        )

    @staticmethod
    def _is_distributed() -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
        )

    def _sample_directions(
        self,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        global_step: Optional[Union[int, torch.Tensor]],
    ) -> torch.Tensor:
        generator = None

        if global_step is not None:
            if isinstance(global_step, torch.Tensor):
                global_step = int(global_step.item())
            else:
                global_step = int(global_step)

            generator = torch.Generator(device=device)
            generator.manual_seed(
                self.base_seed + global_step
            )

        elif self.sync_ddp and self._is_distributed():
            raise ValueError(
                "global_step is required when sync_ddp=True."
            )

        directions = torch.randn(
            latent_dim,
            self.num_slices,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return F.normalize(
            directions,
            p=2,
            dim=0,
            eps=self.eps,
        )

    @staticmethod
    def _distributed_sum(
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        return differentiable_all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        global_step: Optional[
            Union[int, torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape [N, K], got {tuple(x.shape)}."
            )

        if x.shape[0] == 0:
            raise ValueError("The embedding batch is empty.")

        if not x.is_floating_point():
            raise TypeError(
                "x must be a floating-point tensor."
            )

        if (
            self.force_float32
            and x.dtype in {
                torch.float16,
                torch.bfloat16,
            }
        ):
            x = x.float()

        num_local_samples, latent_dim = x.shape

        directions = self._sample_directions(
            latent_dim=latent_dim,
            device=x.device,
            dtype=x.dtype,
            global_step=global_step,
        )

        projected = x @ directions

        integration_points = self.integration_points.to(
            device=x.device,
            dtype=x.dtype,
        )

        target_cf = self.target_cf.to(
            device=x.device,
            dtype=x.dtype,
        )

        phase = (
            projected.unsqueeze(-1)
            * integration_points
        )

        real_sum = torch.cos(phase).sum(dim=0)
        imaginary_sum = torch.sin(phase).sum(dim=0)

        global_count = x.new_tensor(
            float(num_local_samples)
        )

        if self.sync_ddp and self._is_distributed():
            real_sum = self._distributed_sum(real_sum)
            imaginary_sum = self._distributed_sum(
                imaginary_sum
            )

            dist.all_reduce(
                global_count,
                op=dist.ReduceOp.SUM,
            )

        empirical_cf_real = real_sum / global_count
        empirical_cf_imaginary = (
            imaginary_sum / global_count
        )

        squared_error = (
            (empirical_cf_real - target_cf).square()
            + empirical_cf_imaginary.square()
        )

        weighted_error = squared_error * target_cf

        per_slice_loss = torch.trapezoid(
            weighted_error,
            integration_points,
            dim=-1,
        )

        per_slice_loss = (
            global_count * per_slice_loss
        )

        if self.reduction == "mean":
            return per_slice_loss.mean()

        if self.reduction == "sum":
            return per_slice_loss.sum()

        return per_slice_loss




class PretrainingLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.sigreg_weight = float(config.get("sigreg_weight", 0.05))

        if not 0.0 <= self.sigreg_weight <= 1.0:
            raise ValueError(
                "sigreg_weight must be between 0 and 1."
            )

        self.return_components = bool(
            config.get("return_components", False)
        )

        sigreg_config = dict(config.get("sigreg") or {})

        if "integration_range" in sigreg_config:
            sigreg_config["integration_range"] = tuple(
                sigreg_config["integration_range"]
            )

        self.alignment_loss = AlignmentLoss(
            config.get("alignment")
        )

        self.sigreg_loss = SIGRegLoss(
            **sigreg_config
        )

    @staticmethod
    def _reduce_to_scalar(
        loss: torch.Tensor,
    ) -> torch.Tensor:
        if loss.ndim == 0:
            return loss
        return loss.mean()

    def forward(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        global_step: Optional[
            Union[int, torch.Tensor]
        ] = None,
    ):
        if z_global.ndim != 3:
            raise ValueError(
                "z_global must have shape [B, V, D]."
            )

        if z_local.ndim != 3:
            raise ValueError(
                "z_local must have shape [B, L, D]."
            )

        if z_global.shape[0] != z_local.shape[0]:
            raise ValueError(
                "z_global and z_local must have "
                "the same batch size."
            )

        if z_global.shape[-1] != z_local.shape[-1]:
            raise ValueError(
                "z_global and z_local must have "
                "the same latent dimension."
            )

        if z_global.shape[1] == 0:
            raise ValueError(
                "z_global must contain at least one view."
            )

        alignment_loss = self.alignment_loss(
            z_global,
            z_local,
        )

        alignment_loss = self._reduce_to_scalar(
            alignment_loss
        )

        all_views = torch.cat(
            [z_global, z_local],
            dim=1,
        )

        sigreg_losses = []

        for view_index in range(all_views.shape[1]):
            view_loss = self.sigreg_loss(
                all_views[:, view_index, :],
                global_step=global_step,
            )

            sigreg_losses.append(
                self._reduce_to_scalar(view_loss)
            )

        sigreg_loss = torch.stack(
            sigreg_losses
        ).mean()

        total_loss = (
            (1.0 - self.sigreg_weight)
            * alignment_loss
            + self.sigreg_weight
            * sigreg_loss
        )

        if self.return_components:
            return total_loss, {
                "total_loss": total_loss.detach(),
                "alignment_loss": alignment_loss.detach(),
                "sigreg_loss": sigreg_loss.detach(),
                "align_ratio": self._alignment_ratio(
                    alignment_loss,
                    all_views,
                    z_global.shape[1],
                    z_local.shape[1],
                ),
                "z_norm": all_views.detach().float().norm(dim=-1).mean(),
            }

        return total_loss

    @staticmethod
    def _chance_alignment(
        num_global: int,
        num_local: int,
    ) -> float:
        if num_global < 1:
            return 0.0

        total = num_global + num_local

        return (
            (num_global - 1)
            + num_local * (num_global + 1) / num_global
        ) / total

    def _alignment_ratio(
        self,
        alignment_loss: torch.Tensor,
        all_views: torch.Tensor,
        num_global: int,
        num_local: int,
    ) -> torch.Tensor:
        chance = self._chance_alignment(num_global, num_local)
        detached = all_views.detach().float()

        if chance <= 0.0 or detached.shape[0] < 2:
            return torch.zeros((), device=all_views.device)

        variance = detached.reshape(-1, detached.shape[-1]).var(dim=0).mean()

        return alignment_loss.detach() / (chance * variance + 1e-8)







if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 16
    num_global_views = 3
    num_local_views = 4
    latent_dim = 32

    z_global = torch.randn(
        batch_size,
        num_global_views,
        latent_dim,
        device=device,
        requires_grad=True,
    )

    z_local = torch.randn(
        batch_size,
        num_local_views,
        latent_dim,
        device=device,
        requires_grad=True,
    )

    config = {
        "return_components": True,
        "sigreg_weight": 0.05,
        "alignment": {
            "detach_center": False,
            "feature_reduction": "mean",
            "reduction": "mean",
        },
        "sigreg": {
            "num_slices": 64,
            "num_integration_points": 17,
            "integration_range": [-5.0, 5.0],
            "reduction": "mean",
            "base_seed": 0,
            "sync_ddp": False,
            "force_float32": True,
        },
    }

    criterion = PretrainingLoss(config).to(device)

    total_loss, metrics = criterion(
        z_global=z_global,
        z_local=z_local,
        global_step=0,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(metrics["alignment_loss"])
    assert torch.isfinite(metrics["sigreg_loss"])

    total_loss.backward()

    assert z_global.grad is not None
    assert z_local.grad is not None
    assert torch.isfinite(z_global.grad).all()
    assert torch.isfinite(z_local.grad).all()
    assert z_global.grad.abs().sum() > 0
    assert z_local.grad.abs().sum() > 0

    print(f"Device: {device}")
    print(f"Total loss: {metrics['total_loss'].item():.6f}")
    print(
        "Alignment loss: "
        f"{metrics['alignment_loss'].item():.6f}"
    )
    print(
        "SIGReg loss: "
        f"{metrics['sigreg_loss'].item():.6f}"
    )
    print(
        "Global gradient norm: "
        f"{z_global.grad.norm().item():.6f}"
    )
    print(
        "Local gradient norm: "
        f"{z_local.grad.norm().item():.6f}"
    )
    print("All tests passed.")