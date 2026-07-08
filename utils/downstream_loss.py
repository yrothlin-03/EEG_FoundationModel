import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamLoss(nn.Module):
    def __init__(self, loss="label_smoothing", label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.loss = loss
        self.label_smoothing = label_smoothing if loss == "label_smoothing" else 0.0
        if class_weights is not None and not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, target):
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        logs = {"loss": loss.detach()}
        return loss, logs
