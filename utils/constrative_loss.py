import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_features, image_features):
        eeg = F.normalize(eeg_features, dim=-1)
        image = F.normalize(image_features, dim=-1)
        logits = eeg @ image.t() / self.temperature
        targets = torch.arange(eeg.size(0), device=eeg.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        logits = features @ features.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        self_mask = torch.ones_like(mask)
        self_mask.fill_diagonal_(0)
        mask = mask * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos.clamp(min=1)
        loss = -mean_log_prob_pos[pos > 0]
        return loss.mean() if loss.numel() > 0 else logits.sum() * 0.0




class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        z = torch.cat([z_i, z_j], dim=0)

        logits = z @ z.t() / self.temperature
        logits.fill_diagonal_(float("-inf"))

        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        return F.cross_entropy(logits, targets)

