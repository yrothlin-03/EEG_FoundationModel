import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

def masked_reconstruction_loss(pred, target, mask):
    per_elem = F.smooth_l1_loss(pred, target, reduction="none")
    m = mask.unsqueeze(1).to(per_elem.dtype)
    denom = m.sum().clamp_min(1.0) * pred.shape[1]
    return (per_elem * m).sum() / denom

def stft_loss(pred, target, eps=1e-5, sizes=(32, 128, 384)):
    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()

        losses = []
        for n_fft in sizes:
            hop_length = n_fft // 4
            win_length = n_fft
            window = torch.hann_window(win_length, device=pred.device)

            p_frames = pred.unfold(-1, n_fft, hop_length) * window
            t_frames = target.unfold(-1, n_fft, hop_length) * window

            p_mag = torch.fft.rfft(p_frames, dim=-1).abs() + eps
            t_mag = torch.fft.rfft(t_frames, dim=-1).abs() + eps

            sc_loss = torch.norm(p_mag - t_mag, p="fro") / torch.norm(t_mag, p="fro")
            mag_loss = F.l1_loss(torch.log(p_mag), torch.log(t_mag))

            losses.append(sc_loss + mag_loss)

        if not losses:
            return torch.tensor(0.0, device=pred.device)

        return torch.stack(losses).mean()

def vae_loss(posterior):
    loss=posterior.kl()
    return torch.mean(loss)


def generator_loss(logits_fake):
    logits_fake = logits_fake.clamp(-50.0, 50.0)
    return -logits_fake.mean()

def discriminator_loss(logits_real, logits_fake):
    logits_real = logits_real.clamp(-50.0, 50.0)
    logits_fake = logits_fake.clamp(-50.0, 50.0)
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)




class PretrainingLoss(nn.Module):
    def __init__(
        self,
        rec_weight=1.0,
        kl_weight=1e-6,
        stft_weight=1.0,
        stft_sizes=(32, 128, 384),
        sample_rate=200,
        masked_loss_weight=1.0,
        visible_loss_weight=0.0,
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.stft_weight = stft_weight
        self.stft_sizes = tuple(stft_sizes)
        self.sample_rate = sample_rate
        self.masked_loss_weight = masked_loss_weight
        self.visible_loss_weight = visible_loss_weight

    def forward(self, pred, target, posterior, mask=None):
        if mask is not None:
            rec_masked = masked_reconstruction_loss(pred, target, mask)
            rec_visible = masked_reconstruction_loss(pred, target, 1.0 - mask)
            rec = self.masked_loss_weight * rec_masked + self.visible_loss_weight * rec_visible
        else:
            rec = reconstruction_loss(pred, target)
        kl = vae_loss(posterior)
        stft = stft_loss(pred, target, sizes=self.stft_sizes) if self.stft_weight > 0 else pred.new_zeros(())

        total = (
            self.rec_weight * rec
            + self.kl_weight * kl
            + self.stft_weight * stft
        )

        logs = {
            "loss": total.detach(),
            "rec_loss": rec.detach(),
            "vae_regul": kl.detach(),
            "stft_loss": stft.detach(),
        }
        if mask is not None:
            logs["rec_loss_masked"] = rec_masked.detach()
            logs["rec_loss_visible"] = rec_visible.detach()
        if hasattr(posterior, "perplexity"):
            logs["perplexity"] = posterior.perplexity.detach()
        return total, logs

