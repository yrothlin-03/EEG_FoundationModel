import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

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

def eeg_scales(sample_rate=200, w0=6.0, bands=None, voices_per_octave=4):
    if bands is None:
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
            "gamma": (30, min(100, sample_rate / 2 - 1)),
        }

    f_min = min(b[0] for b in bands.values())
    f_max = max(b[1] for b in bands.values())

    s_min = w0 * sample_rate / (2 * torch.pi * f_max)
    s_max = w0 * sample_rate / (2 * torch.pi * f_min)

    n_octaves = torch.log2(torch.tensor(s_max / s_min))
    n_scales = int(n_octaves * voices_per_octave) + 1

    scales = s_min * (2 ** (torch.arange(n_scales) / voices_per_octave))
    return scales.tolist()


def cwt_loss(pred, target, eps=1e-5, sample_rate=200, w0=6.0, voices_per_octave=4):
    device = pred.device
    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()
        orig_shape = pred.shape
        p = pred.reshape(-1, 1, orig_shape[-1])
        t = target.reshape(-1, 1, orig_shape[-1])

        scales = eeg_scales(sample_rate, w0=w0, voices_per_octave=voices_per_octave)

        losses = []
        for scale in scales:
            kernel_size = int(10 * scale)
            if kernel_size % 2 == 0:
                kernel_size += 1
            half = kernel_size // 2
            x = torch.arange(-half, half + 1, device=device, dtype=p.dtype) / scale

            gaussian = torch.exp(-0.5 * x ** 2)
            real = gaussian * torch.cos(w0 * x)
            imag = gaussian * torch.sin(w0 * x)
            norm = 1.0 / (scale ** 0.5)
            real = (real * norm).view(1, 1, -1)
            imag = (imag * norm).view(1, 1, -1)

            p_real = F.conv1d(p, real, padding=half)
            p_imag = F.conv1d(p, imag, padding=half)
            t_real = F.conv1d(t, real, padding=half)
            t_imag = F.conv1d(t, imag, padding=half)

            p_mag = torch.sqrt(p_real ** 2 + p_imag ** 2 + eps ** 2)
            t_mag = torch.sqrt(t_real ** 2 + t_imag ** 2 + eps ** 2)

            sc_loss = torch.norm(p_mag - t_mag, p="fro") / torch.norm(t_mag, p="fro")
            mag_loss = F.l1_loss(torch.log(p_mag + eps), torch.log(t_mag + eps))

            losses.append(sc_loss + mag_loss)

        if not losses:
            return torch.tensor(0.0, device=device)

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
        cwt_weight=0.0,
        stft_sizes=(32, 128, 384),
        sample_rate=200,
        cwt_w0=6.0,
        cwt_voices_per_octave=4,
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.stft_weight = stft_weight
        self.cwt_weight = cwt_weight
        self.stft_sizes = tuple(stft_sizes)
        self.sample_rate = sample_rate
        self.cwt_w0 = cwt_w0
        self.cwt_voices_per_octave = cwt_voices_per_octave

    def forward(self, pred, target, posterior):
        rec = reconstruction_loss(pred, target)
        kl = vae_loss(posterior)
        stft = stft_loss(pred, target, sizes=self.stft_sizes) if self.stft_weight > 0 else pred.new_zeros(())
        cwt = (
            cwt_loss(
                pred, target,
                sample_rate=self.sample_rate,
                w0=self.cwt_w0,
                voices_per_octave=self.cwt_voices_per_octave,
            )
            if self.cwt_weight > 0
            else pred.new_zeros(())
        )

        total = (
            self.rec_weight * rec
            + self.kl_weight * kl
            + self.stft_weight * stft
            + self.cwt_weight * cwt
        )

        logs = {
            "loss": total.detach(),
            "rec_loss": rec.detach(),
            "vae_regul": kl.detach(),
            "stft_loss": stft.detach(),
            "cwt_loss": cwt.detach(),
        }
        if hasattr(posterior, "perplexity"):
            logs["perplexity"] = posterior.perplexity.detach()
        return total, logs

