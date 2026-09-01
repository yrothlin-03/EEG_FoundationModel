import mne
import numpy as np
import torch

from utils import PretrainingTrainer, parse_config
from eeg_preprocessing.loaders import TARGET_CHS, build_pretraining_loaders
from model import EEGFM


RANGE_KEYS = (
    "noise_std",
    "global_amp_scale",
    "temp_shift",
    "canal_scale_gain",
    "time_mask_ratio",
    "freq_mask_ratio",
    "spatial_mask_ratio",
)


def build_positions(channels=TARGET_CHS, montage_name="standard_1020"):
    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]
    missing = [c for c in channels if c not in ch_pos and c.capitalize() not in ch_pos]
    if missing:
        raise KeyError(f"canaux absents du montage {montage_name}: {missing}")
    pos = [ch_pos.get(c, ch_pos.get(c.capitalize())) for c in channels]
    return torch.from_numpy(np.stack(pos)).float()


def build_loaders(data_config):
    kwargs = dict(data_config)
    kwargs["split_ratio"] = tuple(kwargs["split_ratio"])
    for key in RANGE_KEYS:
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    train_loader, val_loader = build_pretraining_loaders(**kwargs)
    if val_loader is None:
        raise ValueError("aucun loader de validation construit, vérifier data.split_ratio.")
    return {"train": train_loader, "val": val_loader}


def build_model(model_config, pos):
    return EEGFM(pos=pos, **model_config)


def main(model_config, pretraining_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = build_loaders(pretraining_config["data"])
    pos = build_positions()

    batch = next(iter(loaders["train"]))
    x, presence = batch[0], batch[1]
    n_views = (len(batch[2]), len(batch[3])) if len(batch) > 3 else (0, 0)
    batch_size, n_channels, seq_len = x.shape
    print(f"batch: {tuple(x.shape)} | presence: {tuple(presence.shape)} "
          f"| vues globales/locales: {n_views[0]}/{n_views[1]}")

    if n_channels != pos.shape[0]:
        raise ValueError(
            f"{n_channels} canaux dans les données mais {pos.shape[0]} positions "
            f"fournies. Vérifier data.channel_mode."
        )

    model_kwargs = dict(model_config["model"])
    model_kwargs.setdefault("window", seq_len)
    if model_kwargs["window"] != seq_len:
        raise ValueError(
            f"model.window={model_kwargs['window']} mais les fenêtres du loader "
            f"font {seq_len} échantillons."
        )

    model = build_model(model_kwargs, pos).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params entraînables: {n_params/1e6:.1f}M | n_tokens: {model.n_tokens} "
          f"| token_stride: {model.token_stride()} | z_dim: {model.z_dim}")

    trainer = PretrainingTrainer(model, loaders, pretraining_config)

    resume = pretraining_config["trainer"].get("resume_from")
    if resume:
        trainer.load_checkpoint(resume)
        print(f"reprise depuis {resume} à l'epoch {trainer.epoch}")
        trainer.pretrain(start_epoch=trainer.epoch)
    else:
        trainer.pretrain()


if __name__ == "__main__":
    model_config, pretraining_config = parse_config()
    main(model_config, pretraining_config)
