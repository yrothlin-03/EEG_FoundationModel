import mne
import numpy as np
import torch

from utils import PretrainingTrainer, parse_config
from eeg_preprocessing.loaders import build_pretraining_loaders, EEGAugment
from model import EEGFM


TARGET_CHS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
              "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ"]


def build_positions(channels=TARGET_CHS, montage_name="standard_1020"):
    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]
    missing = [c for c in channels if c not in ch_pos and c.capitalize() not in ch_pos]
    if missing:
        raise KeyError(f"canaux absents du montage {montage_name}: {missing}")
    pos = [ch_pos.get(c, ch_pos.get(c.capitalize())) for c in channels]
    return torch.from_numpy(np.stack(pos)).float()


def build_model(model_config, pos):
    return EEGFM(pos=pos, **model_config)


def main(model_config, pretraining_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = pretraining_config["data"]
    augment_enabled = data["augment"]
    augmenter = EEGAugment() if augment_enabled else None
    train_loader, val_loader, _ = build_pretraining_loaders(
        datasets=data["datasets"],
        batch_size=data["batch_size"],
        split_ratio=data["split_ratio"],
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        persistent_workers=data["persistent_workers"],
        seed=data["seed"],
        channel_mode=data["channel_mode"],
        augment=augment_enabled,
        augmenter=augmenter,
        scale_factor=data["scale_factor"],
        variant=data["variant"],
    )

    channels = data.get("target_channels") or TARGET_CHS
    pos = build_positions(channels)

    batch = next(iter(train_loader))
    if isinstance(batch, dict):
        x = batch["x"]
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    batch_size, n_channels, seq_len = x.shape
    print(f"batch: {tuple(x.shape)} | channels: {n_channels} | seq_len: {seq_len}")

    if n_channels != pos.shape[0]:
        raise ValueError(
            f"{n_channels} canaux dans les données mais {pos.shape[0]} positions "
            f"fournies. Vérifier data.target_channels / channel_mode."
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
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"params entraînables: {n_params/1e6:.1f}M (encodeur {n_enc/1e6:.1f}M, "
          f"têtes {(n_params-n_enc)/1e6:.1f}M) | "
          f"n_tokens: {model.n_tokens} | token_stride: {model.token_stride()}")

    loaders = {"train": train_loader, "val": val_loader}
    trainer = PretrainingTrainer(model, loaders, pretraining_config)

    resume = pretraining_config["training"].get("resume_from")
    if resume:
        trainer.load_checkpoint(resume)
        print(f"reprise depuis {resume} à l'epoch {trainer.epoch}")
        trainer.pretrain(start_epoch=trainer.epoch)
    else:
        trainer.pretrain()


if __name__ == "__main__":
    model_config, pretraining_config = parse_config()
    main(model_config, pretraining_config)