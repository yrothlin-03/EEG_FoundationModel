import torch

from utils import PretrainingTrainer, parse_config
from eeg_preprocessing.loaders import build_pretraining_loaders, EEGAugment
from model import EEGVAE, Discriminator


def build_model(model_config):
    return EEGVAE(**model_config)


def build_discriminator(disc_config):
    return Discriminator(**disc_config)


def main(model_config, pretraining_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augment_enabled = pretraining_config["data"]["augment"]
    augmenter = EEGAugment() if augment_enabled else None
    train_loader, val_loader, _ = build_pretraining_loaders(
        datasets=pretraining_config["data"]["datasets"],
        batch_size=pretraining_config["data"]["batch_size"],
        split_ratio=pretraining_config["data"]["split_ratio"],
        num_workers=pretraining_config["data"]["num_workers"],
        pin_memory=pretraining_config["data"]["pin_memory"],
        persistent_workers=pretraining_config["data"]["persistent_workers"],
        seed=pretraining_config["data"]["seed"],
        channel_mode=pretraining_config["data"]["channel_mode"],
        augment=augment_enabled,
        augmenter=augmenter,
        scale_factor=pretraining_config["data"]["scale_factor"]
    )

    model = build_model(model_config["model"]).to(device)
    train_loader = train_loader
    batch = next(iter(train_loader))
    x_in, x_tgt = batch
    seq_len = x_in.shape[-1]
    batch_size = x_in.shape[0]
    print(f"Input shape: {x_in.shape}, Target shape: {x_tgt.shape}, Sequence length: {seq_len}")
    model.summary(seq_len=seq_len, batch_size=batch_size, device=device)
    discriminator = build_discriminator(model_config["discriminator"]).to(device)

    loaders = {"train": train_loader, "val": val_loader}
    trainer = PretrainingTrainer(model, discriminator, loaders, pretraining_config)
    trainer.pretrain()


if __name__ == "__main__":
    model_config, pretraining_config = parse_config()
    main(model_config, pretraining_config)
