import torch

from model import DownstreamModel
from utils import DownstreamTrainer
from utils.downstream_parser import parse_config
from eeg_preprocessing.loaders import build_loaders


def build_model(model_config, num_classes, downstream_config, device):
    training = downstream_config["training"]
    head_type = training.get("head", "linear")
    head_config = downstream_config.get("heads", {}).get(head_type, {})
    channel_mode = downstream_config["data"]["channel_mode"]
    input_channels = None if channel_mode == "mapped" else downstream_config["dataset_cfg"].get("num_channels")
    model = DownstreamModel(
        backbone_config=model_config["model"],
        num_classes=num_classes,
        freeze_backbone=training.get("freeze_backbone", True),
        freeze_channel_adaptor=training.get("freeze_channel_adaptor", True),
        probe=head_type,
        probe_config=head_config,
        input_channels=input_channels,
    ).to(device)
    model.load_weights(downstream_config["checkpoints_path"], map_location=device)
    model.print_summary()
    return model


def build_loaders_from_config(downstream_config):
    data = downstream_config["data"]
    train_loader, val_loader, test_loader = build_loaders(
        dataset=downstream_config["dataset"],
        split_ratio=data["split_ratio"],
        batch_size=data["batch_size"],
        seed=data["seed"],
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        persistent_workers=data["persistent_workers"],
        channel_mode=data["channel_mode"],
        return_ch_names=data["return_ch_names"],
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def main(model_config, downstream_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_cfg = downstream_config["dataset_cfg"]
    num_classes = dataset_cfg["n_classes"]
    if "class_weights" in dataset_cfg:
        downstream_config["training"]["class_weights"] = dataset_cfg["class_weights"]

    loaders = build_loaders_from_config(downstream_config)
    model = build_model(model_config, num_classes, downstream_config, device)

    print(f"dataset: {downstream_config['dataset']} | num_classes: {num_classes} | device: {device}")

    trainer = DownstreamTrainer(model, loaders, downstream_config)
    trainer.train()


if __name__ == "__main__":
    model_config, downstream_config = parse_config()
    main(model_config, downstream_config)
