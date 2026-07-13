import torch

from model import ContrastiveModel
from utils import ContrastiveTrainer, CLIPEmbedder
from utils.contrastive_parser import parse_config
from eeg_preprocessing.loaders.teeg_dataset import build_contrastive_loader
from eeg_preprocessing.loaders.multi_dataset import get_pretraining_loader


def build_clip_embedder(contrastive_config, device):
    clip_cfg = contrastive_config.get("clip", {})
    return CLIPEmbedder(
        model_name=clip_cfg.get("model_name", "ViT-L-14"),
        pretrained=clip_cfg.get("pretrained", "laion2b_s32b_b82k"),
        normalize=clip_cfg.get("normalize", True),
        device=device,
    )


def build_model(model_config, contrastive_config, device):
    training = contrastive_config["training"]
    head = dict(contrastive_config.get("head", {}))
    clip_dim = head.pop("clip_dim")
    channel_mode = contrastive_config["data"]["channel_mode"]
    input_channels = None if channel_mode == "mapped" else contrastive_config["dataset_cfg"].get("num_channels")
    model = ContrastiveModel(
        backbone_config=model_config["model"],
        clip_dim=clip_dim,
        freeze_backbone=training.get("freeze_backbone", True),
        freeze_channel_adaptor=training.get("freeze_channel_adaptor", True),
        head_config=head,
        input_channels=input_channels,
    ).to(device)
    model.load_weights(contrastive_config["checkpoints_path"], map_location=device)
    return model


def build_infonce_loaders(contrastive_config, clip_embedder):
    data = contrastive_config["data"]
    dataset_cfg = contrastive_config["dataset_cfg"]
    loader_cfg = {
        "lmdb_path": dataset_cfg["dataset_path"],
        "img_root": dataset_cfg["img_root"],
        "transform": clip_embedder.preprocess,
        "return_meta": False,
        "split_ratio": data["split_ratio"],
        "batch_size": data["batch_size"],
        "seed": data["seed"],
        "num_workers": data["num_workers"],
        "pin_memory": data["pin_memory"],
        "persistent_workers": data["persistent_workers"],
        "channel_mode": data["channel_mode"],
    }
    train_loader, val_loader, test_loader = build_contrastive_loader(loader_cfg)
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def build_supcon_loaders(contrastive_config):
    data = contrastive_config["data"]
    datasets_cfg = contrastive_config["datasets_cfg"]
    names = contrastive_config["supcon_datasets"]
    lmdb_paths = [datasets_cfg[n]["dataset_path"] for n in names]
    print(f"supcon datasets ({len(names)}): {names}")

    train_loader, val_loader = get_pretraining_loader(
        lmdb_paths=lmdb_paths,
        split_ratio=data["split_ratio"],
        batch_size=data["batch_size"],
        seed=data["seed"],
        num_workers=data["num_workers"],
        pin_memory=data["pin_memory"],
        persistent_workers=data["persistent_workers"],
        channel_mode=data["channel_mode"],
        scale_factor=1.0,
        return_labels=True,
    )
    return {"train": train_loader, "val": val_loader, "test": None}


def build_loaders_from_config(contrastive_config, clip_embedder):
    loss_name = contrastive_config["training"].get("loss", "infonce")
    if loss_name == "infonce":
        return build_infonce_loaders(contrastive_config, clip_embedder)
    return build_supcon_loaders(contrastive_config)


def main(model_config, contrastive_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_name = contrastive_config["training"].get("loss", "infonce")

    clip_embedder = build_clip_embedder(contrastive_config, device) if loss_name == "infonce" else None
    loaders = build_loaders_from_config(contrastive_config, clip_embedder)
    model = build_model(model_config, contrastive_config, device)

    data_desc = contrastive_config["dataset"] if loss_name == "infonce" else contrastive_config["supcon_datasets"]
    print(f"data: {data_desc} | loss: {loss_name} | device: {device}")

    trainer = ContrastiveTrainer(model, loaders, contrastive_config, clip_embedder=clip_embedder)
    trainer.train()


if __name__ == "__main__":
    model_config, contrastive_config = parse_config()
    main(model_config, contrastive_config)
