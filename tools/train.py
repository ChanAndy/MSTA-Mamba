#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msta_mamba.config import load_config
from msta_mamba.data import ManifestVideoDataset
from msta_mamba.engine import save_checkpoint, seed_everything
from msta_mamba.losses import KeyCycleConstraintLoss
from msta_mamba.model import MSTAMamba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_cfg, model_cfg, train_cfg = config["dataset"], config["model"], config["train"]
    train_set = ManifestVideoDataset(
        data_cfg["manifest"], "train", data_cfg["num_frames"], data_cfg["image_size"], True
    )
    loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["workers"],
        pin_memory=True,
    )
    model = MSTAMamba(
        num_classes=model_cfg["num_classes"],
        num_frames=model_cfg["num_frames"],
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        feature_dims=tuple(model_cfg["feature_dims"]),
        feature_dim=model_cfg["feature_dim"],
    ).to(device)
    criterion = KeyCycleConstraintLoss(**config["loss"]).to(device)
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=train_cfg["warmup_epochs"])
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, train_cfg["epochs"] - train_cfg["warmup_epochs"])
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[train_cfg["warmup_epochs"]]
    )
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{train_cfg['epochs']}")
        for batch in progress:
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, phase = model(video)
            loss, parts = criterion(logits, phase, labels, valid)
            loss.backward()
            optimizer.step()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", ce=f"{parts['ce'].item():.4f}")
        scheduler.step()
        save_checkpoint(output / "last.pth", model, epoch=epoch + 1, config=config)
        print(f"epoch={epoch + 1} mean_loss={running / len(loader):.6f}")


if __name__ == "__main__":
    main()

