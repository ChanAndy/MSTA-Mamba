#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msta_mamba.config import load_config
from msta_mamba.data import ManifestVideoDataset
from msta_mamba.engine import load_checkpoint
from msta_mamba.metrics import localization_errors
from msta_mamba.model import MSTAMamba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg = config["dataset"], config["model"]
    dataset = ManifestVideoDataset(
        data_cfg["manifest"], args.split, data_cfg["num_frames"], data_cfg["image_size"]
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False)
    model = MSTAMamba(
        num_classes=model_cfg["num_classes"],
        num_frames=model_cfg["num_frames"],
        backbone_name=model_cfg["backbone"],
        pretrained=False,
        feature_dims=tuple(model_cfg["feature_dims"]),
        feature_dim=model_cfg["feature_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    errors: dict[str, list[int]] = {"ed": [], "es": []}
    with torch.inference_mode():
        for batch in tqdm(loader):
            logits, _ = model(batch["video"].to(device))
            current = localization_errors(logits.cpu(), batch["labels"], batch["valid"])
            for key in errors:
                errors[key].extend(current[key])
    result = {
        key: {"n": len(values), "mae": statistics.fmean(values), "sd": statistics.pstdev(values)}
        for key, values in errors.items()
        if values
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

