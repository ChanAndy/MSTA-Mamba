#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msta_mamba.config import load_config
from msta_mamba.engine import load_checkpoint
from msta_mamba.model import MSTAMamba


def read_clip(path: str, frames: int, image_size: int) -> tuple[torch.Tensor, int]:
    capture = cv2.VideoCapture(path)
    decoded = []
    while len(decoded) < frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded.append(torch.from_numpy(frame).permute(2, 0, 1))
    capture.release()
    valid = len(decoded)
    if not decoded:
        raise RuntimeError(f"Could not decode {path}")
    while len(decoded) < frames:
        decoded.append(torch.zeros_like(decoded[0]))
    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size)),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(torch.stack(decoded)).unsqueeze(0), valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    data_cfg, model_cfg = config["dataset"], config["model"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MSTAMamba(
        num_classes=model_cfg["num_classes"],
        num_frames=model_cfg["num_frames"],
        backbone_name=model_cfg["backbone"],
        pretrained=False,
        feature_dims=tuple(model_cfg["feature_dims"]),
        feature_dim=model_cfg["feature_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    clip, valid = read_clip(args.input, data_cfg["num_frames"], data_cfg["image_size"])
    model.eval()
    with torch.inference_mode():
        probabilities = model(clip.to(device))[0].softmax(-1)[0, :valid]
    print(f"ED frame: {int(probabilities[:, 1].argmax())}")
    if model_cfg["num_classes"] == 3:
        print(f"ES frame: {int(probabilities[:, 2].argmax())}")


if __name__ == "__main__":
    main()

