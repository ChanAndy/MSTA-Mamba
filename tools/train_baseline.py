#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msta_mamba.baseline import MambaVisionBaseline
from msta_mamba.config import load_config
from msta_mamba.engine import save_checkpoint, seed_everything
from msta_mamba.paper_data import EchoNetPhaseDataset, POCUSFrameListDataset


def make_datasets(config: dict):
    data = config["dataset"]
    if data["name"] == "pocus":
        train = POCUSFrameListDataset(data["train_list"], data["image_size"], True, data["num_frames"])
        test = POCUSFrameListDataset(data["test_list"], data["image_size"], False, data["num_frames"])
    elif data["name"] == "echonet":
        common = dict(
            root=data["root"], metadata=data["metadata"], image_size=data["image_size"],
            num_frames=data["num_frames"], seed=config["seed"]
        )
        train = EchoNetPhaseDataset(split="train", training=True, **common)
        test = EchoNetPhaseDataset(split="test", training=False, **common)
    else:
        raise ValueError(f"Unknown dataset: {data['name']}")
    return train, test


def evaluate(model, loader, device, dataset_name: str) -> dict[str, float]:
    model.eval()
    by_video: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            logits = model(batch["video"].to(device)).softmax(-1).cpu()
            for i, video_id in enumerate(batch["video_id"]):
                valid = batch["valid"][i]
                by_video[video_id].append(
                    (int(batch["offset"][i]), logits[i][valid], batch["labels"][i][valid])
                )

    errors: dict[int, list[int]] = defaultdict(list)
    for parts in by_video.values():
        parts.sort(key=lambda item: item[0])
        scores = torch.cat([item[1] for item in parts])
        labels = torch.cat([item[2] for item in parts])
        for phase_class in range(1, scores.shape[-1]):
            truth = torch.where(labels == phase_class)[0]
            if len(truth):
                prediction = scores[:, phase_class].argmax()
                errors[phase_class].append(int((truth - prediction).abs().min()))

    result = {"videos": len(by_video)}
    names = {1: "ed", 2: "es"}
    for phase_class, values in errors.items():
        name = names[phase_class]
        result[f"{name}_mae"] = statistics.fmean(values)
        result[f"{name}_sd"] = statistics.pstdev(values)
    if dataset_name == "pocus" and 1 in errors:
        result.update({f"ed_at_{k}": sum(value <= k for value in errors[1]) / len(errors[1]) for k in range(5)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    train_set, test_set = make_datasets(config)
    settings = config["train"]
    micro_batch = settings.get("micro_batch_size", settings["batch_size"])
    if settings["batch_size"] % micro_batch:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    accumulation_steps = settings["batch_size"] // micro_batch
    train_loader = DataLoader(
        train_set, batch_size=micro_batch, shuffle=True,
        num_workers=settings["workers"], pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=micro_batch, shuffle=False,
        num_workers=settings["workers"], pin_memory=True,
    )
    model = MambaVisionBaseline(
        num_classes=config["dataset"]["num_classes"],
        backbone_name=config["model"]["backbone"],
        feature_dims=tuple(config["model"]["feature_dims"]),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=settings["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer, T_max=settings["epochs"] - settings["warmup_epochs"])
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[settings["warmup_epochs"]])
    weights = torch.tensor(settings["class_weights"], device=device)
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start_epoch, best = 0, math.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"], state.get("best_mae", best)

    metrics_path = output / "metrics.csv"
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{settings['epochs']}")):
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            logits = model(video)
            loss = F.cross_entropy(logits[valid], labels[valid], weight=weights)
            (loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        scheduler.step()
        metrics = evaluate(model, test_loader, device, config["dataset"]["name"])
        metrics.update(epoch=epoch + 1, train_loss=statistics.fmean(losses))
        fieldnames = list(metrics)
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)
        primary = metrics["ed_mae"]
        checkpoint = {
            "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best_mae": min(best, primary),
            "config": config,
        }
        save_checkpoint(output / "last.pth", model, **checkpoint)
        if primary < best:
            best = primary
            save_checkpoint(output / "best.pth", model, **checkpoint)
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
