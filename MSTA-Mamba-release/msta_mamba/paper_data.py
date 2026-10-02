from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


def clip_transform(image_size: int, training: bool) -> v2.Compose:
    operations = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if training:
        operations += [
            v2.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(15),
        ]
    else:
        operations += [v2.Resize((image_size, image_size))]
    operations += [v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return v2.Compose(operations)


class POCUSFrameListDataset(Dataset):
    def __init__(self, list_file: str, image_size: int, training: bool, num_frames: int = 32):
        self.num_frames = num_frames
        self.transform = clip_transform(image_size, training)
        videos: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        with Path(list_file).open(encoding="utf-8") as handle:
            for line in handle:
                frame_path, label = line.strip().rsplit(maxsplit=1)
                stem = Path(frame_path).stem
                video_id, frame_text = stem.rsplit("-", 1)
                videos[video_id].append((int(frame_text), frame_path, int(label)))
        self.samples = []
        for video_id, frames in videos.items():
            frames.sort()
            for start in range(0, len(frames), num_frames):
                self.samples.append((video_id, start, frames[start : start + num_frames]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        video_id, start, records = self.samples[index]
        images = [torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1) for _, path, _ in records]
        labels = [label for _, _, label in records]
        valid = [True] * len(records)
        while len(images) < self.num_frames:
            images.append(torch.zeros_like(images[0]))
            labels.append(0)
            valid.append(False)
        return {
            "video": self.transform(torch.stack(images)),
            "labels": torch.tensor(labels),
            "valid": torch.tensor(valid),
            "video_id": video_id,
            "offset": start,
        }


class EchoNetPhaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        metadata: str,
        split: str,
        image_size: int,
        training: bool,
        num_frames: int = 32,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.num_frames = num_frames
        self.training = training
        self.seed = seed
        self.transform = clip_transform(image_size, training)
        with Path(metadata).open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["Split"].upper() == split.upper()]
        self.rows = []
        for row in rows:
            ed, es, total = int(row["ED"]), int(row["ES"]), int(row["NumberOfFrames"])
            if es < ed or es - ed > num_frames:
                continue
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        ed, es, total = int(row["ED"]), int(row["ES"]), int(row["NumberOfFrames"])
        # Submitted-code convention: interval length is ES-ED and the final
        # sampled frame carries the ES label. This retains the 74 exclusions
        # reported in the manuscript instead of 87 with inclusive endpoints.
        es_target = es - 1
        first = max(0, es_target - self.num_frames + 1)
        last = min(ed, max(0, total - self.num_frames))
        rng = random.Random(self.seed + index)
        start = rng.randint(first, last) if self.training else (first + last) // 2
        capture = cv2.VideoCapture(str(self.root / "Videos" / f"{row['FileName']}.avi"))
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        capture.release()
        if not decoded:
            raise RuntimeError(f"Failed to decode {row['FileName']}")
        # A small number of EchoNet videos have fewer than 32 total frames.
        # Follow the submitted preprocessing and wrap neighboring frames.
        images = [decoded[(start + offset) % len(decoded)] for offset in range(self.num_frames)]
        labels = torch.zeros(self.num_frames, dtype=torch.long)
        labels[ed - start] = 1
        labels[es_target - start] = 2
        return {
            "video": self.transform(torch.stack(images)),
            "labels": labels,
            "valid": torch.ones(self.num_frames, dtype=torch.bool),
            "video_id": row["FileName"],
            "offset": start,
        }
