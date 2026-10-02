from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ManifestVideoDataset(Dataset):
    """Load clips described by the repository's portable CSV manifest."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        num_frames: int,
        image_size: int,
        training: bool = False,
    ):
        self.num_frames = num_frames
        self.training = training
        manifest = Path(manifest)
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"].lower() == split.lower()]
        self.root = manifest.parent
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")

        operations: list[torch.nn.Module] = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if training:
            operations.extend(
                [
                    v2.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0)),
                    v2.RandomHorizontalFlip(),
                    v2.RandomVerticalFlip(),
                    v2.RandomRotation(15),
                ]
            )
        else:
            operations.append(v2.Resize((image_size, image_size)))
        operations.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = v2.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def _read_video(self, path: Path) -> list[torch.Tensor]:
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        capture.release()
        if not frames:
            raise RuntimeError(f"Could not decode video: {path}")
        return frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = self.root / video_path
        frames = self._read_video(video_path)
        labels = json.loads(row["labels"])
        if len(labels) != len(frames):
            raise ValueError(f"Label/frame mismatch for {row['video_id']}")

        # Prepared manifests normally contain exactly 32 frames. Random start is
        # retained for compatible longer clips; evaluation remains deterministic.
        max_start = max(0, len(frames) - self.num_frames)
        start = random.randint(0, max_start) if self.training else 0
        frames = frames[start : start + self.num_frames]
        labels = labels[start : start + self.num_frames]
        valid = [True] * len(frames)
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))
            labels.append(0)
            valid.append(False)

        clip = torch.stack(frames)
        # v2 applies identical random geometry to all frames in one tensor.
        clip = self.transform(clip)
        return {
            "video": clip,
            "labels": torch.tensor(labels, dtype=torch.long),
            "valid": torch.tensor(valid, dtype=torch.bool),
            "video_id": row["video_id"],
        }

