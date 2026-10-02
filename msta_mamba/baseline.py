from __future__ import annotations

import torch
from torch import nn

from .model import load_mamba, load_mambavision


class MambaVisionBaseline(nn.Module):
    """Table II baseline: MambaVision features with scale-wise temporal modeling."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "nvidia/MambaVision-S-1K",
        pretrained: bool = True,
        feature_dims: tuple[int, ...] = (96, 192, 384, 768),
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.backbone = backbone or load_mambavision(backbone_name, pretrained)
        self.temporal = nn.ModuleList(load_mamba(dim) for dim in feature_dims)
        self.head = nn.Linear(sum(feature_dims), num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = video.shape
        output = self.backbone(video.reshape(batch * time, channels, height, width))
        features = list(output[1] if isinstance(output, (tuple, list)) else output.hidden_states)[-4:]
        sequences = []
        for feature, temporal, dim in zip(features, self.temporal, self.feature_dims):
            sequence = feature.mean(dim=(-2, -1)).reshape(batch, time, dim)
            sequences.append(temporal(sequence))
        return self.head(torch.cat(sequences, dim=-1))

