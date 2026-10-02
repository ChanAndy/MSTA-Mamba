from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .modules import SpatioTemporalAlignment, TemporalContextMixer


def load_mambavision(name: str, pretrained: bool = True) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    if pretrained:
        return AutoModel.from_pretrained(name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    return AutoModel.from_config(config, trust_remote_code=True)


def load_mamba(feature_dim: int) -> nn.Module:
    from mamba_ssm import Mamba

    return Mamba(d_model=feature_dim, d_state=16, d_conv=4, expand=2)


class MSTAMamba(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int = 32,
        backbone_name: str = "nvidia/MambaVision-S-1K",
        pretrained: bool = True,
        feature_dims: tuple[int, ...] = (96, 192, 384, 768),
        feature_dim: int = 256,
        backbone: nn.Module | None = None,
        temporal_factory: Callable[[int], nn.Module] = load_mamba,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.feature_dims = feature_dims
        self.backbone = backbone or load_mambavision(backbone_name, pretrained)
        self.tcm = nn.ModuleList(TemporalContextMixer(dim) for dim in feature_dims)
        self.phase_model = temporal_factory(feature_dim)
        self.sta = SpatioTemporalAlignment(list(feature_dims), feature_dim)
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim * len(feature_dims), 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _backbone_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        output = self.backbone(x)
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            features = output[1]
        elif hasattr(output, "hidden_states"):
            features = output.hidden_states
        else:
            raise RuntimeError("Backbone must return four stage feature maps")
        features = list(features)[-4:]
        if len(features) != 4 or any(feature.ndim != 4 for feature in features):
            raise RuntimeError("Expected four [B*T, C, H, W] backbone features")
        return features

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape [B, T, 3, H, W]")
        batch, time, channels, height, width = video.shape
        if time != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, received {time}")

        raw = self._backbone_features(video.reshape(batch * time, channels, height, width))
        mixed = []
        for feature, tcm, channels_i in zip(raw, self.tcm, self.feature_dims):
            shaped = feature.reshape(batch, time, channels_i, *feature.shape[-2:])
            mixed.append(tcm(shaped).reshape(batch * time, channels_i, *feature.shape[-2:]))

        # Eq. (8): p5 provides the global phase-aware sequence.
        top_map = self.sta.output[-1](self.sta.lateral[-1](torch.cat((raw[-1], mixed[-1]), dim=1)))
        top = top_map.mean(dim=(-2, -1)).reshape(batch, time, -1)
        phase = self.phase_model(top)

        pyramid = self.sta(raw, mixed, phase, batch, time)
        target_size = pyramid[1].shape[-2:]
        pyramid = [
            feature if feature.shape[-2:] == target_size else F.interpolate(
                feature, target_size, mode="bilinear", align_corners=False
            )
            for feature in pyramid
        ]
        logits = self.head(torch.cat(pyramid, dim=1)).reshape(batch, time, -1)
        return logits, phase
