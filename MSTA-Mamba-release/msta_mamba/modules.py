from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalContextMixer(nn.Module):
    """Paper Sec. II-A: depth-wise temporal mixing on the first 2/G channels."""

    def __init__(self, channels: int, groups: int = 8, kernel_size: int = 3):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.group_channels = channels // groups
        self.temporal_conv = nn.Conv1d(
            self.group_channels,
            self.group_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=self.group_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("TCM expects [B, T, C, H, W]")
        batch, time, _, height, width = x.shape
        mixed_channels = 2 * self.group_channels
        selected, identity = x[:, :, :mixed_channels], x[:, :, mixed_channels:]
        selected = selected.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, mixed_channels, time
        )
        first, second = selected.split(self.group_channels, dim=1)
        selected = torch.cat((self.temporal_conv(first), self.temporal_conv(second)), dim=1)
        selected = selected.reshape(batch, height, width, mixed_channels, time)
        selected = selected.permute(0, 4, 3, 1, 2)
        return torch.cat((selected, identity), dim=2)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class SpatioTemporalAlignment(nn.Module):
    """Paper Sec. II-B, equations (6)-(11)."""

    def __init__(self, in_dims: list[int], feature_dim: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            ConvNormAct(2 * channels, feature_dim, 1) for channels in in_dims
        )
        self.output = nn.ModuleList(
            ConvNormAct(feature_dim, feature_dim, 3) for _ in in_dims
        )

    @staticmethod
    def _sequence(feature: torch.Tensor, batch: int, time: int) -> torch.Tensor:
        return feature.mean(dim=(-2, -1)).reshape(batch, time, -1)

    @staticmethod
    def align(sequence: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # cdist is batched, so videos are never compared across the batch.
        distances = torch.cdist(sequence, reference)
        weights = torch.softmax(-distances, dim=-1)
        return weights @ reference

    def forward(
        self,
        raw: list[torch.Tensor],
        mixed: list[torch.Tensor],
        phase: torch.Tensor,
        batch: int,
        time: int,
    ) -> list[torch.Tensor]:
        lateral = [layer(torch.cat((a, b), dim=1)) for layer, a, b in zip(self.lateral, raw, mixed)]

        pyramid: list[torch.Tensor] = [torch.empty(0)] * len(lateral)
        pyramid[-1] = self.output[-1](lateral[-1])
        for index in range(len(lateral) - 2, -1, -1):
            up = F.interpolate(
                pyramid[index + 1], size=lateral[index].shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid[index] = self.output[index](lateral[index] + up)

        reference = self._sequence(pyramid[-1], batch, time)
        aligned_pyramid = []
        for feature in reversed(pyramid):
            sequence = self._sequence(feature, batch, time)
            aligned = self.align(sequence, reference)
            modulation = (aligned * phase).reshape(batch * time, -1, 1, 1)
            aligned_pyramid.append(feature + modulation)
            reference = sequence
        return list(reversed(aligned_pyramid))

