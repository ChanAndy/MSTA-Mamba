from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def event_centers(labels: torch.Tensor, phase_class: int, valid: torch.Tensor) -> list[int]:
    """Merge consecutive positive frames and return one center per event."""
    indices = torch.where((labels == phase_class) & valid)[0].tolist()
    if not indices:
        return []
    events: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if index == events[-1][-1] + 1:
            events[-1].append(index)
        else:
            events.append([index])
    return [event[len(event) // 2] for event in events]


class KeyCycleConstraintLoss(nn.Module):
    """KCCL from paper equations (12)-(21)."""

    def __init__(
        self,
        class_weights: list[float],
        alpha: float = 0.4,
        beta: float = 0.2,
        rhythm_weight: float = 0.5,
    ):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.alpha = alpha
        self.beta = beta
        self.rhythm_weight = rhythm_weight

    def forward(
        self, logits: torch.Tensor, phase: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = valid.bool()
        ce = F.cross_entropy(logits[valid], labels[valid], weight=self.class_weights)
        response = phase.mean(dim=-1)
        anchors: list[torch.Tensor] = []
        rhythms: list[torch.Tensor] = []
        consistency: list[torch.Tensor] = []

        phase_classes = range(1, logits.shape[-1])
        for batch_index in range(labels.shape[0]):
            for phase_class in phase_classes:
                centers = event_centers(labels[batch_index], phase_class, valid[batch_index])
                target = 1.0 if phase_class == 1 else 0.0
                for center in centers:
                    anchors.append((response[batch_index, center] - target).abs())
                    if center >= 3 and valid[batch_index, center - 3 : center + 1].all():
                        delta = phase[batch_index, center - 3 : center + 1].diff(dim=0)
                        if phase_class == 1:  # Eq. (16), ED
                            rhythms.append(F.relu(delta[1:] - delta[:-1]).mean())
                        else:  # Eq. (17), ES
                            rhythms.append(F.relu(delta[:-1] - delta[1:]).mean())
                for left, right in zip(centers, centers[1:]):
                    consistency.append(F.mse_loss(phase[batch_index, right], phase[batch_index, left]))

        zero = logits.sum() * 0.0
        anchor = torch.stack(anchors).mean() if anchors else zero
        rhythm = torch.stack(rhythms).mean() if rhythms else zero
        consist = torch.stack(consistency).mean() if consistency else zero
        single = anchor + self.rhythm_weight * rhythm
        total = ce + self.alpha * single + self.beta * consist
        return total, {"ce": ce, "anchor": anchor, "rhythm": rhythm, "consistency": consist}

