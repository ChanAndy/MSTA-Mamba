from __future__ import annotations

import torch


def localization_errors(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
) -> dict[str, list[int]]:
    """Absolute frame error for ED (class 1) and, when present, ES (class 2)."""
    probabilities = logits.softmax(dim=-1)
    output: dict[str, list[int]] = {"ed": [], "es": []}
    for batch_index in range(logits.shape[0]):
        length = int(valid[batch_index].sum())
        for name, phase_class in (("ed", 1), ("es", 2)):
            if phase_class >= logits.shape[-1]:
                continue
            truth = torch.where((labels[batch_index, :length] == phase_class))[0]
            if not len(truth):
                continue
            prediction = int(probabilities[batch_index, :length, phase_class].argmax())
            output[name].append(int((truth - prediction).abs().min()))
    return output

