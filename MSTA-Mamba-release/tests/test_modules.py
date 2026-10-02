import torch
from torch import nn

from msta_mamba.losses import KeyCycleConstraintLoss, event_centers
from msta_mamba.model import MSTAMamba
from msta_mamba.modules import SpatioTemporalAlignment, TemporalContextMixer


def test_tcm_preserves_shape_and_identity_channels():
    module = TemporalContextMixer(16, groups=8)
    x = torch.randn(2, 5, 16, 3, 3)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, :, 4:], x[:, :, 4:])


def test_alignment_does_not_mix_batch_items():
    sequence = torch.tensor([[[0.0], [1.0]], [[100.0], [101.0]]])
    reference = sequence.clone()
    aligned = SpatioTemporalAlignment.align(sequence, reference)
    assert aligned[0].max() < 2
    assert aligned[1].min() > 99


def test_consecutive_labels_are_one_event():
    labels = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0])
    assert event_centers(labels, 1, torch.ones_like(labels, dtype=torch.bool)) == [2, 6]


def test_kccl_is_differentiable():
    logits = torch.randn(2, 8, 3, requires_grad=True)
    phase = torch.randn(2, 8, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 1, 0, 2, 0, 0, 0], [0, 1, 0, 0, 2, 0, 0, 0]])
    valid = torch.ones(2, 8, dtype=torch.bool)
    loss, _ = KeyCycleConstraintLoss([1, 10, 10])(logits, phase, labels, valid)
    loss.backward()
    assert logits.grad is not None
    assert phase.grad is not None


class DummyBackbone(nn.Module):
    def forward(self, x):
        n = x.shape[0]
        features = [
            torch.randn(n, 8, 16, 16, device=x.device),
            torch.randn(n, 16, 8, 8, device=x.device),
            torch.randn(n, 32, 4, 4, device=x.device),
            torch.randn(n, 64, 2, 2, device=x.device),
        ]
        return None, features


def test_model_forward_with_injected_backbone():
    model = MSTAMamba(
        num_classes=3,
        num_frames=4,
        feature_dims=(8, 16, 32, 64),
        feature_dim=64,
        backbone=DummyBackbone(),
        temporal_factory=lambda _: nn.Identity(),
    )
    logits, phase = model(torch.randn(2, 4, 3, 64, 64))
    assert logits.shape == (2, 4, 3)
    assert phase.shape == (2, 4, 64)

