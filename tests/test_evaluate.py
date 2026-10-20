"""Tests for (multi-view) test-set evaluation."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import test as evaluate_test


class FixedLogits(nn.Module):
    """'Model' whose input already is the logits, so views fully control the predictions."""

    def forward(self, x):
        """Return the input unchanged."""
        return x


def _loader(logits, labels):
    """A non-shuffled loader that yields ``(logits, labels)``."""
    return DataLoader(TensorDataset(logits, labels), batch_size=2)


def test_multi_view_averages_probabilities():
    """Averaging softmax outputs fixes a sample that a single view gets wrong."""
    labels = torch.tensor([0, 1, 1])
    view_a = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.2, 0.0]])  # sample 2 slightly wrong
    view_b = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 4.0]])  # sample 2 strongly right
    device = torch.device('cpu')
    _, preds_a, acc_a, _ = evaluate_test(FixedLogits(), _loader(view_a, labels), device)
    views = [_loader(view_a, labels), _loader(view_b, labels)]
    targets, preds, acc, loss = evaluate_test(FixedLogits(), views, device)
    assert preds_a == [0, 1, 0] and acc_a == pytest.approx(2 / 3)
    assert targets == [0, 1, 1] and preds == [0, 1, 1] and acc == pytest.approx(1.0)
    assert loss > 0


def test_views_must_be_aligned():
    """Views that iterate different samples are rejected instead of silently mis-averaged."""
    logits = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        evaluate_test(FixedLogits(), [_loader(logits, torch.tensor([0, 1, 1])),
                                      _loader(logits, torch.tensor([1, 0, 1]))],
                      torch.device('cpu'))
