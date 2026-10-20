"""Tests for the training-loop helpers."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import batch_correct_preds, build_optimizer, build_scheduler, run_epoch, train


class TinyModel(nn.Module):
    """Linear classifier exposing the backbone / head parameter split used by the trainer."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        """Classify a batch of feature vectors."""
        return self.head(torch.relu(self.backbone(x)))

    def backbone_parameters(self):
        """Backbone parameters."""
        return list(self.backbone.parameters())

    def head_parameters(self):
        """Head parameters."""
        return list(self.head.parameters())


def test_batch_correct_preds():
    """Correct predictions are counted from argmax logits."""
    logits = torch.tensor([[2.0, 0.1], [0.0, 1.0], [3.0, -1.0]])
    assert batch_correct_preds(logits, torch.tensor([0, 1, 1])) == 2


def test_optimizer_uses_discriminative_lrs():
    """The backbone LR is lr * backbone_lr_mult and the head LR is lr."""
    opt = build_optimizer(TinyModel(), lr=1e-3, backbone_lr_mult=0.1)
    assert [g['lr'] for g in opt.param_groups] == pytest.approx([1e-4, 1e-3])


def test_scheduler_warmup_then_cosine_to_zero():
    """The LR ramps up during warm-up, peaks at the base LR, then decays to ~0."""
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]['lr'])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[5] < lrs[9] == pytest.approx(1.0)
    assert lrs[50] < lrs[20] and lrs[-1] < 0.01


def test_run_epoch_trains_and_reports_full_dataset_accuracy():
    """Training reduces the loss, and accuracy is computed over every sample."""
    torch.manual_seed(0)
    x = torch.randn(30, 4)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 1).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)  # last batch has 6 samples
    model = TinyModel()
    opt = build_optimizer(model, lr=5e-2, backbone_lr_mult=1.0)
    crit = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    first = run_epoch(model, crit, loader, device)
    for _ in range(30):
        run_epoch(model, crit, loader, device, optimizer=opt)
    last = run_epoch(model, crit, loader, device)
    assert last['loss'] < first['loss']
    assert len(last['preds']) == len(last['targets']) == 30
    expected_acc = sum(p == t for p, t in zip(last['preds'], last['targets'])) / 30
    assert last['acc'] == pytest.approx(expected_acc)


def test_train_without_val_keeps_last_epoch(tmp_path):
    """Final train+val mode (no val loader) runs every epoch and checkpoints the last one."""
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,))),
                        batch_size=4)
    model = TinyModel()
    opt = build_optimizer(model, lr=1e-2)
    sched = build_scheduler(opt, total_steps=3 * len(loader), warmup_steps=1)
    _, history = train({'train': loader, 'val': None}, model, nn.CrossEntropyLoss(), opt, sched,
                       torch.device('cpu'), str(tmp_path), n_epochs=3, patience=1)
    assert len(history['train_loss']) == 3 and not history['val_acc']
    assert history['best_epoch'] == 3
    assert torch.load(tmp_path / 'best.pt')['epoch'] == 3
