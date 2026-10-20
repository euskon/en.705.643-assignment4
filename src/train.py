"""
Module: train.py

Training loop for the UCF50 video classifiers.

Functions:
    - build_optimizer: AdamW with a lower learning rate for the pretrained backbone than for
      the newly initialised head (discriminative learning rates).
    - build_scheduler: linear warm-up followed by cosine decay, stepped once per batch.
    - batch_correct_preds: number of correct top-1 predictions in a mini-batch.
    - run_epoch: one pass over a DataLoader (training or evaluation) with mixed precision.
    - train: full training run with per-epoch validation, best-checkpoint saving, early
      stopping and metric logging (e.g. to Weights & Biases).
"""

import math
import os

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def build_optimizer(model, lr, weight_decay=0.01, backbone_lr_mult=0.1):
    """
    Create an AdamW optimizer with discriminative learning rates.

    The pretrained backbone uses ``lr * backbone_lr_mult``, so the ImageNet features are only
    fine-tuned gently. The randomly initialised head (LSTM / classifier) uses the full ``lr``.

    Args:
        model (nn.Module): Model exposing ``backbone_parameters()`` and ``head_parameters()``.
        lr (float): Head learning rate.
        weight_decay (float): Decoupled weight decay.
        backbone_lr_mult (float): Backbone learning-rate multiplier.

    Returns:
        torch.optim.AdamW: The optimizer (group 0 = backbone, group 1 = head).
    """
    groups = [
        {'params': model.backbone_parameters(), 'lr': lr * backbone_lr_mult, 'name': 'backbone'},
        {'params': model.head_parameters(), 'lr': lr, 'name': 'head'},
    ]
    return optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, total_steps, warmup_steps):
    """
    Linear warm-up from 0 to the base LR, then cosine decay to 0, stepped once per batch.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of optimisation steps in the run.
        warmup_steps (int): Number of warm-up steps.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The scheduler.
    """
    def lr_lambda(step):
        """Return the LR multiplier for ``step``."""
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)


def batch_correct_preds(output, target):
    """
    Count the correct top-1 predictions in a mini-batch.

    Args:
        output (Tensor): Logits of shape (batch_size, n_classes).
        target (Tensor): Labels of shape (batch_size,).

    Returns:
        int: Number of correct predictions.
    """
    return (output.argmax(dim=1) == target).sum().item()


def run_epoch(model, criterion, dataloader, device, optimizer=None, *, scheduler=None,
              scaler=None, grad_clip=1.0, desc=None):
    """
    Run one epoch of training (if ``optimizer`` is given) or evaluation.

    Uses fp16 autocast on CUDA. The loss is averaged per *sample*: each batch's mean loss is
    weighted by its size, so a smaller last batch does not skew the epoch average.

    Args:
        model (nn.Module): Model to run.
        criterion (callable): Loss function with mean reduction.
        dataloader (DataLoader): Data source.
        device (torch.device): Device to run on.
        optimizer (Optimizer, optional): Enables training mode and parameter updates.
        scheduler (LRScheduler, optional): Stepped after every optimizer step.
        scaler (torch.amp.GradScaler, optional): Loss scaler for fp16 training.
        grad_clip (float): Max gradient norm (``0`` disables clipping).
        desc (str, optional): tqdm progress-bar label.

    Returns:
        dict: ``{'loss': float, 'acc': float, 'targets': list[int], 'preds': list[int]}``
    """
    # pylint: disable=too-many-arguments,too-many-locals
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == 'cuda'
    total_loss, total_correct, total_seen = 0.0, 0, 0
    targets, preds = [], []

    with torch.set_grad_enabled(training):
        for x_batch, y_batch in tqdm(dataloader, desc=desc, leave=False):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = model(x_batch)
                loss = criterion(output.float(), y_batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n = y_batch.size(0)
            total_loss += loss.item() * n
            total_correct += batch_correct_preds(output, y_batch)
            total_seen += n
            targets.extend(y_batch.tolist())
            preds.extend(output.argmax(dim=1).tolist())

    return {'loss': total_loss / max(1, total_seen), 'acc': total_correct / max(1, total_seen),
            'targets': targets, 'preds': preds}


def train(dataloaders, model, criterion, optimizer, scheduler, device, out_dir, n_epochs=30,
          patience=None, log_fn=None, checkpoint_extra=None):
    """
    Train with per-epoch validation and keep the weights with the best validation accuracy.

    Model selection uses **only** the validation split. The test split is never used
    here, so the reported test accuracy is unbiased.

    If ``dataloaders`` has no ``'val'`` entry (final training on train+val with a recipe
    already chosen on validation), the full fixed schedule runs and the *last* epoch's
    weights are saved. Early stopping is disabled in that case.

    Args:
        dataloaders (dict): ``{'train': DataLoader, 'val': DataLoader | None}``.
        model (nn.Module): Model to train (already on ``device``).
        criterion (callable): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Per-batch LR scheduler.
        device (torch.device): Device to train on.
        out_dir (str): Directory where ``best.pt`` is written.
        n_epochs (int): Maximum number of epochs.
        patience (int, optional): Stop early when validation accuracy has not improved for
            this many epochs. ``None`` disables early stopping.
        log_fn (callable, optional): ``log_fn(metrics: dict, epoch: int)`` called once per
            epoch, e.g. a ``wandb.log`` wrapper.
        checkpoint_extra (dict, optional): Extra entries saved in the checkpoint (config,
            class names, ...) so evaluation can rebuild the model.

    Returns:
        tuple: (model, history)
            - model (nn.Module): The model with the best validation (or final) weights.
            - history (dict): Per-epoch lists of ``train_loss``, ``train_acc``,
              ``val_loss``, ``val_acc`` (empty without validation) and ``lr``, plus
              ``best_epoch``.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, 'best.pt')
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [],
               'best_epoch': 0}
    best_val_acc, epochs_since_best = -1.0, 0
    has_val = dataloaders.get('val') is not None

    for epoch in range(1, n_epochs + 1):
        tr = run_epoch(model, criterion, dataloaders['train'], device, optimizer,
                       scheduler=scheduler, scaler=scaler, desc=f'epoch {epoch} train')
        lr = optimizer.param_groups[-1]['lr']  # head LR reached at the end of this epoch
        history['train_loss'].append(tr['loss'])
        history['train_acc'].append(tr['acc'])
        history['lr'].append(lr)
        metrics = {'epoch': epoch, 'lr': lr, 'train/loss': tr['loss'], 'train/acc': tr['acc']}
        msg = (f"Epoch {epoch}/{n_epochs} | lr {lr:.2e} | train loss {tr['loss']:.4f} "
               f"acc {100 * tr['acc']:.2f}%")

        if has_val:
            val = run_epoch(model, criterion, dataloaders['val'], device,
                            desc=f'epoch {epoch} val')
            history['val_loss'].append(val['loss'])
            history['val_acc'].append(val['acc'])
            improved = val['acc'] > best_val_acc
            if improved:
                best_val_acc, epochs_since_best = val['acc'], 0
            else:
                epochs_since_best += 1
            metrics.update({'val/loss': val['loss'], 'val/acc': val['acc'],
                            'val/best_acc': best_val_acc})
            msg += (f" | val loss {val['loss']:.4f} acc {100 * val['acc']:.2f}%"
                    f"{' *best*' if improved else ''}")
        else:
            improved = True  # no validation: always keep the latest weights

        if improved:
            history['best_epoch'] = epoch
            torch.save({'model_state': model.state_dict(), 'epoch': epoch,
                        'val_acc': best_val_acc if has_val else None,
                        **(checkpoint_extra or {})}, best_path)
        print(msg, flush=True)
        if log_fn is not None:
            log_fn(metrics, epoch)
        if has_val and patience is not None and epochs_since_best >= patience:
            print(f'Early stopping: no val improvement for {patience} epochs.')
            break

    model.load_state_dict(torch.load(best_path, map_location=device)['model_state'])
    return model, history
