"""
Module: evaluate.py

Test-set evaluation helpers (formerly ``test.py``, renamed because ``test`` shadows the
Python standard-library package of the same name and confuses pytest).

Functions:
    - predict_probs: Softmax probabilities for every sample of a DataLoader.
    - predict_views: Softmax probabilities averaged over several views (test-time augmentation).
    - metrics_from_probs: Predictions, accuracy and loss from (averaged) probabilities, which
      is also how model ensembles are scored.
    - test: Single- or multi-view evaluation returning labels, predictions, accuracy and loss.
    - get_test_report: Per-class precision / recall / F1 (scikit-learn classification report).
    - get_confusion_matrix: Full ``n_classes x n_classes`` confusion matrix.
"""

import torch
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def predict_probs(model, dataloader, device):
    """
    Return softmax probabilities and labels for every sample, in DataLoader order.

    Args:
        model (nn.Module): Trained model (already on ``device``).
        dataloader (DataLoader): Non-shuffled DataLoader.
        device (torch.device): Device to run on.

    Returns:
        tuple: (probs (N, n_classes) float tensor on CPU, labels (N,) long tensor on CPU)
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16,
                                         enabled=device.type == 'cuda'):
        for x_batch, y_batch in tqdm(dataloader, desc='test', leave=False):
            logits = model(x_batch.to(device, non_blocking=True)).float()
            probs.append(F.softmax(logits, dim=1).cpu())
            labels.append(y_batch)
    return torch.cat(probs), torch.cat(labels)


def predict_views(model, dataloaders, device):
    """
    Average softmax probabilities over several views of the same test samples.

    Each DataLoader must iterate the same videos in the same order but may show a different
    *view* (temporal offset, horizontal flip). Averaging the softmax outputs over the views is
    standard multi-view test-time augmentation.

    Args:
        model (nn.Module): Trained model (already on ``device``).
        dataloaders (DataLoader | list[DataLoader]): One DataLoader per view.
        device (torch.device): Device to run on.

    Returns:
        tuple: (mean_probs (N, n_classes), labels (N,)) as CPU tensors.

    Raises:
        ValueError: If the views do not iterate identical label sequences.
    """
    if not isinstance(dataloaders, (list, tuple)):
        dataloaders = [dataloaders]
    mean_probs, labels = None, None
    for loader in dataloaders:
        probs, view_labels = predict_probs(model, loader, device)
        if labels is not None and not torch.equal(labels, view_labels):
            raise ValueError('All test views must iterate the same samples in the same order.')
        labels = view_labels
        mean_probs = probs if mean_probs is None else mean_probs + probs
    return mean_probs / len(dataloaders), labels


def metrics_from_probs(probs, labels):
    """
    Turn class probabilities into predictions, accuracy and loss.

    Args:
        probs (Tensor): (N, n_classes) probabilities (e.g. averaged over views or models).
        labels (Tensor): (N,) ground-truth labels.

    Returns:
        tuple: (targets, outputs, accuracy, loss)
            - targets (list[int]): Ground-truth labels.
            - outputs (list[int]): Predicted labels.
            - accuracy (float): Fraction of correct predictions over *all* samples.
            - loss (float): Mean negative log-likelihood of ``probs``.
    """
    preds = probs.argmax(dim=1)
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels).item()
    accuracy = (preds == labels).float().mean().item()
    return labels.tolist(), preds.tolist(), accuracy, loss


def test(model, dataloaders, device):
    """
    Evaluate a model on the test set (single view, or multi-view TTA with several loaders).

    Args:
        model (nn.Module): Trained model (already on ``device``).
        dataloaders (DataLoader | list[DataLoader]): One DataLoader per view.
        device (torch.device): Device to run on.

    Returns:
        tuple: (targets, outputs, accuracy, loss), see ``metrics_from_probs``.
    """
    return metrics_from_probs(*predict_views(model, dataloaders, device))


def get_test_report(targets, outputs, target_names):
    """
    Build a per-class classification report.

    Args:
        targets (list[int]): Ground-truth labels.
        outputs (list[int]): Predicted labels.
        target_names (list[str]): Class names, indexed by label.

    Returns:
        str: Text report with precision, recall, F1 and support for each class.
    """
    return classification_report(targets, outputs, labels=list(range(len(target_names))),
                                 target_names=target_names, digits=3, zero_division=0)


def get_confusion_matrix(targets, outputs, n_classes):
    """
    Compute the confusion matrix (rows = true class, columns = predicted class).

    Args:
        targets (list[int]): Ground-truth labels.
        outputs (list[int]): Predicted labels.
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: ``(n_classes, n_classes)`` integer matrix.
    """
    return confusion_matrix(targets, outputs, labels=list(range(n_classes)))
