"""
Module: run.py

Main entry point for training and evaluating UCF50 video classifiers.

    # train (run ``prepare_data.py`` first) - see train.sh
    python src/run.py --mode train --model_type lrcn --batch_size 8

    # evaluate a saved checkpoint on the held-out test groups - see test.sh
    python src/run.py --mode eval --ckpt runs/<run_name>/best.pt

Train mode:
    1. Load the group-based train / val / test split from ``<frame_dir>/splits.json``.
    2. Build clip-consistent transforms, ``UCF50Dataset`` objects and DataLoaders.
    3. Build the model (``lrcn`` or ``3dcnn``), AdamW with discriminative LRs, and a
       warm-up + cosine LR schedule, using a label-smoothed cross-entropy loss.
    4. Train, keeping the checkpoint with the best *validation* accuracy, and log every epoch
       to Weights & Biases.
    5. Evaluate that best checkpoint once on the *test* split and log the test metrics.

Eval mode:
    Rebuild the model from the configuration stored in the checkpoint, evaluate on the
    test split, and print accuracy, a per-class report and the most-confused class pairs.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from evaluate import get_confusion_matrix, get_test_report, metrics_from_probs, predict_views
from models import build_model
from train import build_optimizer, build_scheduler, train
from utils import (compose_data_transforms, seed_everything, test_dloaders, train_val_dloaders,
                   transform_stats)
from video_datasets import UCF50Dataset, load_splits

# Settings stored in the checkpoint; eval mode rebuilds the exact model/input pipeline from them.
MODEL_KEYS = ('model_type', 'n_classes', 'cnn_backbone', 'pretrained', 'rnn_hidden_size',
              'rnn_n_layers', 'bidirectional', 'temporal_pool', 'dropout', 'fr_per_vid',
              'img_size')


def args_parser(argv=None):
    """
    Parse command-line arguments.

    Args:
        argv (list[str], optional): Argument list (defaults to ``sys.argv``).

    Returns:
        argparse.Namespace: Parsed arguments (see ``--help`` for every option).
    """
    p = argparse.ArgumentParser(description='UCF50 video classification (train / eval)')
    p.add_argument('-m', '--mode', choices=['train', 'eval'], required=True)
    # data
    p.add_argument('-fd', '--frame_dir', default='data/UCF50_frames',
                   help='Extracted frames + splits.json (output of prepare_data.py)')
    p.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Frames sampled per clip')
    p.add_argument('--img_size', type=int, default=None,
                   help='Square input size (default 224 for lrcn, 160 for 3dcnn)')
    p.add_argument('-nw', '--num_workers', type=int, default=6, help='DataLoader workers')
    p.add_argument('--test_clips', type=int, default=3,
                   help='Temporal views per test video for test-time augmentation (1 = off)')
    p.add_argument('--test_hflip', action=argparse.BooleanOptionalAction, default=True,
                   help='Also average over horizontally flipped test views')
    # model
    p.add_argument('-mt', '--model_type', choices=['lrcn', '3dcnn'], default='lrcn')
    p.add_argument('-cnn', '--cnn_backbone', default='convnext_tiny',
                   help='LRCN backbone: resnet18/34/50/101/152, convnext_tiny, convnext_small, '
                        'efficientnet_v2_s')
    p.add_argument('-p', '--pretrained', action=argparse.BooleanOptionalAction, default=True,
                   help='Start from ImageNet weights (--no-pretrained to train from scratch)')
    p.add_argument('-rhs', '--rnn_hidden_size', type=int, default=512)
    p.add_argument('-rnl', '--rnn_n_layers', type=int, default=1)
    p.add_argument('--bidirectional', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--temporal_pool', choices=['mean', 'last'], default='mean')
    p.add_argument('-d', '--dropout', type=float, default=0.5)
    # optimisation
    p.add_argument('-bs', '--batch_size', type=int, default=8)
    p.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help='Head learning rate')
    p.add_argument('--backbone_lr_mult', type=float, default=0.1,
                   help='Backbone LR = learning_rate * backbone_lr_mult')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('-ne', '--n_epochs', type=int, default=20)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=6,
                   help='Early-stopping patience in epochs (0 disables)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_on_val', action='store_true',
                   help='Final model: train on train+val with a fixed schedule (no validation, '
                        'no early stopping; the last epoch is kept). Choose the recipe on val '
                        'first.')
    # bookkeeping
    p.add_argument('--eval_split', choices=['val', 'test'], default='test',
                   help='Split scored in eval mode (use val to choose ensembles / settings)')
    p.add_argument('-c', '--ckpt', nargs='+',
                   help='Checkpoint(s) to evaluate (eval mode); several = averaged ensemble')
    p.add_argument('--out_dir', default='runs', help='Where run folders are created')
    p.add_argument('--run_name', default=None, help='Run folder / W&B run name')
    p.add_argument('--wandb_project', default='ucf50-video-classification')
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    return p.parse_args(argv)


def init_wandb(args, cfg, job_type):
    """
    Start a Weights & Biases run (or return ``None`` when W&B is disabled).

    Args:
        args (argparse.Namespace): Parsed arguments (project, mode, run name).
        cfg (dict): Configuration logged with the run.
        job_type (str): ``'train'`` or ``'eval'``.

    Returns:
        wandb.sdk.wandb_run.Run | None: The active run.
    """
    if args.wandb_mode == 'disabled':
        return None
    import wandb  # pylint: disable=import-outside-toplevel
    run = wandb.init(project=args.wandb_project, name=cfg['run_name'], config=cfg,
                     job_type=job_type, mode=args.wandb_mode)
    run.define_metric('epoch')
    for metric in ('train/*', 'val/*', 'lr'):
        run.define_metric(metric, step_metric='epoch')
    return run


def make_test_loaders(test_samples, cfg, batch_size, num_workers, n_clips=1, hflip=False):
    """
    Build one test DataLoader per evaluation view.

    View ``k`` of ``n_clips`` samples each temporal segment at relative offset
    ``(k + 0.5) / n_clips``. With ``hflip`` every temporal view is also evaluated mirrored.
    ``n_clips=1, hflip=False`` gives the single center view.

    Args:
        test_samples (list[tuple[str, int]]): ``(video_dir, label)`` test samples.
        cfg (dict): Model configuration (``model_type``, ``img_size``, ``fr_per_vid``).
        batch_size (int): Mini-batch size.
        num_workers (int): DataLoader workers.
        n_clips (int): Number of temporal views.
        hflip (bool): Add mirrored views.

    Returns:
        list[DataLoader]: One loader per view (``n_clips * (2 if hflip else 1)``).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    h, w, mean, std = transform_stats(cfg['model_type'], cfg['img_size'])
    loaders = []
    for flip in ([False, True] if hflip else [False]):
        _, eval_tf = compose_data_transforms(h, w, mean, std, eval_hflip=flip)
        for k in range(n_clips):
            dataset = UCF50Dataset(test_samples, cfg['fr_per_vid'], eval_tf, train=False,
                                   offset=(k + 0.5) / n_clips)
            loaders.append(test_dloaders(dataset, batch_size, cfg['model_type'],
                                         num_workers)['test'])
    return loaders


def report_test(probs, labels, classes, out_dir, wandb_run=None, prefix='test', n_views=1):
    """
    Score test probabilities, print and save the reports, and log them to W&B.

    Args:
        probs (Tensor): (N, n_classes) test probabilities (averaged over views / models).
        labels (Tensor): (N,) ground-truth labels.
        classes (list[str]): Class names.
        out_dir (str): Directory for ``<prefix>_report.txt`` / ``<prefix>_confusion.npy``.
        wandb_run (wandb Run, optional): Active W&B run.
        prefix (str): Metric prefix, e.g. ``'test'`` or ``'lrcn/test_tta'`` (``/`` is
            replaced by ``_`` in file names).
        n_views (int): Number of averaged predictions (for the printed summary only).

    Returns:
        tuple: (accuracy, loss)
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    targets, outputs, accuracy, loss = metrics_from_probs(probs, labels)
    report = get_test_report(targets, outputs, classes)
    cmat = get_confusion_matrix(targets, outputs, len(classes))
    off_diag = [(int(cmat[i, j]), classes[i], classes[j])
                for i in range(len(classes)) for j in range(len(classes)) if i != j and cmat[i, j]]

    summary = (f'[{prefix}] {n_views} view(s) | accuracy {100 * accuracy:.2f}% | '
               f'loss {loss:.4f} | n={len(targets)}')
    print(summary)
    print(report)
    print('Most confused (true -> predicted):')
    for count, true_cls, pred_cls in sorted(off_diag, reverse=True)[:10]:
        print(f'  {count:3d}  {true_cls} -> {pred_cls}')

    os.makedirs(out_dir, exist_ok=True)
    file_stem = os.path.join(out_dir, prefix.replace('/', '_'))
    with open(f'{file_stem}_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary + '\n\n' + report)
    np.save(f'{file_stem}_confusion.npy', cmat)

    if wandb_run is not None:
        import wandb  # pylint: disable=import-outside-toplevel
        wandb_run.log({f'{prefix}/acc': accuracy, f'{prefix}/loss': loss,
                       f'{prefix}/confusion_matrix': wandb.plot.confusion_matrix(
                           y_true=targets, preds=outputs, class_names=classes,
                           title=f'{prefix} confusion matrix')})
        wandb_run.summary.update({f'{prefix}/acc': accuracy, f'{prefix}/loss': loss})
    return accuracy, loss


def evaluate_views(model, test_samples, cfg, args, device, classes, out_dir, wandb_run,
                   tag='', split='test'):
    """
    Report single-view metrics and, if enabled, multi-view (TTA) metrics on a held-out split.

    Args:
        model (nn.Module): Trained model on ``device``.
        test_samples (list[tuple[str, int]]): ``(video_dir, label)`` samples to score.
        cfg (dict): Model configuration.
        args (argparse.Namespace): Parsed arguments (batch size, workers, TTA flags).
        device (torch.device): Device.
        classes (list[str]): Class names.
        out_dir (str): Directory for the reports.
        wandb_run (wandb Run | None): Active W&B run.
        tag (str): Prefix for metric names, e.g. ``'lrcn/'`` when scoring ensemble members.
        split (str): Name of the scored split (``'test'`` or ``'val'``), used in metric names.

    Returns:
        tuple: (results, probs)
            - results (dict): ``<tag><split>_acc``, ``<tag><split>_loss`` and, with TTA,
              ``<tag><split>_tta_acc`` / ``<tag><split>_tta_loss``.
            - probs (dict): ``{'<split>': (probs, labels), '<split>_tta': (probs, labels)}``,
              used for ensembling.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    view_sets = {split: make_test_loaders(test_samples, cfg, args.batch_size, args.num_workers)}
    if args.test_clips > 1 or args.test_hflip:
        view_sets[f'{split}_tta'] = make_test_loaders(test_samples, cfg, args.batch_size,
                                                  args.num_workers, args.test_clips,
                                                  args.test_hflip)
    results, probs = {}, {}
    for kind, loaders in view_sets.items():
        probs[kind] = predict_views(model, loaders, device)
        acc, loss = report_test(*probs[kind], classes, out_dir, wandb_run, tag + kind,
                                len(loaders))
        results.update({f'{tag}{kind}_acc': acc, f'{tag}{kind}_loss': loss})
    return results, probs


def run_train(args, device):
    """
    Train a model as configured by ``args`` and evaluate it on the test split.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Device to train on.

    Returns:
        dict: Final metrics (``best_epoch``, ``best_val_acc`` and the ``evaluate_views``
        test metrics).
    """
    # pylint: disable=too-many-locals
    seed_everything(args.seed)
    splits, classes, split_meta = load_splits(args.frame_dir)
    cfg = vars(args).copy()
    cfg.update(n_classes=len(classes), split_meta=split_meta,
               n_train=len(splits['train']), n_val=len(splits['val']), n_test=len(splits['test']),
               img_size=transform_stats(args.model_type, args.img_size)[0])
    cfg['run_name'] = args.run_name or f"{args.model_type}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.out_dir, cfg['run_name'])
    os.makedirs(run_dir, exist_ok=True)

    h, w, mean, std = transform_stats(args.model_type, cfg['img_size'])
    tr_tf, eval_tf = compose_data_transforms(h, w, mean, std)
    train_samples = splits['train'] + (splits['val'] if args.train_on_val else [])
    cfg['n_train'] = len(train_samples)
    print(f"Run {cfg['run_name']}: train {cfg['n_train']} / val {cfg['n_val']} / "
          f"test {cfg['n_test']} clips, {len(classes)} classes, device {device}"
          + (' (training on train+val)' if args.train_on_val else ''))
    dataloaders = train_val_dloaders(
        UCF50Dataset(train_samples, args.fr_per_vid, tr_tf, train=True),
        UCF50Dataset(splits['val'], args.fr_per_vid, eval_tf, train=False),
        args.batch_size, args.model_type, args.num_workers)
    if args.train_on_val:
        dataloaders['val'] = None

    model = build_model(cfg).to(device, memory_format=torch.channels_last
                                if args.model_type == 'lrcn' else torch.preserve_format)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay,
                                args.backbone_lr_mult)
    steps_per_epoch = len(dataloaders['train'])
    scheduler = build_scheduler(optimizer, steps_per_epoch * args.n_epochs,
                                int(steps_per_epoch * args.warmup_epochs))

    wandb_run = init_wandb(args, cfg, 'train')
    log_fn = (lambda metrics, step: wandb_run.log(metrics)) if wandb_run else None
    model, history = train(dataloaders, model, criterion, optimizer, scheduler, device, run_dir,
                           args.n_epochs, args.patience or None, log_fn,
                           checkpoint_extra={'config': {k: cfg[k] for k in MODEL_KEYS},
                                             'classes': classes})

    results = {'best_epoch': history['best_epoch'],
               'best_val_acc': max(history['val_acc'], default=None)}
    results.update(evaluate_views(model, splits['test'], cfg, args, device, classes, run_dir,
                                  wandb_run)[0])
    with open(os.path.join(run_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({'config': cfg, 'history': history, 'results': results}, f, indent=1)
    if wandb_run is not None:
        wandb_run.summary.update(results)
        wandb_run.finish()
    print(f"Best checkpoint: {os.path.join(run_dir, 'best.pt')}")
    return results


def load_checkpoint(path, device):
    """
    Rebuild a trained model from a checkpoint written by ``train.train``.

    Args:
        path (str): Checkpoint path.
        device (torch.device): Device to load onto.

    Returns:
        tuple: (model in eval mode, model config dict, class names)
    """
    ckpt = torch.load(path, map_location=device)
    cfg = dict(ckpt['config'], pretrained=False)  # weights come from the checkpoint
    model = build_model(cfg)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), cfg, ckpt['classes']


def run_eval(args, device):
    """
    Evaluate one checkpoint, or an ensemble of several, on the test (or val) split.

    The model architecture, clip length and input size come from each checkpoint, so only
    ``--ckpt`` (and optionally ``--frame_dir``) need to be given. With several checkpoints,
    each model is scored on its own and the models' (TTA-averaged) probabilities are then
    averaged with equal weights and scored as the ensemble (``ens_test`` / ``ens_test_tta``).
    ``--eval_split val`` scores the validation split instead, so ensembles and TTA settings
    can be chosen without looking at the test set.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Device to evaluate on.

    Returns:
        dict: Test metrics.
    """
    # pylint: disable=too-many-locals
    if not args.ckpt:
        raise ValueError('--ckpt is required in eval mode.')
    splits, classes, _ = load_splits(args.frame_dir)
    names = [os.path.basename(os.path.dirname(c)) for c in args.ckpt]
    is_ensemble = len(args.ckpt) > 1
    run_name = args.run_name or (('ensemble-' if is_ensemble else 'eval-') + '+'.join(names)
                                 + ('' if args.eval_split == 'test' else f'-{args.eval_split}'))
    wandb_run = init_wandb(args, {'run_name': run_name, 'ckpt': args.ckpt,
                                  'eval_split': args.eval_split,
                                  'test_clips': args.test_clips, 'test_hflip': args.test_hflip},
                           'eval')
    results, member_probs = {}, []
    for path, name in zip(args.ckpt, names):
        model, cfg, ckpt_classes = load_checkpoint(path, device)
        if ckpt_classes != classes:
            raise ValueError(f'Class list in splits.json does not match {path}.')
        res, probs = evaluate_views(model, splits[args.eval_split], cfg, args, device, classes,
                                    os.path.dirname(path), wandb_run,
                                    tag=f'{name}/' if is_ensemble else '', split=args.eval_split)
        results.update(res)
        member_probs.append(probs)
        del model
        torch.cuda.empty_cache()

    if is_ensemble:
        out_dir = os.path.join(args.out_dir, run_name)
        for kind in member_probs[0]:  # '<split>' and (with TTA) '<split>_tta'
            labels = member_probs[0][kind][1]
            if any(not torch.equal(m[kind][1], labels) for m in member_probs):
                raise ValueError('Ensemble members were evaluated on different samples.')
            probs = torch.stack([m[kind][0] for m in member_probs]).mean(dim=0)
            acc, loss = report_test(probs, labels, classes, out_dir, wandb_run, f'ens_{kind}',
                                    len(member_probs))
            results.update({f'ens_{kind}_acc': acc, f'ens_{kind}_loss': loss})
    if wandb_run is not None:
        wandb_run.summary.update(results)
        wandb_run.finish()
    return results


def main(args):
    """
    Dispatch to training or evaluation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    if args.mode == 'train':
        run_train(args, device)
    else:
        run_eval(args, device)


if __name__ == '__main__':
    main(args_parser())
