"""
Module: utils.py

Helper functions shared by the UCF50 video-classification pipeline:

    - Frame extraction: uniformly sampling frames from a video file (``get_frames``) and
      writing them to disk as JPEGs (``store_frames``).
    - Transforms: per-model input statistics (``transform_stats``) and *clip-consistent*
      train / eval augmentation pipelines (``compose_data_transforms``).
    - DataLoaders: building train / val / test loaders with the right collate function.
    - Reproducibility: seeding every RNG used during training (``seed_everything``).
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from video_datasets import collate_fn_3dcnn, collate_fn_rnn

# ImageNet statistics. Every model in this project starts from ImageNet (object recognition)
# weights, including the inflated 3D CNN, so the inputs are always normalised with these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default square input size per model type.
DEFAULT_IMG_SIZE = {'lrcn': 224, '3dcnn': 160}


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Every frame is decoded first and the samples are then picked with ``np.linspace`` over the
    *actual* number of decoded frames. ``CAP_PROP_FRAME_COUNT`` is only an estimate for many
    AVI files, so indices computed from it can point past the end of the video.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample. If the video is shorter than this, some
            frames are repeated so that exactly ``n_frames`` frames are always returned.

    Returns:
        tuple: (frames, v_len)
            - frames (list[np.ndarray]): ``n_frames`` sampled frames as RGB ``uint8`` arrays.
            - v_len (int): Number of frames that were successfully decoded.

    Notes:
        If the video cannot be opened or contains no frames, ``([], 0)`` is returned.
    """
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print('Failed to open video:', vid)
        return [], 0
    all_frames = []
    while True:
        success, frame = v_cap.read()
        if not success:
            break
        all_frames.append(frame)
    v_cap.release()

    v_len = len(all_frames)
    if v_len == 0:
        print('No frames found in video:', vid)
        return [], 0
    frame_idx = np.linspace(0, v_len - 1, n_frames).round().astype(np.int64)
    frames = [cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2RGB) for idx in frame_idx]
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images in ``store_path``.

    Frames are named ``frame_0000.jpg``, ``frame_0001.jpg``, ... The zero padding means a
    plain lexical sort gives temporal order. Each frame is converted from RGB back to BGR
    because ``cv2.imwrite`` expects BGR.

    Args:
        frames (list[np.ndarray]): RGB frames to save.
        store_path (str): Directory where the frames are written (created if missing).

    Returns:
        None
    """
    os.makedirs(store_path, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path_to_frame = os.path.join(store_path, f'frame_{idx:04d}.jpg')
        cv2.imwrite(path_to_frame, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def transform_stats(model='lrcn', img_size=None):
    """
    Return the input size and normalisation statistics for a model type.

    Args:
        model (str): Model type, ``'lrcn'`` or ``'3dcnn'``.
        img_size (int, optional): Override for the square input size. Defaults to
            ``DEFAULT_IMG_SIZE[model]``.

    Returns:
        tuple: (h, w, mean, std)

    Raises:
        ValueError: If ``model`` is not a supported model type.
    """
    if model not in DEFAULT_IMG_SIZE:
        raise ValueError(f'model_type must be one of {sorted(DEFAULT_IMG_SIZE)}, got {model!r}')
    size = img_size or DEFAULT_IMG_SIZE[model]
    return size, size, IMAGENET_MEAN, IMAGENET_STD


def compose_data_transforms(height, width, mean, std, eval_hflip=False):
    """
    Compose train and eval transforms for clips of shape ``(T, C, H, W)`` (``uint8``).

    The transforms are ``torchvision.transforms.v2`` transforms applied to the *whole clip
    at once*, so the random parameters (crop box, flip, colour jitter) are sampled once per
    clip and are identical for every frame. The original code applied the transforms frame
    by frame, which flipped and shifted each frame independently and broke the motion
    signal.

    Train: random resized crop (scale 0.35-1), horizontal flip, light colour jitter, normalise.
    Eval:  resize the short side to the target size, center crop, normalise.

    Args:
        height (int): Output height.
        width (int): Output width.
        mean (list[float]): Per-channel normalisation mean.
        std (list[float]): Per-channel normalisation standard deviation.
        eval_hflip (bool): Mirror every eval clip (a deterministic view for test-time
            augmentation).

    Returns:
        tuple: (train_transforms, val_test_transforms)
    """
    train_transforms = v2.Compose([
        v2.RandomResizedCrop((height, width), scale=(0.35, 1.0), ratio=(0.75, 1.333),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    val_test_transforms = v2.Compose([
        v2.Resize(min(height, width), antialias=True),
        v2.CenterCrop((height, width)),
        *([v2.RandomHorizontalFlip(p=1.0)] if eval_hflip else []),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def _collate_for(model):
    """Return the collate function that matches ``model`` ('lrcn' or '3dcnn')."""
    return collate_fn_rnn if model == 'lrcn' else collate_fn_3dcnn


def _loader(dataset, batch_size, model, shuffle, num_workers, persistent=True):
    """
    Build one DataLoader with the settings shared by every split.

    ``persistent`` keeps worker processes alive between epochs. That is useful for train/val
    but wasteful for test loaders, which are iterated once (and several may exist at once
    for multi-view evaluation).
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate_for(model), num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=persistent and num_workers > 0)


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn', num_workers=6):
    """
    Create the training and validation DataLoaders.

    The training loader shuffles and drops the last incomplete batch, which keeps BatchNorm
    batch statistics stable. The validation loader keeps every sample and uses twice the
    batch size because no gradients are stored.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int): Training mini-batch size.
        model (str): ``'lrcn'`` (B, T, C, H, W batches) or ``'3dcnn'`` (B, C, T, H, W batches).
        num_workers (int): DataLoader worker processes.

    Returns:
        dict: ``{'train': DataLoader, 'val': DataLoader}``
    """
    return {'train': _loader(train_dataset, batch_size, model, True, num_workers),
            'val': _loader(val_dataset, 2 * batch_size, model, False, num_workers)}


def test_dloaders(test_dataset, batch_size, model='lrcn', num_workers=6):
    """
    Create the test DataLoader.

    Args:
        test_dataset (Dataset): Test dataset.
        batch_size (int): Training mini-batch size (the test loader uses twice this).
        model (str): ``'lrcn'`` or ``'3dcnn'``.
        num_workers (int): DataLoader worker processes.

    Returns:
        dict: ``{'test': DataLoader}``
    """
    return {'test': _loader(test_dataset, 2 * batch_size, model, False, num_workers,
                            persistent=False)}


def seed_everything(seed):
    """
    Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs for reproducible runs.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
