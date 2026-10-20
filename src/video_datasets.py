"""
Module: video_datasets.py

Dataset handling for UCF50.

UCF50 file names look like ``v_<Action>_g<GG>_c<CC>.avi``. ``GG`` is the *group*: all clips
in a group were cut from the same long source video (same actor, background, camera and
lighting). Clips of one group are therefore near-duplicates of each other, so a random
per-clip split leaks test information into training. The dataset authors require
splitting by group, and this module does that:

    - ``parse_ucf50_name``: pull the group and clip ids out of a UCF50 file name.
    - ``find_duplicate_videos``: find byte-identical videos (UCF50 ships 14 such pairs,
      some of them in *different* groups, so even a group split would leak).
    - ``group_split``: assign whole groups to train / val / test.
    - ``load_dataset`` / ``save_splits`` / ``load_splits``: create and read the
      ``splits.json`` index written by ``prepare_data.py``.
    - ``UCF50Dataset``: PyTorch dataset that samples a fixed number of frames per clip.
    - ``collate_fn_rnn`` / ``collate_fn_3dcnn``: batch layouts for the two model types.
"""

import glob
import hashlib
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, read_file

_UCF50_NAME_RE = re.compile(r'_g(\d+)_c(\d+)')

SPLITS_FILENAME = 'splits.json'


def parse_ucf50_name(name):
    """
    Parse the group and clip ids from a UCF50 video (or frame directory) name.

    Args:
        name (str): A name or path such as ``'.../v_Biking_g12_c01.avi'``.

    Returns:
        tuple: (group, clip) as ints.

    Raises:
        ValueError: If the name does not follow the UCF50 ``_gXX_cYY`` convention.
    """
    match = _UCF50_NAME_RE.search(os.path.basename(name))
    if match is None:
        raise ValueError(f'Not a UCF50 video name: {name!r}')
    return int(match.group(1)), int(match.group(2))


def file_md5(path, chunk_size=1 << 20):
    """
    Compute the MD5 hex digest of a file.

    Args:
        path (str): Path to the file.
        chunk_size (int): Read size in bytes.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_videos(video_paths, hash_fn=file_md5):
    """
    Find byte-identical videos.

    For each set of identical files, the first path in sorted order is kept and the others
    are reported as duplicates to drop.

    Args:
        video_paths (Iterable[str]): Video file paths.
        hash_fn (callable): Function mapping a path to a content hash.

    Returns:
        dict: ``{duplicate_path: kept_path}`` for every path that should be dropped.
    """
    first_seen = {}
    duplicates = {}
    for path in sorted(video_paths):
        digest = hash_fn(path)
        if digest in first_seen:
            duplicates[path] = first_seen[digest]
        else:
            first_seen[digest] = path
    return duplicates


def group_split(groups, n_val_groups=3, n_test_groups=4, n_core_groups=25, seed=0):
    """
    Choose which UCF50 groups go to the validation and test sets.

    The core groups ``1..n_core_groups`` are present in every class. They are shuffled
    with ``seed``, and the first ``n_test_groups`` go to test and the next ``n_val_groups``
    to validation. The same group ids are used for every class, so the class balance stays
    close to the full dataset. A few classes have extra groups (26-30); those always go to
    training.

    Args:
        groups (Iterable[int]): Group id of every sample.
        n_val_groups (int): Number of core groups held out for validation.
        n_test_groups (int): Number of core groups held out for testing.
        n_core_groups (int): Number of groups shared by all classes (25 for UCF50).
        seed (int): Seed for the group permutation.

    Returns:
        tuple: (split_names, val_groups, test_groups)
            - split_names (list[str]): ``'train'``, ``'val'`` or ``'test'`` per sample.
            - val_groups (list[int]): Sorted validation group ids.
            - test_groups (list[int]): Sorted test group ids.
    """
    if n_val_groups + n_test_groups >= n_core_groups:
        raise ValueError('Not enough groups left for training.')
    perm = np.random.default_rng(seed).permutation(np.arange(1, n_core_groups + 1))
    test_groups = sorted(int(g) for g in perm[:n_test_groups])
    val_groups = sorted(int(g) for g in perm[n_test_groups:n_test_groups + n_val_groups])
    split_names = []
    for group in groups:
        if group in test_groups:
            split_names.append('test')
        elif group in val_groups:
            split_names.append('val')
        else:
            split_names.append('train')
    return split_names, val_groups, test_groups


def save_splits(path, classes, splits, meta=None):
    """
    Write the dataset index to JSON.

    Args:
        path (str): Output ``splits.json`` path.
        classes (list[str]): Class names, indexed by label.
        splits (dict): ``{'train'|'val'|'test': [(rel_video_dir, label, group), ...]}``.
        meta (dict, optional): Extra provenance stored with the splits (seed, held-out
            groups, removed duplicates, ...).
    """
    payload = {'classes': list(classes), 'meta': meta or {}}
    for name, samples in splits.items():
        payload[name] = [{'video': v, 'label': int(l), 'group': int(g)} for v, l, g in samples]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)


def load_splits(frame_dir):
    """
    Load ``<frame_dir>/splits.json`` as written by ``prepare_data.py``.

    Args:
        frame_dir (str): Root directory of the extracted frames.

    Returns:
        tuple: (splits, classes, meta)
            - splits (dict): ``{'train'|'val'|'test': [(abs_video_dir, label), ...]}``.
            - classes (list[str]): Class names, indexed by label.
            - meta (dict): Provenance information saved with the splits.
    """
    path = os.path.join(frame_dir, SPLITS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found - run `python src/prepare_data.py` first.')
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    splits = {name: [(os.path.join(frame_dir, s['video']), s['label']) for s in payload[name]]
              for name in ('train', 'val', 'test')}
    return splits, payload['classes'], payload.get('meta', {})


def load_dataset(frame_dir):
    """
    List every extracted video in ``frame_dir`` (one sub-directory per class).

    Args:
        frame_dir (str): Directory laid out as ``<frame_dir>/<Class>/<video_stem>/frame_*.jpg``.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): ``{video_dir: label}``.
            - label_dict (dict): ``{class_name: label}``, with labels assigned in sorted
              class-name order.
    """
    classes = sorted(d for d in os.listdir(frame_dir)
                     if os.path.isdir(os.path.join(frame_dir, d)))
    label_dict = {cls: idx for idx, cls in enumerate(classes)}
    vid_dataset = {}
    for cls in classes:
        cls_dir = os.path.join(frame_dir, cls)
        for vid in sorted(os.listdir(cls_dir)):
            vid_dataset[os.path.join(cls_dir, vid)] = label_dict[cls]
    return vid_dataset, label_dict


def sample_frame_indices(n_available, n_frames, train, offset=0.5):
    """
    Choose which stored frames to load for one clip (TSN-style segment sampling).

    The ``n_available`` frames are divided into ``n_frames`` equal segments. Training picks
    a random frame inside each segment (temporal jitter). Evaluation picks the frame at a
    fixed relative ``offset`` in each segment (0.5 = center), so every frame in a clip comes
    from a different part of the video. Evaluating several offsets and averaging the
    predictions gives multi-view test-time augmentation.

    Args:
        n_available (int): Number of frames stored for the video (>= 1).
        n_frames (int): Number of frames to return.
        train (bool): Random offsets when True, deterministic ``offset`` when False.
        offset (float): Relative position inside each segment in eval mode, in ``[0, 1)``.

    Returns:
        np.ndarray: ``n_frames`` sorted indices in ``[0, n_available)``. If the video has fewer
        frames than requested, some indices repeat.
    """
    edges = np.linspace(0, n_available, n_frames + 1)
    if train:
        idx = edges[:-1] + np.random.rand(n_frames) * (edges[1:] - edges[:-1])
    else:
        idx = edges[:-1] + offset * (edges[1:] - edges[:-1])
    return np.clip(np.floor(idx).astype(np.int64), 0, n_available - 1)


class UCF50Dataset(Dataset):
    """
    PyTorch dataset of UCF50 clips stored as directories of JPEG frames.

    Each item is a clip of exactly ``fr_per_vid`` frames picked with
    ``sample_frame_indices``, so every clip is the same length and no padding is needed.
    The transform runs on the whole ``(T, C, H, W)`` clip, so random augmentations are the
    same for every frame.

    Args:
        samples (list[tuple[str, int]]): ``(video_dir, label)`` pairs.
        fr_per_vid (int): Frames per clip.
        transforms (callable, optional): Transform applied to the ``uint8`` clip tensor.
        train (bool): Random temporal jitter when True, deterministic sampling when False.
        offset (float): Relative frame position inside each temporal segment in eval mode.
    """

    def __init__(self, samples, fr_per_vid, transforms=None, train=False, offset=0.5):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.train = train
        self.offset = offset
        self._frame_cache = {}

    def __len__(self):
        """Return the number of clips."""
        return len(self.samples)

    def _frame_paths(self, video_dir):
        """Return the sorted frame paths of a video, cached per worker."""
        if video_dir not in self._frame_cache:
            paths = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
            if not paths:
                raise FileNotFoundError(f'No frames found in {video_dir}')
            self._frame_cache[video_dir] = paths
        return self._frame_cache[video_dir]

    def __getitem__(self, idx):
        """
        Load, sample and transform one clip.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (clip, label). ``clip`` is a ``(T, C, H, W)`` tensor (float after the
            standard transforms) and ``label`` is an int.
        """
        video_dir, label = self.samples[idx]
        paths = self._frame_paths(video_dir)
        indices = sample_frame_indices(len(paths), self.fpv, self.train, self.offset)
        clip = torch.stack([decode_jpeg(read_file(paths[i])) for i in indices])
        clip = tv_tensors.Video(clip)
        if self.transforms is not None:
            clip = self.transforms(clip)
        return clip.as_subclass(torch.Tensor), label


# Backwards-compatible alias for the original class name.
VideoDataset = UCF50Dataset


def collate_fn_rnn(batch):
    """
    Collate clips for frame-wise models such as LRCN.

    Args:
        batch (list[tuple[Tensor, int]]): ``(clip (T, C, H, W), label)`` samples.

    Returns:
        tuple: (clips (B, T, C, H, W), labels (B,))
    """
    clips, labels = zip(*batch)
    return torch.stack(clips), torch.tensor(labels, dtype=torch.long)


def collate_fn_3dcnn(batch):
    """
    Collate clips for 3D CNNs, which expect channels before time.

    Args:
        batch (list[tuple[Tensor, int]]): ``(clip (T, C, H, W), label)`` samples.

    Returns:
        tuple: (clips (B, C, T, H, W), labels (B,))
    """
    clips, labels = collate_fn_rnn(batch)
    return clips.transpose(1, 2).contiguous(), labels
