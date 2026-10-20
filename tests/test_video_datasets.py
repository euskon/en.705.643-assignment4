"""Tests for UCF50 dataset handling: name parsing, leak-free splitting, dedupe and loading."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from video_datasets import (UCF50Dataset, collate_fn_3dcnn, collate_fn_rnn,
                            find_duplicate_videos, group_split, parse_ucf50_name,
                            sample_frame_indices)


def test_parse_ucf50_name():
    """Group and clip ids are read from both file names and paths."""
    assert parse_ucf50_name('v_Biking_g12_c01.avi') == (12, 1)
    assert parse_ucf50_name('/x/Billiards/v_Billards_g02_c07') == (2, 7)
    with pytest.raises(ValueError):
        parse_ucf50_name('not_a_ucf_video.avi')


def test_group_split_has_no_group_overlap():
    """Every group lands in exactly one split, and extra groups (>25) always go to train."""
    groups = [g for g in range(1, 31) for _ in range(5)]
    names, val_groups, test_groups = group_split(groups, n_val_groups=3, n_test_groups=4, seed=0)
    by_split = {s: {g for g, n in zip(groups, names) if n == s} for s in ('train', 'val', 'test')}
    assert by_split['test'] == set(test_groups) and len(test_groups) == 4
    assert by_split['val'] == set(val_groups) and len(val_groups) == 3
    assert not by_split['train'] & by_split['val']
    assert not by_split['train'] & by_split['test']
    assert {26, 27, 28, 29, 30} <= by_split['train']
    # Deterministic for a fixed seed
    assert group_split(groups, 3, 4, seed=0)[2] == test_groups


def test_find_duplicate_videos_keeps_first_sorted_copy():
    """Identical content is reported once, mapped to the copy that is kept."""
    contents = {'a/x.avi': 'h1', 'b/y.avi': 'h2', 'c/z.avi': 'h1', 'a/w.avi': 'h1'}
    dups = find_duplicate_videos(contents, hash_fn=contents.get)
    assert dups == {'a/x.avi': 'a/w.avi', 'c/z.avi': 'a/w.avi'}


@pytest.mark.parametrize('n_available,n_frames', [(32, 16), (32, 8), (5, 16), (1, 4)])
def test_sample_frame_indices(n_available, n_frames):
    """Indices are in range, sorted, the right length, and deterministic in eval mode."""
    for train in (True, False):
        idx = sample_frame_indices(n_available, n_frames, train)
        assert len(idx) == n_frames
        assert idx.min() >= 0 and idx.max() < n_available
        assert np.all(np.diff(idx) >= 0)
    np.testing.assert_array_equal(sample_frame_indices(n_available, n_frames, False),
                                  sample_frame_indices(n_available, n_frames, False))


def test_eval_sampling_spans_whole_video():
    """Eval sampling covers the video start to end instead of only the first frames."""
    idx = sample_frame_indices(32, 8, train=False)
    assert idx[0] <= 2 and idx[-1] >= 29


@pytest.fixture(name='fake_video_dir')
def fixture_fake_video_dir(tmp_path):
    """A 12-frame video whose frame i is a solid image with intensity 10 * i."""
    video_dir = tmp_path / 'Biking' / 'v_Biking_g01_c01'
    os.makedirs(video_dir)
    for i in range(12):
        Image.new('RGB', (8, 6), (10 * i,) * 3).save(video_dir / f'frame_{i:04d}.jpg',
                                                     quality=100)
    return str(video_dir)


def test_dataset_loads_frames_in_temporal_order(fake_video_dir):
    """Frames come back in temporal order (frame_0010 after frame_0002), as uint8 T x C x H x W."""
    ds = UCF50Dataset([(fake_video_dir, 7)], fr_per_vid=12, transforms=None, train=False)
    clip, label = ds[0]
    assert label == 7
    assert clip.shape == (12, 3, 6, 8) and clip.dtype == torch.uint8
    means = clip.float().mean(dim=(1, 2, 3))
    assert torch.all(means[1:] > means[:-1])


def test_collate_layouts():
    """LRCN batches are (B, T, C, H, W); 3D-CNN batches are (B, C, T, H, W)."""
    batch = [(torch.zeros(4, 3, 8, 8), 1), (torch.ones(4, 3, 8, 8), 2)]
    x, y = collate_fn_rnn(batch)
    assert x.shape == (2, 4, 3, 8, 8) and y.tolist() == [1, 2]
    x3d, _ = collate_fn_3dcnn(batch)
    assert x3d.shape == (2, 3, 4, 8, 8)
    assert torch.equal(x3d[1, :, 2], x[1, 2])
