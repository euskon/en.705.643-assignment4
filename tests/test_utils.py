"""Tests for frame extraction and clip-level transforms."""

import os

import cv2
import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from utils import compose_data_transforms, get_frames, store_frames, transform_stats


@pytest.fixture(name='synthetic_video')
def fixture_synthetic_video(tmp_path):
    """Write a 40-frame AVI where frame i has brightness ~ 5 * i."""
    path = str(tmp_path / 'v_Test_g01_c01.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 24))
    for i in range(40):
        writer.write(np.full((24, 32, 3), 5 * i, dtype=np.uint8))
    writer.release()
    return path


def test_get_frames_returns_exactly_n_uniform_frames(synthetic_video):
    """Exactly n_frames come back (the original returned n + 1), spanning first to last."""
    frames, v_len = get_frames(synthetic_video, n_frames=8)
    assert v_len == 40
    assert len(frames) == 8
    brightness = [f.mean() for f in frames]
    assert brightness[0] < 5 and brightness[-1] > 185
    assert all(b2 > b1 for b1, b2 in zip(brightness, brightness[1:]))


def test_get_frames_unreadable_video(tmp_path):
    """A missing or corrupt video returns ([], 0) instead of raising."""
    assert get_frames(str(tmp_path / 'missing.avi'), 4) == ([], 0)


def test_store_frames_zero_padded_names(tmp_path):
    """Stored names are zero-padded so a lexical sort is also temporal order."""
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 12
    store_frames(frames, str(tmp_path / 'vid'))
    names = sorted(os.listdir(tmp_path / 'vid'))
    assert names[0] == 'frame_0000.jpg' and names[-1] == 'frame_0011.jpg'
    assert names == [f'frame_{i:04d}.jpg' for i in range(12)]


def test_train_transform_is_consistent_across_frames():
    """Random crop / flip / jitter use the same parameters for every frame of a clip."""
    torch.manual_seed(0)
    frame = torch.randint(0, 256, (1, 3, 60, 80), dtype=torch.uint8)
    clip = tv_tensors.Video(frame.repeat(6, 1, 1, 1))
    train_tf, _ = compose_data_transforms(*transform_stats('lrcn', 32))
    for _ in range(5):
        out = train_tf(clip)
        assert out.shape == (6, 3, 32, 32) and out.dtype == torch.float32
        assert all(torch.equal(out[0], out[i]) for i in range(1, 6))


def test_eval_transform_is_deterministic():
    """The eval pipeline has no randomness."""
    clip = tv_tensors.Video(torch.randint(0, 256, (4, 3, 60, 80), dtype=torch.uint8))
    _, eval_tf = compose_data_transforms(*transform_stats('3dcnn', 32))
    assert torch.equal(eval_tf(clip), eval_tf(clip))


def test_transform_stats_rejects_unknown_model():
    """Unsupported model types raise a clear error."""
    with pytest.raises(ValueError):
        transform_stats('transformer')
