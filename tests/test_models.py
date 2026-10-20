"""Tests for the LRCN and inflated R3D-18 models."""

import pytest
import torch
from torchvision import models
from torchvision.models import video as video_models

from models import LRCN, R3D18, build_model, inflate_resnet18_to_r3d18


@pytest.fixture(name='lrcn')
def fixture_lrcn():
    """A small, randomly initialised LRCN in eval mode."""
    torch.manual_seed(0)
    return LRCN(hidden_size=16, n_layers=1, dropout_rate=0.0, n_classes=5, pretrained=False,
                cnn_model='resnet18', bidirectional=True, temporal_pool='mean').eval()


def test_lrcn_output_shape(lrcn):
    """LRCN maps (B, T, C, H, W) to (B, n_classes)."""
    with torch.no_grad():
        assert lrcn(torch.randn(3, 4, 3, 32, 32)).shape == (3, 5)


@pytest.mark.parametrize('temporal_pool', ['mean', 'last'])
def test_lrcn_samples_are_independent(temporal_pool):
    """
    Regression test for the main bug: a clip's prediction must not depend on the other
    clips in its batch. The original LSTM read the batch as the time axis.
    """
    torch.manual_seed(0)
    model = LRCN(16, 1, 0.0, 5, pretrained=False, cnn_model='resnet18',
                 temporal_pool=temporal_pool).eval()
    clip_a, clip_b, clip_c = (torch.randn(1, 4, 3, 32, 32) for _ in range(3))
    with torch.no_grad():
        alone = model(clip_a)[0]
        after_b = model(torch.cat([clip_b, clip_a]))[1]
        after_c = model(torch.cat([clip_c, clip_a]))[1]
    torch.testing.assert_close(alone, after_b, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after_b, after_c, atol=1e-5, rtol=1e-4)


def test_lrcn_uses_temporal_order(lrcn):
    """Reversing the frames changes the output, so the LSTM really reads them in sequence."""
    clip = torch.randn(1, 6, 3, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(lrcn(clip), lrcn(clip.flip(1)))


def test_lrcn_rejects_unknown_backbone():
    """An unsupported backbone raises instead of silently using a different one."""
    with pytest.raises(ValueError):
        LRCN(16, 1, 0.0, 5, pretrained=False, cnn_model='vgg16')


def test_inflation_copies_and_averages_kernels():
    """Each inflated kernel summed over time equals the source 2D kernel."""
    torch.manual_seed(0)
    resnet = models.resnet18(weights=None)
    r3d = video_models.r3d_18(weights=None)
    n_copied = inflate_resnet18_to_r3d18(r3d, resnet)
    assert n_copied == len(resnet.state_dict()) - 2  # all but fc.weight / fc.bias
    torch.testing.assert_close(r3d.stem[0].weight.sum(dim=2), resnet.conv1.weight)
    torch.testing.assert_close(r3d.layer3[0].conv2[0].weight.sum(dim=2),
                               resnet.layer3[0].conv2.weight)
    torch.testing.assert_close(r3d.layer2[0].conv1[1].running_var, resnet.layer2[0].bn1.running_var)


def test_build_model_r3d_output_shape():
    """The 3D CNN maps (B, C, T, H, W) to (B, n_classes)."""
    model = build_model({'model_type': '3dcnn', 'n_classes': 7, 'dropout': 0.0,
                         'pretrained': False}).eval()
    assert isinstance(model, R3D18)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 4, 32, 32)).shape == (2, 7)
