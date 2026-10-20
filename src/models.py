"""
Module: models.py

Video classification models for UCF50. Both models start from ImageNet weights (pure object
recognition pre-training); no video pre-training such as Kinetics is used.

Classes / functions:
    LRCN: Long-term Recurrent Convolutional Network. A 2D CNN encodes each frame, an LSTM
          (optionally bidirectional) models the frame sequence, and the per-step LSTM outputs
          are pooled over time and classified.
    R3D18: 3D ResNet-18 whose 3D kernels are *inflated* from ImageNet ResNet-18 2D kernels
           (the I3D "bootstrapping" trick).
    inflate_resnet18_to_r3d18: Copy and inflate ImageNet ResNet-18 weights into an R3D-18.
    build_model: Factory that builds a model from the run configuration.
"""

import torch
from torch import nn
from torchvision import models
from torchvision.models import video as video_models

# 2D backbones LRCN supports: name -> (builder, best ImageNet-1k (object recognition) weights,
# attribute path of the final ImageNet classification layer that gets replaced by Identity).
BACKBONES = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 'fc'),
    'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 'fc'),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 'fc'),
    'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 'fc'),
    'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 'fc'),
    'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                      'classifier.2'),
    'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                       'classifier.2'),
    'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
                          'classifier.1'),
}


def build_backbone(name, pretrained=True):
    """
    Build a 2D ImageNet CNN with its classification layer replaced by ``nn.Identity``.

    Args:
        name (str): One of ``BACKBONES``.
        pretrained (bool): Load the ImageNet-1k weights.

    Returns:
        tuple: (backbone, num_features). The backbone maps (N, 3, H, W) images to
        (N, num_features) pooled features.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    if name not in BACKBONES:
        raise ValueError(f'Unsupported CNN backbone {name!r}; choose one of {sorted(BACKBONES)}.')
    builder, weights, head_path = BACKBONES[name]
    net = builder(weights=weights if pretrained else None)
    num_features = net.get_submodule(head_path).in_features
    net.set_submodule(head_path, nn.Identity())
    return net, num_features


class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    Pipeline: frames (B, T, C, H, W) -> CNN features (B, T, F) -> LSTM (B, T, H') ->
    temporal pooling (B, H') -> dropout -> linear classifier (B, n_classes).

    Args:
        hidden_size (int): LSTM hidden size per direction.
        n_layers (int): Number of stacked LSTM layers.
        dropout_rate (float): Dropout before the classifier (and between stacked LSTM layers).
        n_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet weights into the CNN backbone.
        cnn_model (str): One of ``BACKBONES`` (ResNet, ConvNeXt or EfficientNet-V2).
        bidirectional (bool): Run the LSTM in both temporal directions.
        temporal_pool (str): ``'mean'`` averages the LSTM outputs over all time steps;
            ``'last'`` uses only the final step (the original LRCN head).

    Raises:
        ValueError: If ``cnn_model`` or ``temporal_pool`` is not supported.
    """

    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True,
                 cnn_model='resnet34', bidirectional=False, temporal_pool='last'):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        if temporal_pool not in ('mean', 'last'):
            raise ValueError("temporal_pool must be 'mean' or 'last'.")
        # Pooled features instead of ImageNet logits.
        self.base_model, num_features = build_backbone(cnn_model, pretrained)

        self.temporal_pool = temporal_pool
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout_rate if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), n_classes)

    def forward(self, x):
        """
        Classify a batch of clips.

        All ``B * T`` frames go through the CNN in one call. The LSTM is ``batch_first``, so
        each clip is a separate sequence. (The original code fed the LSTM ``(B, 1, F)``
        tensors without ``batch_first``, so the LSTM read the *batch* as the time axis and
        carried hidden state from one video to the next.)

        Args:
            x (Tensor): Clips of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        bs, ts = x.shape[:2]
        feats = self.base_model(x.flatten(0, 1)).view(bs, ts, -1)
        out, _ = self.rnn(feats)
        out = out.mean(dim=1) if self.temporal_pool == 'mean' else out[:, -1]
        return self.fc(self.dropout(out))

    def backbone_parameters(self):
        """Return the parameters of the ImageNet-pretrained CNN backbone."""
        return list(self.base_model.parameters())

    def head_parameters(self):
        """Return the parameters trained from scratch (LSTM and classifier)."""
        return list(self.rnn.parameters()) + list(self.fc.parameters())


def inflate_resnet18_to_r3d18(r3d, resnet2d):
    """
    Initialise an R3D-18 from ImageNet ResNet-18 weights by kernel inflation (I3D, Carreira &
    Zisserman 2017).

    Each 2D kernel ``W`` of shape (O, I, k, k) is repeated ``t`` times along a new temporal
    axis and divided by ``t``. On a video of identical frames the inflated network then gives
    the same activations as the 2D network on one frame, so the ImageNet features carry over
    exactly. BatchNorm parameters and running statistics are copied unchanged. The ImageNet
    classifier (``fc``) is skipped.

    Args:
        r3d (nn.Module): ``torchvision.models.video.r3d_18`` instance to initialise in place.
        resnet2d (nn.Module): ``torchvision.models.resnet18`` with ImageNet weights.

    Returns:
        int: Number of tensors copied into ``r3d``.
    """
    def map_key(key):
        """Translate a ResNet-18 parameter name to the matching R3D-18 name."""
        if key.startswith('conv1.'):
            return 'stem.0.' + key[len('conv1.'):]
        if key.startswith('bn1.'):
            return 'stem.1.' + key[len('bn1.'):]
        if key.startswith('layer'):
            for idx, name in ((1, 'conv1'), (2, 'conv2')):
                key = key.replace(f'.{name}.weight', f'.{name}.0.weight')
                key = key.replace(f'.bn{idx}.', f'.{name}.1.')
        return key

    target = r3d.state_dict()
    copied = 0
    with torch.no_grad():
        for key, value in resnet2d.state_dict().items():
            if key.startswith('fc.'):
                continue
            new_key = map_key(key)
            dst = target[new_key]
            if value.dim() == 4:  # conv kernel -> inflate over time
                t = dst.shape[2]
                value = value.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
            if dst.shape != value.shape:
                raise RuntimeError(f'Shape mismatch for {key} -> {new_key}: '
                                   f'{tuple(value.shape)} vs {tuple(dst.shape)}')
            dst.copy_(value)
            copied += 1
    return copied


class R3D18(nn.Module):
    """
    3D ResNet-18 video classifier initialised from inflated ImageNet ResNet-18 weights.

    Args:
        n_classes (int): Number of output classes.
        dropout_rate (float): Dropout before the classifier.
        pretrained (bool): Inflate ImageNet ResNet-18 weights into the 3D kernels.
    """

    def __init__(self, n_classes, dropout_rate=0.5, pretrained=True):
        super().__init__()
        net = video_models.r3d_18(weights=None)
        if pretrained:
            inflate_resnet18_to_r3d18(
                net, models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))
        net.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(net.fc.in_features, n_classes))
        self.net = net

    def forward(self, x):
        """
        Classify a batch of clips.

        Args:
            x (Tensor): Clips of shape (batch_size, channels, time_steps, height, width).

        Returns:
            Tensor: Logits of shape (batch_size, n_classes).
        """
        return self.net(x)

    def backbone_parameters(self):
        """Return the parameters initialised from inflated ImageNet weights."""
        return [p for name, p in self.net.named_parameters() if not name.startswith('fc.')]

    def head_parameters(self):
        """Return the classifier parameters trained from scratch."""
        return list(self.net.fc.parameters())


def build_model(cfg):
    """
    Build the model described by a run configuration.

    Args:
        cfg (dict): Must contain ``model_type`` and ``n_classes``. LRCN also reads
            ``rnn_hidden_size``, ``rnn_n_layers``, ``dropout``, ``pretrained``,
            ``cnn_backbone``, ``bidirectional`` and ``temporal_pool``.

    Returns:
        nn.Module: The model.

    Raises:
        ValueError: For an unknown ``model_type``.
    """
    if cfg['model_type'] == 'lrcn':
        return LRCN(hidden_size=cfg['rnn_hidden_size'], n_layers=cfg['rnn_n_layers'],
                    dropout_rate=cfg['dropout'], n_classes=cfg['n_classes'],
                    pretrained=cfg['pretrained'], cnn_model=cfg['cnn_backbone'],
                    bidirectional=cfg.get('bidirectional', False),
                    temporal_pool=cfg.get('temporal_pool', 'last'))
    if cfg['model_type'] == '3dcnn':
        return R3D18(n_classes=cfg['n_classes'], dropout_rate=cfg['dropout'],
                     pretrained=cfg['pretrained'])
    raise ValueError(f"Unknown model_type {cfg['model_type']!r} (use 'lrcn' or '3dcnn').")
