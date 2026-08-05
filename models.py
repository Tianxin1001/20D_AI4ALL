"""
Model architectures for the NIH Chest X-ray multi-label classifier.

CheXNet: DenseNet-121 pretrained on ImageNet, fine-tuned end-to-end
(Rajpurkar et al., 2017, https://arxiv.org/abs/1711.05225). Heavy to train
locally on limited RAM/no-CUDA hardware.

SimpleCNN: a small from-scratch convolutional net (~400K params, vs.
CheXNet's ~7M) intended as a lighter-weight baseline that trains at smaller
image sizes on modest local hardware, using the same multi-label task,
losses, and data pipeline as CheXNet.
"""
import torch.nn as nn
from torchvision import models

from preprocessing import NIH_CLASSES


class CheXNet(nn.Module):
    """DenseNet-121 backbone with a linear classifier head for multi-label output."""

    def __init__(self, num_classes=len(NIH_CLASSES), dropout=0.2, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


SIMPLE_CNN_CHANNELS = [32, 64, 128, 256]


class SimpleCNN(nn.Module):
    """Small from-scratch CNN: N conv blocks (32->64->128->256 channels),
    global average pool, dropout, linear head. Trained from random init —
    no pretrained weights to download or fine-tune.

    num_blocks controls depth (default 4 = the original architecture). Each
    block halves the spatial dimensions, so depth also sets total downsampling:
    4 blocks = 16x (224px input -> 14x14 final feature map), 3 = 8x (28x28),
    2 = 4x (56x56). Fewer blocks preserve finer spatial detail but reduce both
    channel capacity and receptive field."""

    def __init__(self, num_classes=len(NIH_CLASSES), dropout=0.3, num_blocks=4):
        super().__init__()
        if not 1 <= num_blocks <= len(SIMPLE_CNN_CHANNELS):
            raise ValueError(
                f"num_blocks must be between 1 and {len(SIMPLE_CNN_CHANNELS)}, got {num_blocks}"
            )
        channels = SIMPLE_CNN_CHANNELS[:num_blocks]
        blocks, in_channels = [], 3
        for out_channels in channels:
            blocks.append(self._conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.num_blocks = num_blocks
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(name, num_classes=len(NIH_CLASSES), dropout=0.2, pretrained=True,
                num_blocks=4):
    if name == "chexnet":
        return CheXNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout, num_blocks=num_blocks)
    raise ValueError(f"Unknown model: {name!r} (expected 'chexnet' or 'simple_cnn')")
