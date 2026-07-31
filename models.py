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


class SimpleCNN(nn.Module):
    """Small from-scratch CNN: 4 conv blocks (32->64->128->256 channels),
    global average pool, dropout, linear head. Trained from random init —
    no pretrained weights to download or fine-tune."""

    def __init__(self, num_classes=len(NIH_CLASSES), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
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


def build_model(name, num_classes=len(NIH_CLASSES), dropout=0.2, pretrained=True):
    if name == "chexnet":
        return CheXNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model: {name!r} (expected 'chexnet' or 'simple_cnn')")
