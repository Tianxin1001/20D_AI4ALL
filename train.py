"""
Training script for the NIH Chest X-ray multi-label classifier.

Architecture: DenseNet-121 pretrained on ImageNet, fine-tuned end-to-end
with a replaced classifier head — this is the CheXNet architecture
(Rajpurkar et al., 2017, "CheXNet: Radiologist-Level Pneumonia Detection
on Chest X-Rays with Deep Learning", https://arxiv.org/abs/1711.05225).

Consumes preprocessing.py's get_dataloaders() (patient-level splits +
class weighting/oversampling) and NIH_CLASSES.
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
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


def compute_per_class_auc(y_true, y_pred):
    """Per-class ROC-AUC. Returns NaN for classes with only one label present
    in the batch of ground truth (undefined AUC), which are excluded from the mean."""
    aucs = np.full(y_true.shape[1], np.nan)
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if len(np.unique(col)) > 1:
            aucs[i] = roc_auc_score(col, y_pred[:, i])
    return aucs


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    # Binarize smoothed labels (>=0.5) so AUC is computed against true positives
    y_true_binary = (y_true >= 0.5).astype(int)
    y_pred = np.concatenate(all_probs, axis=0)

    per_class_auc = compute_per_class_auc(y_true_binary, y_pred)
    mean_auc = float(np.nanmean(per_class_auc))

    return total_loss / max(n_samples, 1), mean_auc, per_class_auc


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return total_loss / max(n_samples, 1)


class ExperimentLogger:
    """Append-only CSV run log — one row per epoch, one file per run."""

    def __init__(self, log_dir="experiments", run_name=None):
        os.makedirs(log_dir, exist_ok=True)
        run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{run_name}.csv")
        self.fields = ["epoch", "train_loss", "val_loss", "val_mean_auc"] + \
            [f"val_auc_{c}" for c in NIH_CLASSES]
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.fields)

    def log(self, epoch, train_loss, val_loss, val_mean_auc, per_class_auc):
        row = [epoch, train_loss, val_loss, val_mean_auc] + list(per_class_auc)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
