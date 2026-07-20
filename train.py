"""
Training script for the NIH Chest X-ray multi-label classifier.

Architecture: DenseNet-121 pretrained on ImageNet, fine-tuned end-to-end
with a replaced classifier head — this is the CheXNet architecture
(Rajpurkar et al., 2017, "CheXNet: Radiologist-Level Pneumonia Detection
on Chest X-Rays with Deep Learning", https://arxiv.org/abs/1711.05225).

Consumes preprocessing.py's get_dataloaders() (patient-level splits +
class weighting/oversampling) and NIH_CLASSES.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import models

from preprocessing import NIH_CLASSES, get_dataloaders, set_seed


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


def main():
    parser = argparse.ArgumentParser(description="Train CheXNet (DenseNet-121) on NIH Chest X-ray")
    parser.add_argument("--csv_path", required=True, help="Path to data_split.csv from the EDA notebook")
    parser.add_argument("--image_dir", required=True, help="Directory containing the chest X-ray images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Low LR for fine-tuning a pretrained backbone")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience, in epochs, on val AUC")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_oversampling", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_oversampling=not args.no_oversampling,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
    )

    model = CheXNet(dropout=args.dropout, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.run_name or 'best'}.pt")
    logger = ExperimentLogger(run_name=args.run_name)

    best_val_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mean_auc, per_class_auc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_mean_auc)
        logger.log(epoch, train_loss, val_loss, val_mean_auc, per_class_auc)

        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_mean_auc={val_mean_auc:.4f} | {elapsed:.1f}s")

        if val_mean_auc > best_val_auc:
            best_val_auc = val_mean_auc
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mean_auc": val_mean_auc,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"  -> New best val_mean_auc={val_mean_auc:.4f}, checkpoint saved to {best_ckpt_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val AUC improvement for {args.patience} epochs.")
                break

    print(f"\nBest val_mean_auc: {best_val_auc:.4f}")
    print(f"Run log: {logger.path}")

    # Final test-set evaluation using the best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mean_auc, test_per_class_auc = evaluate(model, test_loader, device, criterion)
    print(f"\nTest set — loss={test_loss:.4f} | mean_auc={test_mean_auc:.4f}")
    for cls, auc in zip(NIH_CLASSES, test_per_class_auc):
        print(f"  {cls:<20}: {auc:.4f}" if not np.isnan(auc) else f"  {cls:<20}: n/a (single class in test set)")


if __name__ == "__main__":
    main()
