"""
Training script for the NIH Chest X-ray multi-label classifier.

Supports two architectures via --model:
  - chexnet: DenseNet-121 pretrained on ImageNet, fine-tuned end-to-end
    (Rajpurkar et al., 2017, "CheXNet: Radiologist-Level Pneumonia Detection
    on Chest X-Rays with Deep Learning", https://arxiv.org/abs/1711.05225).
  - simple_cnn: small from-scratch conv net, lighter to train locally.

Both share the same preprocessing.py get_dataloaders() pipeline (patient-level
splits + class weighting/oversampling) and NIH_CLASSES, so runs are directly
comparable.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from models import build_model
from preprocessing import NIH_CLASSES, get_dataloaders, set_seed


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
def evaluate(model, loader, device, criterion, return_predictions=False):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_labels, all_probs, all_idx = [], [], []

    for batch in loader:
        # Datasets built with return_meta=True yield (image, label, row_index)
        if len(batch) == 3:
            images, labels, idx = batch
            all_idx.append(idx.numpy())
        else:
            images, labels = batch
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
    avg_loss = total_loss / max(n_samples, 1)

    if return_predictions:
        row_idx = np.concatenate(all_idx, axis=0) if all_idx else np.arange(len(y_pred))
        return avg_loss, mean_auc, per_class_auc, y_true_binary, y_pred, row_idx
    return avg_loss, mean_auc, per_class_auc


def save_test_predictions(dataset, y_true, y_pred, row_idx, path):
    """Writes one row per test image: metadata + true label + predicted probability
    for all 14 classes. This is the input the fairness audit needs — without it,
    per-subgroup recall would require re-running inference every time.
    """
    meta_cols = [c for c in ['Image Index', 'Patient ID', 'Patient Age',
                             'Patient Gender', 'View Position'] if c in dataset.df.columns]
    out = dataset.df.iloc[row_idx][meta_cols].reset_index(drop=True)
    for i, cls in enumerate(NIH_CLASSES):
        out[f'true_{cls}'] = y_true[:, i]
        out[f'prob_{cls}'] = y_pred[:, i]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Per-image test predictions ({len(out):,} rows) saved to {path}")


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

    def log_test(self, test_loss, test_mean_auc, per_class_auc):
        """Appends the final test-set row so results survive the process exiting."""
        row = ["test", "", test_loss, test_mean_auc] + list(per_class_auc)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train CheXNet (DenseNet-121) on NIH Chest X-ray")
    parser.add_argument("--csv_path", required=True, help="Path to data_split.csv from the EDA notebook")
    parser.add_argument("--image_dir", required=True, help="Directory containing the chest X-ray images")
    parser.add_argument("--model", choices=["chexnet", "simple_cnn"], default="simple_cnn",
                         help="Architecture to train (default: simple_cnn — lighter for local training)")
    parser.add_argument("--num_blocks", type=int, default=4,
                        help="simple_cnn only: number of conv blocks (1-4). Each block halves "
                             "spatial dims, so this also sets total downsampling. Ignored for chexnet.")
    parser.add_argument("--width", type=float, default=1.0,
                        help="simple_cnn only: channel multiplier. 1.0 = 32/64/128/256 (~393K "
                             "params); 2.0 = 64/128/256/512 (~1.5M). The full-data run was still "
                             "underfitting at 15 epochs, so extra capacity is the indicated fix.")
    parser.add_argument("--pooling", choices=["avg", "avgmax"], default="avg",
                        help="simple_cnn only: 'avgmax' concatenates global max pooling with "
                             "global average pooling. Targets small focal findings (Mass, Nodule), "
                             "whose signal average pooling dilutes across the feature map.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Low LR for fine-tuning a pretrained backbone")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience, in epochs, on val AUC")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_oversampling", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pos_weight", type=float, default=20.0,
                        help="Cap on per-class pos_weight. Uncapped, Hernia reaches ~493, "
                             "which destabilises training.")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--predictions_dir", default="predictions",
                        help="Where to write per-image test predictions for the fairness audit")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_oversampling=not args.no_oversampling,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        max_pos_weight=args.max_pos_weight,
        return_test_meta=True,
    )

    model = build_model(args.model, dropout=args.dropout, pretrained=True,
                        num_blocks=args.num_blocks, width=args.width,
                        pooling=args.pooling).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}"
          + (f" ({args.num_blocks} blocks, width {args.width}x, {args.pooling} pooling)"
             if args.model == "simple_cnn" else "")
          + f" — {n_params:,} parameters")
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
    test_loss, test_mean_auc, test_per_class_auc, y_true, y_pred, row_idx = evaluate(
        model, test_loader, device, criterion, return_predictions=True)
    print(f"\nTest set — loss={test_loss:.4f} | mean_auc={test_mean_auc:.4f}")
    for cls, auc in zip(NIH_CLASSES, test_per_class_auc):
        print(f"  {cls:<20}: {auc:.4f}" if not np.isnan(auc) else f"  {cls:<20}: n/a (single class in test set)")

    # Persist test results — the run log only records per-epoch validation metrics,
    # so without this the test numbers and per-image predictions are lost on exit.
    run = args.run_name or "run"
    logger.log_test(test_loss, test_mean_auc, test_per_class_auc)
    save_test_predictions(test_loader.dataset, y_true, y_pred, row_idx,
                          os.path.join(args.predictions_dir, f"{run}_test_predictions.csv"))

    # Validation predictions too, so the fairness audit can pick decision thresholds on
    # validation data rather than calibrating on the test set it then reports.
    val_loader.dataset.return_meta = True
    _, _, _, v_true, v_pred, v_idx = evaluate(
        model, val_loader, device, criterion, return_predictions=True)
    save_test_predictions(val_loader.dataset, v_true, v_pred, v_idx,
                          os.path.join(args.predictions_dir, f"{run}_val_predictions.csv"))


if __name__ == "__main__":
    main()
