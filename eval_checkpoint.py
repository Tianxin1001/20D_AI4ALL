"""
Runs a trained checkpoint over the test (and optionally val) split and writes
per-image predictions, without re-training.

Exists because train.py only ever writes predictions/ as the tail end of a
full training run. simple_cnn_full.pt is already trained (see
reports/full_dataset_run_report.md) — this reproduces the same
predictions/simple_cnn_full_test_predictions.csv that run would have written
under its original setup: data/data_split.csv, seed 42, img_size 224, 4-block
simple_cnn. Re-uses evaluate() and save_test_predictions() from train.py so
the output schema matches exactly what fairness_eval.ipynb expects.

    python eval_checkpoint.py --checkpoint checkpoints/simple_cnn_full.pt \
        --csv_path data/data_split.csv --image_dir data/images \
        --run_name simple_cnn_full
"""
import argparse

import torch
import torch.nn as nn

from models import build_model
from preprocessing import get_dataloaders, set_seed
from train import evaluate, save_test_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint and dump per-image predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model", choices=["chexnet", "simple_cnn"], default="simple_cnn")
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--pooling", choices=["avg", "avgmax"], default="avg")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["test"],
                        help="Which splits to dump predictions for")
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
        num_workers=args.num_workers,
        max_pos_weight=args.max_pos_weight,
        return_test_meta=True,
    )

    model = build_model(args.model, pretrained=False, num_blocks=args.num_blocks,
                        width=args.width, pooling=args.pooling).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {args.checkpoint} (epoch {checkpoint.get('epoch')}, "
          f"val_mean_auc={checkpoint.get('val_mean_auc')})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))

    if "test" in args.splits:
        test_loss, test_mean_auc, _, y_true, y_pred, row_idx = evaluate(
            model, test_loader, device, criterion, return_predictions=True)
        print(f"Test — loss={test_loss:.4f} | mean_auc={test_mean_auc:.4f}")
        save_test_predictions(test_loader.dataset, y_true, y_pred, row_idx,
                              f"{args.predictions_dir}/{args.run_name}_test_predictions.csv")

    if "val" in args.splits:
        val_loader.dataset.return_meta = True
        val_loss, val_mean_auc, _, v_true, v_pred, v_idx = evaluate(
            model, val_loader, device, criterion, return_predictions=True)
        print(f"Val — loss={val_loss:.4f} | mean_auc={val_mean_auc:.4f}")
        save_test_predictions(val_loader.dataset, v_true, v_pred, v_idx,
                              f"{args.predictions_dir}/{args.run_name}_val_predictions.csv")


if __name__ == "__main__":
    main()
