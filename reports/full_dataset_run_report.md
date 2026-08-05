# First full-dataset training run — SimpleCNN on 112,120 images

Run name `simple_cnn_full`. This is the project's first training run on the complete NIH
release; every earlier result in `reports/` was produced on the 5,606-image Kaggle sample.

## Why this run exists

Three ablations on the sample (`resolution_ablation_report.md`, `depth_ablation_report.md`)
converged on the same conclusion: the sample's 829-image test set could not resolve the
differences we were trying to measure. Per-class AUC swung by up to 0.258 between
configurations, against architectural effects of around 0.01. Continuing to tune on it could
not produce a result that survived its own noise.

This also matches the guidance given in the Week 11 check-in on July 30 — train on the full
training set rather than a subsample.

Four pipeline defects had to be fixed first; all four were invisible on the sample and would
have surfaced only here. They are documented at the end of `resolution_ablation_report.md` and
were fixed in commit `933d215`.

## Setup

| | |
|---|---|
| Dataset | NIH ChestX-ray14, full release — 112,120 images / 30,805 patients |
| Split | `make_data_split.py --seed 42` → 77,971 train / 17,002 val / 17,131 test; zero patient overlap verified |
| Model | `SimpleCNN`, 4 blocks, 392,974 parameters, trained from scratch |
| Image size | 224 × 224 |
| Batch size | 64 |
| Learning rate | 1e-4, `ReduceLROnPlateau` |
| Dropout / label smoothing | 0.2 / 0.1 |
| Oversampling | on |
| `pos_weight` cap | 20 (uncapped, Hernia reached 458) |
| Epochs | 15, early stopping patience 4 |
| Hardware | Kaggle T4 ×2, ~850 s/epoch, ~3.5 hours total |

Sanity checks that passed on first contact with the full data, each confirming a fix:

- `Indexed 112,120 images across 12 directories` — the recursive path index resolving the
  nested `images_001/images/` … `images_012/images/` layout
- `Removed 16 records with impossible age > 100` — exactly the 16 records documented in
  `data_quality_report.md`
- `View position breakdown — PA: 67,299 | AP: 44,805` — matching the documented 60/40 split
- `Verified patient-level isolation: No patient overlap across splits`

## Results

**Test mean AUC: 0.7225.** Best validation mean AUC 0.7101, reached at epoch 15.

### Per-class test ROC-AUC, against the best sample-based run

| Class | Sample @224 | **Full dataset** | Gain |
|---|---|---|---|
| Emphysema | 0.461 | **0.823** | **+0.362** |
| Cardiomegaly | 0.444 | **0.733** | **+0.290** |
| Pneumothorax | 0.624 | 0.805 | +0.181 |
| Consolidation | 0.604 | 0.765 | +0.161 |
| Mass | 0.495 | 0.618 | +0.123 |
| Effusion | 0.669 | 0.782 | +0.113 |
| Pleural_Thickening | 0.578 | 0.675 | +0.098 |
| Pneumonia | 0.534 | 0.627 | +0.093 |
| Atelectasis | 0.634 | 0.691 | +0.057 |
| Infiltration | 0.612 | 0.659 | +0.047 |
| Fibrosis | 0.646 | 0.693 | +0.047 |
| Edema | 0.821 | 0.857 | +0.036 |
| Nodule | 0.565 | 0.574 | +0.010 |
| Hernia | n/a | **0.812** | — |
| **Mean (13 shared)** | **0.5912** | **0.7156** | **+0.124** |

Hernia is measurable for the first time. The sample's 829-image test split contained no positive
Hernia case at all, so its AUC was undefined in every previous run.

Cardiomegaly moving from 0.444 — *below chance* — to 0.733 confirms the depth ablation's
conclusion that its instability was a sample-size artefact, not a property of the model.

## Findings

**1. Data volume dominated every architectural change by more than an order of magnitude.**

| Change | Effect on test mean AUC |
|---|---|
| Input resolution, 128 → 224 | +0.004 |
| Depth, 4 → 3 blocks | −0.014 |
| Depth, 4 → 2 blocks | −0.011 |
| **Sample → full dataset** | **+0.131** |

Same model, same code, same hyperparameters. The dataset was worth roughly thirty times more
than anything we changed about the architecture.

**2. The model was still underfitting when the run ended.** Best validation AUC landed on the
*final* epoch (0.7101) and early stopping never triggered. Training loss fell 1.65 → 1.48 and
validation loss 1.29 → 1.14, both still decreasing, with no divergence between them. There is
headroom in both epochs and capacity — this run should not be read as the architecture's ceiling.

**3. Small focal findings remain the weakest classes, and this is now a robust result.** Mass
(0.618) and Nodule (0.574) are the two lowest. They were also the two lowest in every sample
run, but there the finding was inside the noise floor; on a 17,131-image test set it is not.
This is consistent with the mechanism proposed in `depth_ablation_report.md`: global average
pooling dilutes a focal lesion's signal across the whole feature map. It motivates the `avgmax`
pooling change tested in run `simple_cnn_v2_full`.

**4. The architecture question became much less important.** The comparison that drove earlier
discussion was CheXNet at 0.725 versus SimpleCNN at 0.646, both on the sample. SimpleCNN on the
full dataset reaches 0.7225 — level with CheXNet's sample result using a model with
eighteen times fewer parameters. Pretraining matters most when data is scarce; with 77,971
training images that advantage shrinks. The question is not settled — CheXNet on the full set
would likely score higher, and the published result is 0.841 — but it is no longer the decision
that most affects the project.

## What this unblocks

Per-image test predictions (17,131 rows, with Patient ID, sex, age and view position) are
written to `predictions/simple_cnn_full_test_predictions.csv`. Cell counts by sex × age band
give at least 20 positive cases in all eight cells for 9 of the 14 findings, which is the
threshold `fairness_eval.ipynb` requires before quoting a recall. The fairness audit — the
project's core research question — is measurable for the first time.

## Limitations

- Single run, single seed. Run-to-run variance was quantified in `depth_ablation_report.md` and
  is not negligible, though the 17,131-image test set here is 20× the sample's.
- Stopped at 15 epochs while still improving, for scheduling reasons rather than convergence.
- Absolute performance sits well below the published CheXNet benchmark of 0.841 on this
  dataset. That gap is expected — 393K parameters trained from scratch versus 7M pretrained on
  ImageNet — and should be stated plainly rather than omitted.
- Labels remain NLP-mined with an estimated ~10% error rate; nothing about training on more
  data addresses label quality.

## Files

| File | Contents |
|---|---|
| `reports/simple_cnn_full_results.csv` | Per-epoch validation metrics and the final test row |
| `predictions/simple_cnn_full_test_predictions.csv` | Per-image predictions — gitignored, ~4 MB, shared via Drive |
| `checkpoints/simple_cnn_full.pt` | Trained weights — gitignored, ~1.6 MB, shared via Drive |

Note on naming: `sample_dataset_full_training_results.csv` predates this run and refers to a
complete 15-epoch run *on the sample*, not to the full dataset.
