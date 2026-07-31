# Sample Dataset Training Report — SimpleCNN Pipeline Sanity Check

## Purpose

Before committing to a long training run on the full 112,120-image dataset, we ran the new
`SimpleCNN` architecture (see `models.py`) against the Kaggle NIH sample dataset (5,606 images)
to confirm the whole pipeline — patient-level split, data loading, class weighting,
oversampling, multi-label loss, AUC evaluation, checkpointing — works correctly end-to-end.
These are **not** tuned or complete training runs (1-2 epochs each); treat the AUCs as a
sanity signal, not a performance benchmark.

## Setup

| | |
|---|---|
| Dataset | Kaggle NIH sample (`nih_sample/`) — 5,606 images |
| Split | Patient-level 70/15/15 — 3,961 train / 816 val / 829 test images, zero patient overlap |
| Model | `SimpleCNN` — 4 conv blocks (32→64→128→256 ch), global avg pool, dropout, linear head — ~393K params, trained from scratch |
| Image size | 128×128 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Label smoothing | 0.1 |
| Oversampling | On (rare-class weighted sampler) |

Two runs were done back-to-back:

| Run | Epochs | Device | Purpose |
|---|---|---|---|
| `sanity_check` | 2 | CPU | Initial pipeline validation |
| `sanity_check_mps` | 1 | MPS (Apple GPU) | Confirmed `train.py`'s device-selection bug fix — it previously only checked for CUDA and silently ran on CPU even when MPS was available |

## Results

Full per-class numbers are in `sample_dataset_training_results.csv` (val = per-epoch validation AUC, test = final test-set AUC recomputed from the saved checkpoint).

**Validation mean AUC by epoch:**

| Run | Epoch 1 | Epoch 2 |
|---|---|---|
| sanity_check (CPU) | 0.593 | 0.606 |
| sanity_check_mps (MPS) | 0.599 | — |

**Test-set mean AUC:** 0.580 (sanity_check) / 0.580 (sanity_check_mps) — consistent between the two runs, as expected for the same architecture/data with only a 1-epoch difference.

**Per-class test AUC, sanity_check (2 epochs, CPU):**

| Class | AUC |
|---|---|
| Edema | 0.795 |
| Pleural_Thickening | 0.624 |
| Consolidation | 0.556 → *(see note)* |
| Infiltration | 0.602 |
| Pneumothorax | 0.596 |
| Atelectasis | 0.620 |
| Cardiomegaly | 0.571 |
| Effusion | 0.584 |
| Nodule | 0.543 |
| Pneumonia | 0.558 |
| Mass | 0.503 |
| Fibrosis | 0.564 |
| Emphysema | 0.421 |
| Hernia | n/a (no positive Hernia cases in the sample's test split — expected given its extreme rarity) |

## Findings

1. **Pipeline works end-to-end.** Both runs completed without errors, produced sane loss curves (decreasing train loss, stable val loss), and mean test AUC (~0.58) sits meaningfully above the 0.50 random baseline after just 1-2 epochs — confirming data loading, oversampling, class weighting, and multi-label AUC evaluation are all wired correctly.
2. **Some signal is already visible on easier classes** (Edema ~0.80, Pleural_Thickening ~0.62, Atelectasis ~0.62, Infiltration ~0.60) even this early, which is a reasonable sign for the architecture.
3. **Weaker/near-chance classes** (Emphysema ~0.42-0.46, Mass ~0.50-0.52) are unsurprising at 1-2 epochs on a 5,606-image subset — too little training and too little data per class to expect real separation yet.
4. **Hernia AUC is undefined** on this sample's test split — it's NIH's rarest class by far (227/112,120 in the full dataset), and the 5,606-image sample apparently has no positive Hernia case in its ~829-image test split, so ROC-AUC can't be computed (matches `train.py`'s documented behavior for single-class columns).
5. **Bug fix confirmed:** the MPS run validated that `train.py` now correctly selects Apple's GPU backend instead of silently falling back to CPU.
6. Discovered and fixed a **separate, more serious bug** in `preprocessing.py`'s age-cleaning step while preparing the full dataset: the age-suffix regex only matched `sample_labels.csv`'s `"058Y"`-style format, silently converting every age to `NaN` (and therefore dropping every row) when run against the full dataset's plain-integer age format. Both sample and full datasets now parse correctly — see `preprocessing.py` lines ~304-312.

## Next step

A real multi-epoch training run (`simple_cnn`, 15 epochs, early stopping patience 4) is in progress on the full 112,120-image dataset; results will be reported separately once it completes.
