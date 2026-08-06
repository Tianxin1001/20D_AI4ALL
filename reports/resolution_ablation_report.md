# Resolution ablation — SimpleCNN at 224px vs. 128px

## Purpose

Two models had been trained on the Kaggle NIH sample dataset by different team members, but
never compared side by side under matched conditions:

- **CheXNet** (DenseNet-121, ImageNet-pretrained) — trained by Junaid, best val mean AUC **0.725**
- **SimpleCNN** (from-scratch, ~393K params) — trained by Krisha, best val mean AUC **0.646**

The 0.079 gap could not be attributed, because the two runs differed in **two** variables at
once: architecture (DenseNet-121 vs. SimpleCNN) *and* input resolution (224×224 vs. 128×128).
This is a confound.

This run isolates the resolution variable: **SimpleCNN trained at 224px, with every other
hyperparameter held identical to Krisha's 128px baseline.**

### Hypothesis

Krisha's weakest classes were all small, focal findings (Mass 0.494, Pleural_Thickening 0.571,
Nodule 0.554) while her strongest were large or diffuse (Edema 0.819, Cardiomegaly 0.429 aside,
Effusion 0.657). A nodule is radiologically defined as <3 cm; on a 1024×1024 source image that
is roughly 88 px, but only ~11 px at 128×128 input. After SimpleCNN's four max-pool stages
(16× downsampling) such a lesion occupies well under one cell of the final feature map.

We therefore predicted a **differential effect**: raising input resolution should improve
small-lesion classes substantially more than large/diffuse classes.

A competing mechanism was identified in advance: `SimpleCNN` ends with
`AdaptiveAvgPool2d(1)` (global average pooling), which averages the entire feature map into a
single vector. Since >95% of a chest radiograph is normal tissue, a focal lesion's signal is
diluted — and *more* so at higher resolution (1/196 of a 14×14 map at 224px, vs. 1/64 of an
8×8 map at 128px). The net direction was therefore not predictable a priori and had to be
measured.

## Setup

Identical to `sample_dataset_training_report.md` except for `--img_size`.

| | |
|---|---|
| Dataset | Kaggle NIH sample — 5,606 images |
| Split | `make_data_split.py --seed 42` → 3,961 train / 816 val / 829 test; zero patient overlap |
| Model | `SimpleCNN` (`--model simple_cnn`) |
| **Image size** | **224 × 224** (baseline: 128 × 128) |
| Batch size | 16 |
| Learning rate | 1e-4, `ReduceLROnPlateau` (factor 0.1, patience 2) |
| Dropout | 0.2 |
| Label smoothing | 0.1 |
| Oversampling | On |
| Epochs | 15, early stopping disabled (`--patience 15`) so the full curve is comparable |
| Device | MPS (Apple GPU), ~60 s/epoch |

Note: the pre-existing `pos_weight` issues (uncapped values, and positive counts computed
*after* label smoothing) were deliberately **left unfixed** for this run, so that resolution
remains the only changed variable.

Per-epoch and test results: `simple_cnn_224_results.csv`.

## Results

**Reproducibility check:** a 1-epoch smoke test and the full run produced an identical epoch-1
val mean AUC of 0.5870, confirming `set_seed(42)` gives deterministic runs.

### Aggregate

| | 128px (baseline) | 224px | Δ |
|---|---|---|---|
| Best val mean AUC | 0.6465 (ep 14) | 0.6457 (ep 13) | **−0.0008** |
| Final-5-epoch val mean | 0.6387 | 0.6415 | +0.0028 |
| **Test mean AUC** | **0.5868** | **0.5912** | **+0.0044** |

Within-run epoch-to-epoch swing was 0.055 (128px) and 0.059 (224px) — roughly **20× larger
than the difference between the two runs**. The correct statement is that no difference was
detected, not that the two are equal.

### Per-class test ROC-AUC

| Class | 128px | 224px | Δ |
|---|---|---|---|
| Atelectasis | 0.647 | 0.634 | −0.013 |
| Cardiomegaly | 0.429 | 0.444 | +0.015 |
| Effusion | 0.657 | 0.669 | +0.012 |
| Infiltration | 0.610 | 0.612 | +0.002 |
| **Mass** | **0.494** | **0.495** | **+0.001** |
| **Nodule** | **0.554** | **0.565** | **+0.010** |
| Pneumonia | 0.526 | 0.534 | +0.008 |
| Pneumothorax | 0.563 | 0.624 | +0.061 |
| Consolidation | 0.601 | 0.604 | +0.003 |
| Edema | 0.819 | 0.821 | +0.003 |
| Emphysema | 0.455 | 0.461 | +0.006 |
| Fibrosis | 0.703 | 0.646 | −0.057 |
| **Pleural_Thickening** | **0.571** | **0.578** | **+0.007** |
| Hernia | n/a | n/a | — |
| **Mean (13 defined)** | **0.5867** | **0.5912** | **+0.0044** |

Hernia is undefined on this split — the 829-image test set contains no positive Hernia case
(227 / 112,120 in the full dataset).

### Hypothesis test

| Group | Classes | Mean Δ |
|---|---|---|
| **Hypothesis target** (small, focal) | Mass, Nodule, Pleural_Thickening | **+0.006** |
| **Control** (large, diffuse) | Edema, Cardiomegaly, Infiltration | **+0.007** |

The hypothesis predicted a differential effect. The two groups moved by the same amount, so the
hypothesis is **not supported**. Mass — the single class the hypothesis most directly targeted —
moved +0.001.

The two largest per-class changes (Pneumothorax +0.061, Fibrosis −0.057) are in opposite
directions and neither belongs to the small-lesion group; with a diff standard deviation of
0.025 these are consistent with run-to-run noise rather than a resolution effect.

## Findings

1. **Input resolution has no measurable effect on SimpleCNN's performance on this dataset**,
   either in aggregate or differentially by lesion size.

2. **This isolates CheXNet's advantage as architectural.** With resolution now matched at 224px:

   | Comparison | Gap |
   |---|---|
   | CheXNet@224 vs. SimpleCNN@128 (original, confounded) | +0.078 |
   | CheXNet@224 vs. SimpleCNN@224 (resolution-matched) | **+0.079** |

   The gap is unchanged, so resolution contributes approximately nothing to it. The remaining
   explanation is the DenseNet-121 architecture and its ImageNet pretraining. Consistent with
   this, CheXNet's **first** epoch (0.710) already exceeded SimpleCNN's best across 15 epochs
   (0.646) — the signature of transfer learning rather than of a better-tuned run.

3. **The likely bottleneck is architectural, not input fidelity.** Global average pooling
   collapses all spatial information before the classifier, diluting focal-lesion signal
   regardless of input resolution. If small-lesion performance is a priority, the productive
   changes are to the pooling/head (e.g. max or attention pooling, or a patch-based head), not
   to `--img_size`.

4. **The learning-rate schedule is doing real work.** Val mean AUC stalled at epochs 7–8,
   triggering `ReduceLROnPlateau` (1e-4 → 1e-5); epoch 9 jumped +0.031 to 0.6408 and val loss
   dropped from 1.83 to 1.66 and remained stable thereafter.

## Limitations

- Single seed, single run per configuration. With within-run swing of ~0.06, distinguishing
  effects smaller than ~0.02 would require multiple seeds.
- All results are on the 5,606-image sample (5% of the full dataset). Absolute AUCs are low
  and should not be read as model capability. Some classes have very few positives in the
  sample (Junaid reported Pneumonia ≈ 39 images, Hernia ≈ 9), so their per-class AUCs are
  close to meaningless.
- Junaid's CheXNet run logged validation metrics only, so all three-way comparisons use
  validation figures; the 128 vs. 224 comparison uses test figures and is unaffected.
- Junaid's run predates `make_data_split.py`; its split came from the EDA notebook. The logic
  and seed match, but this was not verified byte-for-byte.

## Implications for the fairness audit

The project's core research question is per-class recall disaggregated by sex and age band. At
a test mean AUC of ~0.59, subgroup recall differences are not separable from noise, so
architecture selection is not independent of the fairness deliverable — a model at ~0.72 is a
materially better substrate for that analysis.

## Recommendation

Re-open the CheXNet decision with resolution-matched evidence. The July 29 decision to drop
CheXNet rested on the observation that its sample-dataset AUCs were low, but that comparison
was never made under matched conditions, and the low absolute numbers are better explained by
sample size (5,606 images; 9 positive Hernia cases) than by architecture.

The compute objection that motivated the decision is real — CheXNet overheated a laptop — but
it is addressable: the full NIH dataset is a public Kaggle dataset that mounts directly in a
Kaggle GPU notebook, requiring no local download and no local hardware.

Three options for the team:

1. Run CheXNet once on the full dataset via Kaggle GPU and decide on real evidence.
2. Keep SimpleCNN and instead modify its pooling/head, which finding 3 identifies as the
   actual bottleneck.
3. Run both in parallel, per the existing division of work.

## Reproduce

```bash
python make_data_split.py \
  --csv_path nih_sample/sample/sample_labels.csv \
  --output data_split_sample.csv --seed 42

python train.py --model simple_cnn \
  --csv_path data_split_sample.csv \
  --image_dir nih_sample/sample/images \
  --img_size 224 --batch_size 16 --lr 1e-4 \
  --dropout 0.2 --label_smoothing 0.1 \
  --epochs 15 --patience 15 --num_workers 2 \
  --run_name simple_cnn_224
```

## Known gaps in the pipeline

Surfaced while preparing this run; all block the full-dataset run or the fairness audit:

1. `NIHChestXrayDataset` resolves images as `image_dir / filename`, assuming a flat directory.
   The full dataset is nested across `images_001/images/` … `images_012/images/`.
2. `compute_class_weights()` is uncapped (Hernia `pos_weight` ≈ 493) and counts positives
   *after* label smoothing, inflating every weight by ~11%.
3. `NIHChestXrayDataset.__getitem__` returns `(image, label)` only — no patient ID, sex, or
   age — so predictions cannot be joined back to demographics for subgroup analysis.
4. `ExperimentLogger` records per-epoch validation metrics only. Test-set results are printed
   but never written to disk, and per-image predictions are discarded entirely.
