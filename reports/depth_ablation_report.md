# Depth ablation — SimpleCNN with 2, 3 and 4 conv blocks

Follow-up to `resolution_ablation_report.md`. Read that first for context.

## Purpose

The resolution ablation found that input size (128 vs. 224) had no measurable effect. One
plausible explanation is that the network discards fine spatial detail internally regardless of
what it is fed: `SimpleCNN` has four max-pool stages, so it downsamples 16×, and a 224px input
arrives at the classifier as a 14×14 feature map.

This ablation tests that explanation by varying network depth, which directly controls total
downsampling:

| Blocks | Channels | Downsampling | 224px → final map | Params |
|---|---|---|---|---|
| 2 | 3→32→64 | 4× | 56 × 56 | ~20K |
| 3 | 3→32→64→128 | 8× | 28 × 28 | ~96K |
| 4 (baseline) | 3→32→64→128→256 | 16× | 14 × 14 | ~393K |

### Hypothesis

If spatial-detail loss is the bottleneck for small, focal findings, reducing depth should
improve small-lesion classes (Mass, Nodule, Pleural_Thickening) relative to large/diffuse
control classes (Edema, Cardiomegaly, Infiltration) — a differential effect, not a uniform one.

Two mechanisms were expected to push the other way: fewer blocks means less channel capacity
and a smaller receptive field, and global average pooling dilutes a focal signal over *more*
cells when the final map is larger (3,136 cells at 2 blocks vs. 196 at 4).

## Setup

`models.py` was extended with a `num_blocks` parameter (default 4, so existing behaviour is
unchanged) and `train.py` with a matching `--num_blocks` flag. All other hyperparameters are
identical to the 224px run in `resolution_ablation_report.md`: sample dataset, `seed 42`
patient-level split, batch 16, lr 1e-4, dropout 0.2, label smoothing 0.1, oversampling on,
15 epochs, early stopping disabled.

Per-epoch and test results: `simple_cnn_224_3blocks_results.csv`,
`simple_cnn_224_2blocks_results.csv`.

## Results

### Aggregate

| | 4 blocks | 3 blocks | 2 blocks |
|---|---|---|---|
| Best val mean AUC | 0.6457 | 0.6341 | 0.5803 |
| **Test mean AUC** | **0.5912** | **0.5775** | **0.5802** |
| Final train loss | 1.99 | 2.13 | 2.23 |
| Final val loss | 1.64 | 1.74 | 1.76 |

Train and validation loss both rise monotonically as depth is reduced, i.e. the shallower
models underfit — the expected consequence of cutting channel capacity from 256 to 128 to 64.
Test mean AUC, however, is **not** monotonic (0.591 → 0.578 → 0.580).

### Per-class test ROC-AUC

| Class | 4 blk | 3 blk | 2 blk | Range | Monotonic |
|---|---|---|---|---|---|
| Atelectasis | 0.634 | 0.625 | 0.567 | 0.067 | yes |
| Cardiomegaly | 0.444 | 0.522 | 0.702 | **0.258** | yes |
| Effusion | 0.669 | 0.620 | 0.618 | 0.051 | yes |
| Infiltration | 0.612 | 0.597 | 0.600 | 0.015 | no |
| **Mass** (small) | 0.495 | 0.506 | 0.487 | 0.019 | no |
| **Nodule** (small) | 0.565 | 0.575 | 0.552 | 0.023 | no |
| Pneumonia | 0.534 | 0.445 | 0.559 | 0.115 | no |
| Pneumothorax | 0.624 | 0.570 | 0.589 | 0.054 | no |
| Consolidation | 0.604 | 0.603 | 0.553 | 0.051 | yes |
| Edema | 0.821 | 0.801 | 0.731 | 0.090 | yes |
| Emphysema | 0.461 | 0.401 | 0.418 | 0.060 | no |
| Fibrosis | 0.646 | 0.657 | 0.525 | 0.133 | no |
| **Pleural_Thickening** (small) | 0.578 | 0.586 | 0.639 | 0.061 | yes |

Mean per-class range across the three depths: **0.077**. Non-monotonic in **7 of 13** classes.

### Hypothesis test

At 3 blocks the three small-lesion classes all moved in the predicted direction
(+0.012, +0.010, +0.008; group mean +0.010) while the other ten averaged −0.021. Taken alone
this looks like the predicted differential effect.

**It did not replicate at 2 blocks.** Mass fell back to 0.487 (below the 4-block baseline) and
Nodule to 0.552 (also below baseline). Only Pleural_Thickening continued to rise. The
group-level mean still increases monotonically (0.546 → 0.556 → 0.560), but that trend rests
entirely on one class out of three.

**The hypothesis is not supported.**

## Findings

1. **These differences are noise-dominated.** The claimed small-lesion effect is ~0.010, while
   the mean per-class swing across the three configurations is 0.077 — nearly an order of
   magnitude larger. Seven of thirteen classes move non-monotonically, which is not what a
   systematic architectural effect looks like.

2. **The clearest evidence is a control class.** Cardiomegaly swings 0.444 → 0.522 → 0.702, a
   range of 0.258 — larger than every small-lesion class combined, in a class the hypothesis
   predicted should be least affected. No architectural account explains this. Note also that
   0.444 is *below chance* for a finding that CheXNet scored 0.767 on in this project's own
   sample run; an unstable per-class estimate is the parsimonious explanation.

3. **The 829-image test set cannot resolve effects of this size.** This is the substantive
   result of the ablation, and it supersedes the architectural question.

4. **Depth does cost capacity, as expected.** Train and val loss both degrade monotonically with
   fewer blocks. That part of the picture is consistent and unsurprising; it simply isn't
   visible in per-class AUC because the noise floor is higher than the effect.

## Consequence for the fairness audit

The project's core research question is per-class recall disaggregated by sex and age band.
This ablation puts a hard number on why that cannot be answered from the sample dataset:

If a per-class AUC computed over the **whole** 829-image test set swings by up to 0.26 between
runs, then splitting that same test set into sex × age-band cells — roughly 50–100 images each,
with single-digit positive counts for most findings — yields subgroup recall estimates that are
entirely noise. Any "recall gap" measured there would be unreproducible.

**The fairness audit requires the full 112,120-image dataset.** This is not a preference for
more data; it is a measured floor on what the sample can support.

## Limitations

- Single seed per configuration. Repeating each depth across several seeds would let us report
  a proper noise band instead of inferring one from between-configuration spread.
- Depth was reduced from the top, so it is confounded with channel width (2 blocks caps at 64
  channels) and receptive-field size. A cleaner isolation would hold width constant and vary
  only stride/pooling.
- Sample dataset only; several classes have too few positives for a stable estimate
  (Pneumonia ≈ 39 images, Hernia ≈ 9). Hernia is undefined throughout.

## Process note

An earlier reading of the 3-block run treated the three same-direction small-lesion moves as a
confirmed differential effect. That call was premature: the effect size (0.010) had not been
compared against the noise floor, which the 2-block run subsequently showed to be ~0.077. The
correction generalises — **quantify run-to-run variance before reporting an effect**, and
prefer a control group whose spread can be measured over a target group read in isolation.

## Reproduce

```bash
for n in 2 3; do
  python train.py --model simple_cnn --num_blocks $n \
    --csv_path data_split_sample.csv \
    --image_dir nih_sample/sample/images \
    --img_size 224 --batch_size 16 --lr 1e-4 \
    --dropout 0.2 --label_smoothing 0.1 \
    --epochs 15 --patience 15 --num_workers 2 \
    --run_name simple_cnn_224_${n}blocks
done
```

## Next steps

Three ablations on the sample dataset (resolution ×1, depth ×2) now converge on the same
conclusion: **the sample is exhausted as an experimental substrate.** Further architecture
search on 5,606 images cannot produce a result that survives its own noise.

The pipeline gaps listed at the end of `resolution_ablation_report.md` should be fixed and a
full-dataset run started, after which the fairness audit becomes measurable.
