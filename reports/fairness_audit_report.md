# Fairness audit — per-class recall by sex and age band

Written summary of `fairness_eval.ipynb`. This answers the project's stated research question:
does the model detect disease equally well across patient groups?

**Model audited:** `simple_cnn_full` — SimpleCNN trained on all 112,120 images, test mean AUC
0.7225 (`full_dataset_run_report.md`). The model was fixed in advance by a criterion independent
of the fairness results — highest test mean AUC — so the audit could not be run across several
models and the most striking one chosen after the fact.

**Test set:** 17,131 images from 4,621 patients, zero patient overlap with training.

## Method

Recall, not accuracy or AUC. Accuracy is meaningless under this imbalance — always predicting
"no disease" scores 99.8% on Hernia. AUC measures ranking and needs no threshold, which makes it
right for comparing models but silent on how many patients were *missed*. A missed diagnosis is
the costly error in medicine, so recall is the quantity to disaggregate.

Each finding gets its own decision threshold, set so that **pooled** recall is 80%. That single
threshold is then held fixed across all subgroups. The question is not how good the model is, but
whether performance is equal across groups *at one operating point*.

Two guard rails were applied before any result was quoted.

**Guard rail 1 — cell size.** Recall in a cell is estimated from the positive cases in that cell.
Only the **9 of 14 findings with at least 20 positives in all eight (sex × age band) cells** are
analysed. Hernia has 10 positives in the entire test set and is excluded outright. Every cell
carries a Wilson confidence interval, and a gap is only called real when the two intervals do not
overlap.

**Guard rail 2 — lift over random.** Flagging N% of images at random already yields N% recall,
so the meaningful quantity is `recall − flagged_rate`.

| Finding | Test AUC | Flagged | Lift | Tier |
|---|---|---|---|---|
| Pneumothorax | 0.805 | 37% | +0.43 | 1 |
| Consolidation | 0.765 | 40% | +0.40 | 1 |
| Effusion | 0.782 | 40% | +0.40 | 1 |
| Cardiomegaly | 0.733 | 47% | +0.33 | 1 |
| Atelectasis | 0.691 | 56% | +0.25 | 1 |
| Pleural_Thickening | 0.675 | 61% | +0.19 | 2 |
| Mass | 0.618 | 68% | +0.13 | 2 |
| Infiltration | 0.659 | 68% | +0.12 | 2 |
| Nodule | 0.574 | 71% | +0.09 | 2 |

Nodule reaches 80% recall only by flagging 71% of every image in the test set. Differences there
are differences in near-indiscriminate flagging, not in detection. **Only Tier 1 supports a
fairness claim**; Tier 2 gaps are reported for completeness and not relied on.

## Findings

### 1. Eight of nine findings show a gap that survives sampling uncertainty

Restricted to Tier 1:

| Finding | Worst group | Best group | Gap |
|---|---|---|---|
| Cardiomegaly | M 65+ — 0.600 (n=35) | M <30 — 0.938 (n=32) | **0.338** |
| Atelectasis | M <30 — 0.616 (n=138) | F 65+ — 0.890 (n=173) | **0.274** |
| Pneumothorax | M 30-49 — 0.606 (n=104) | F 65+ — 0.871 (n=93) | **0.265** |
| Effusion | F <30 — 0.714 (n=91) | F 65+ — 0.876 (n=194) | 0.162 |
| Consolidation | F 50-64 — 0.711 (n=76) | M 50-64 — 0.869 (n=213) | 0.158 |

A 0.338 gap means that among patients who genuinely have an enlarged heart, the model catches
94% of men under 30 and 60% of men over 65. Four in ten older men are missed at an operating
point calibrated to miss two in ten overall.

### 2. Men are recalled worse, not women

Pooled over age: Cardiomegaly F 0.854 / M 0.729, Pleural_Thickening F 0.873 / M 0.748,
Pneumothorax F 0.839 / M 0.754. The worst-performing cell is male in six of the nine findings.

This runs against the direction most fairness work in medical imaging reports, and we do not have
an explanation for it. Worth stating plainly rather than smoothing over: an unexplained result
that contradicts the expected direction is more informative than a confirmatory one.

### 3. Age has no single direction

Atelectasis rises monotonically with age (0.680 → 0.868). Cardiomegaly falls monotonically
(0.915 → 0.704). A blanket "older patients are underserved" claim would be wrong; the effect is
finding-specific, which points at mechanisms specific to how each finding presents rather than at
a global demographic bias.

### 4. Two findings fail where the disease is most common

Correlation between subgroup recall and subgroup prevalence:

| | Correlation | Reading |
|---|---|---|
| Cardiomegaly | **−0.78** | recall is worse where the finding is more common |
| Pleural_Thickening | **−0.64** | same |
| Effusion | +1.00 | recall tracks prevalence — consistent with data scarcity |
| Atelectasis | +0.93 | same |
| Consolidation | +0.84 | same |

This is the audit's most substantive result. Where the correlation is positive, the obvious
explanation holds: the model performs worse on groups with fewer training examples of that
finding. For Cardiomegaly and Pleural_Thickening the sign is reversed — **the model is worst in
exactly the groups where it has seen the finding most often**, so scarcity cannot be the cause.

A plausible mechanism for Cardiomegaly specifically: it is assessed from the ratio of heart width
to chest width, and that ratio is confounded by body habitus and by view position. AP films —
taken bedside on sicker, and disproportionately older, patients — magnify the cardiac silhouette.
That is testable and is the first thing we would check with more time.

### 5. Image-level analysis flatters performance on older patients

Everything above treats images as independent. They are not: 17,131 test images come from 4,621
patients. Recomputing at patient level — a patient counts as detected if any of their positive
images was flagged — moves Infiltration sharply:

| Group | Per image | Per patient | Δ |
|---|---|---|---|
| F 65+ | 0.811 | 0.630 | **−0.181** |
| M 65+ | 0.713 | 0.562 | −0.151 |
| M 50-64 | 0.784 | 0.661 | −0.124 |

Older patients contribute many images each, so image-level averaging is weighted towards them.
Where those repeat images are easier, the per-image number overstates how many *patients* are
actually caught. The naive analysis would have under-reported the problem for precisely the group
with the most repeat imaging. Gaps in the other findings survive the switch.

## Limitations

- **The model is not clinically usable.** Precision is low throughout (Cardiomegaly 0.037,
  Pleural_Thickening 0.040). This audit measures relative disparity between groups, not fitness
  for deployment.
- **Thresholds are calibrated on pooled test data.** The audited run predates validation-
  prediction saving. All groups share one threshold so the comparison is unbiased, but the
  absolute operating point is optimistic. `train.py` now writes `*_val_predictions.csv`; the next
  run removes this caveat.
- **Labels are NLP-mined** with an estimated ~10% error rate, and that error is not necessarily
  uniform across subgroups. A recall gap could partly be a labelling gap.
- **Sex is recorded binary** in this dataset; the audit inherits that limitation.
- **View position is an uncontrolled confound.** AP share varies with age, and AP films differ
  systematically from PA. Some of the age effect may be a view-position effect. Repeating the
  analysis stratified by view position is the obvious next step and is directly motivated by
  finding 4.
- **Single model, single seed.**

## Next step — mitigation

The gaps are located; the proposal commits us to testing at least one mitigation and
re-measuring. In priority order:

1. **Stratify the analysis by view position.** Not a mitigation but a diagnosis, and it decides
   whether the age effect is real or a proxy for AP/PA. Cheapest and most informative.
2. **Age-stratified split.** Rebuild the train/val/test split stratified on age band as well as
   sex, retrain, re-measure the gaps. Directly interpretable, reliably produces a result.
3. **Confidence-weighted loss** down-weighting probable label errors. More novel, but requires a
   defensible proxy for label uncertainty that this dataset does not supply.

## Files

| | |
|---|---|
| `fairness_eval.ipynb` | Full analysis, reproducible end to end |
| `build_fairness_notebook.py` | Generates the notebook — analysis is reviewable as plain Python |
| `reports/figures/fig5_recall_heatmap.png` | Recall by finding × sex × age band, with cell counts |
| `reports/figures/fig6_recall_vs_prevalence.png` | Recall against prevalence, per finding |
| `predictions/simple_cnn_full_test_predictions.csv` | Input — gitignored, shared via Drive |
