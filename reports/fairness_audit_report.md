# Fairness audit — per-class recall by sex and age band

Written summary of `fairness_eval.ipynb`. This answers the project's stated research question:
does the model detect disease equally well across patient groups?

**Model audited:** `simple_cnn_v2_full` — SimpleCNN with width 2.0 and avg+max pooling, trained
on all 112,120 images, **test mean AUC 0.7647** (`full_dataset_run_report.md`).

The model was fixed in advance by a criterion independent of the fairness results — highest test
mean AUC — so the audit could not be run across several models and the most striking one chosen
afterwards. Two models were trained; this is the better one by that rule.

**Test set:** 17,131 images from 4,621 patients, zero patient overlap with training.

## Method

Recall, not accuracy or AUC. Accuracy is meaningless under this imbalance — always predicting
"no disease" scores 99.8% on Hernia. AUC measures ranking and needs no threshold, which makes it
right for comparing models but silent on how many patients were *missed*. A missed diagnosis is
the costly error in medicine, so recall is the quantity to disaggregate.

Each finding gets its own decision threshold, **fitted on the validation split** so that pooled
validation recall is 80%, then applied unchanged to test. Calibrating the operating point on the
data you then report would make the reported recall optimistic; fitting on validation avoids
that. Every subgroup is judged at the identical threshold, so the question is not how good the
model is but whether performance is equal across groups *at one operating point*.

Two guard rails were applied before any result was quoted.

**Guard rail 1 — cell size.** Recall in a cell is estimated from the positive cases in that cell.
Only the **9 of 14 findings with at least 20 positives in all eight (sex × age band) cells** are
analysed. Hernia has 10 positives in the entire test set and is excluded outright. Every cell
carries a Wilson confidence interval, and a gap is only called real when the two intervals do not
overlap.

**Guard rail 2 — lift over random.** Flagging N% of images at random already yields N% recall,
so the meaningful quantity is `recall − flagged_rate`.

| Finding | Test AUC | Flagged | Recall | Lift | Tier |
|---|---|---|---|---|---|
| Cardiomegaly | 0.868 | 28% | 0.839 | +0.563 | 1 |
| Pneumothorax | 0.843 | 33% | 0.813 | +0.479 | 1 |
| Effusion | 0.842 | 33% | 0.800 | +0.474 | 1 |
| Consolidation | 0.781 | 46% | 0.855 | +0.396 | 1 |
| Atelectasis | 0.747 | 48% | 0.797 | +0.317 | 1 |
| Mass | 0.742 | 47% | 0.770 | +0.301 | 1 |
| Pleural_Thickening | 0.700 | 50% | 0.749 | +0.244 | 1 |
| Infiltration | 0.679 | 67% | 0.817 | +0.148 | 2 |
| Nodule | 0.622 | 60% | 0.733 | +0.134 | 2 |

Nodule and Infiltration reach their recall largely by flagging liberally; differences there are
differences in near-indiscriminate flagging, not in detection. **Only Tier 1 supports a fairness
claim.**

## Findings

### 1. Four of seven Tier-1 findings show a gap that survives sampling uncertainty

| Finding | Worst group | Best group | Gap | Distinguishable |
|---|---|---|---|---|
| Pneumothorax | M 30-49 — 0.635 (n=104) | F 30-49 — 0.892 (n=176) | **0.257** | yes |
| Atelectasis | M <30 — 0.616 (n=138) | F 65+ — 0.855 (n=173) | **0.240** | yes |
| Cardiomegaly | M 65+ — 0.714 (n=35) | M <30 — 0.938 (n=32) | 0.223 | no |
| Mass | F 30-49 — 0.675 (n=114) | M 30-49 — 0.889 (n=126) | **0.213** | yes |
| Effusion | F <30 — 0.681 (n=91) | F 65+ — 0.866 (n=194) | **0.185** | yes |
| Pleural_Thickening | M 50-64 — 0.678 (n=152) | F 65+ — 0.821 (n=39) | 0.143 | no |
| Consolidation | F 50-64 — 0.763 (n=76) | M 50-64 — 0.892 (n=213) | 0.129 | no |

The Pneumothorax gap is the most consequential. Among patients who genuinely have a collapsed
lung, the model flags 89% of women aged 30-49 and 64% of men in the same age band. Pneumothorax
is acute and time-critical; a missed one is not a deferred diagnosis.

Note the two gaps are within the *same* age band, so this is a sex effect that cannot be
explained by age composition.

### 2. Neither sex nor age has a consistent direction

Men are recalled worse on Cardiomegaly (F 0.887 / M 0.777) and Pneumothorax (F 0.861 / M 0.756).
Women are recalled worse on Mass (M 0.795 / F 0.727) and Nodule (M 0.760 / F 0.693). Across age,
Atelectasis improves with age (0.649 → 0.839) while Cardiomegaly declines (0.932 → 0.817), and
Mass is worst in the middle bands.

No blanket claim — "the model underserves women", "the model underserves the elderly" — is
supported. The bias is finding-specific, which points at mechanisms tied to how each finding
presents rather than at a global demographic effect. This is a less quotable result than a single
headline disparity, and it is the one the data supports.

### 3. Mass fails where the disease is most common

Correlation between subgroup recall and subgroup prevalence:

| | Correlation | Reading |
|---|---|---|
| Mass | **−0.83** | recall is worse where the finding is more common |
| Cardiomegaly | −0.47 | same, weaker |
| Pleural_Thickening | −0.22 | same, weak |
| Pneumothorax | +0.91 | recall tracks prevalence — consistent with data scarcity |
| Atelectasis | +0.73 | same |
| Effusion | +0.72 | same |

Where the correlation is positive, the obvious explanation holds: the model does worse on groups
it has seen fewer examples from. **For Mass the sign is reversed** — the model is worst in
exactly the age bands where masses occur most often. Scarcity cannot be the cause, so something
else is producing the gap, and Mass is the finding most worth investigating further.

### 4. Image-level analysis flatters performance on older patients

Everything above treats images as independent. They are not: 17,131 test images come from 4,621
patients. Recomputing at patient level — a patient counts as detected if any of their positive
images was flagged — moves Infiltration sharply:

| Group | Per image | Per patient | Δ |
|---|---|---|---|
| F 65+ | 0.798 | 0.575 | **−0.223** |
| M 65+ | 0.784 | 0.657 | −0.126 |
| F 30-49 | 0.833 | 0.691 | −0.142 |

Older patients contribute many images each, so image-level averaging is weighted towards them.
Where those repeat images are easier, the per-image number overstates how many *patients* are
actually caught. Reporting per-image recall alone would understate the problem for precisely the
group with the most repeat imaging.

### 5. Improving the model closed some gaps and revealed another

The audit was run first on `simple_cnn_full` (0.7225) and then on `simple_cnn_v2_full` (0.7647).
Comparing the two is effectively an intervention study — extra capacity plus `avgmax` pooling,
evaluated against the fairness metric rather than against AUC.

| Finding | Gap v1 | Gap v2 | Effect |
|---|---|---|---|
| Cardiomegaly | 0.338 (distinguishable) | 0.223 (not) | **closed** |
| Consolidation | 0.158 (distinguishable) | 0.129 (not) | **closed** |
| Atelectasis | 0.274 | 0.240 | persists |
| Pneumothorax | 0.265 | 0.257 | persists |
| Effusion | 0.162 | 0.185 | persists |
| Mass | 0.125 (not) | 0.213 (distinguishable) | **emerged** |

Two closed, three persisted, one emerged.

The Cardiomegaly result says its gap was a symptom of model weakness: the group difference
disappeared once the model got better at the task, without anything being done about fairness
directly. The three that persisted did not respond to more capacity, which makes them the
candidates for targeted mitigation.

Mass is the instructive case. Its gap only *appeared* in the better model — in v1 the Mass
operating point required flagging 68% of images, so the comparison was meaningless and no gap was
measurable. **Fixing performance can reveal a disparity that a weak model was hiding.** "No
measurable gap" from a poor model is not reassurance.

## Limitations

- **Precision is low throughout** (Pleural_Thickening 0.046, Cardiomegaly 0.067). This model is
  not clinically usable; the audit measures relative disparity between groups, not fitness for
  deployment.
- **Nodule and Infiltration remain Tier 2** and their gaps carry no weight.
- **Labels are NLP-mined** with an estimated ~10% error rate, and that error is not necessarily
  uniform across subgroups. A recall gap could partly be a labelling gap.
- **Sex is recorded binary** in this dataset; the audit inherits that limitation.
- **View position is an uncontrolled confound.** AP share varies with age, and AP films differ
  systematically from PA. Some of the age effect may be a view-position effect.
- **Single model, single seed**, and the model was still underfitting at 25 epochs.

## Next step — mitigation

Finding 5 already reports one intervention (capacity + pooling) measured against the fairness
metric. The gaps that survived it need something targeted:

1. **Stratify by view position.** A diagnosis rather than a mitigation, and it decides whether the
   age effects are real or proxies for AP/PA. Cheapest and most informative, and directly
   motivated by finding 3.
2. **Age- and sex-stratified split.** Rebuild the split stratified on age band as well as sex,
   retrain, re-measure. Directly interpretable and reliably produces a result.
3. **Threshold adjustment per subgroup.** Equalising recall by group is a post-hoc fix that trades
   precision for parity. Worth measuring even if not deployed, because it quantifies the cost of
   the parity we do not currently have.

## Files

| | |
|---|---|
| `fairness_eval.ipynb` | Full analysis, reproducible end to end |
| `build_fairness_notebook.py` | Generates the notebook — reviewable as plain Python |
| `reports/figures/fig5_recall_heatmap.png` | Recall by finding × sex × age band, with cell counts |
| `reports/figures/fig6_recall_vs_prevalence.png` | Recall against prevalence, per finding |
| `predictions/simple_cnn_v2_full_*.csv` | Inputs — gitignored, shared via Drive |
