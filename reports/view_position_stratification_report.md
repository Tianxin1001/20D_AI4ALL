# View-position stratification — diagnosis, not mitigation

Follow-up to `fairness_audit_report.md`, item 1 of "Next step — mitigation": stratify the
existing audit by View Position (AP/PA) before touching training, since it decides whether the
age effect reported there is real or a proxy for AP/PA positioning.

**Model audited:** `simple_cnn_full`, same checkpoint and test set as the fairness audit
(`checkpoints/simple_cnn_full.pt`, 17,131 test images / 4,621 patients). Predictions were not on
disk (`predictions/` is gitignored, shared via Drive) and were regenerated with
`eval_checkpoint.py`, which reuses `train.py`'s own `evaluate()` / `save_test_predictions()` so
the output matches the original schema exactly. Reproduced test mean AUC: **0.7225**, matching
`full_dataset_run_report.md` to four decimal places — confirms this is the same run.

## Method

Identical to `fairness_eval.ipynb`: per-class threshold calibrated to 80% pooled recall on the
pooled test set, held fixed across every slice below; Wilson score intervals; the same two guard
rails (>=20 positives per cell before a recall is quoted, Tier 1 = lift-over-random >= 0.20).
Restricted to the five Tier 1 findings, since Tier 2 gaps were never reliable to begin with.

Script: `view_position_analysis.py`.

## Premise check: does AP share actually vary by age or sex?

| | AP | PA |
|---|---|---|
| **By age band** <30 / 30-49 / 50-64 / 65+ | 42.5% / 38.4% / 40.8% / 43.9% | 57.5% / 61.6% / 59.2% / 56.1% |
| **By sex** F / M | 38.4% / 42.5% | 61.6% / 57.5% |

AP share is close to flat across age bands (38–44%) and differs by only 4 points between sexes.
This matters for interpretation below: **composition alone is too flat to launder a large gap
through Simpson's paradox.** If the audit's age/sex gaps were purely an artifact of AP/PA mix
shifting between groups, that mix would need to shift by a lot more than 4 points to produce
gaps of 0.16–0.34 recall. It doesn't — so whatever is happening is not simple compositional
confounding.

## Finding 1 — view position is a bigger effect than anything the audit measured

At every age band, the AP-vs-PA recall gap is larger than the age/sex gaps reported in
`fairness_audit_report.md`, for four of the five Tier 1 findings:

| Finding | <30 | 30-49 | 50-64 | 65+ | Direction |
|---|---|---|---|---|---|
| Pneumothorax | AP 0.622 vs PA 0.913 (n=45/69) | AP 0.551 vs PA 0.874 (n=98/182) | AP 0.752 vs PA 0.870 (n=121/177) | AP 0.783 vs PA 0.873 (n=83/79) | PA ≫ AP, every band |
| Consolidation | AP 0.981 vs PA 0.297 (n=103/37) | AP 0.981 vs PA 0.268 (n=158/71) | AP 0.977 vs PA 0.392 (n=215/74) | AP 0.953 vs PA 0.276 (n=107/29) | AP ≫ PA, every band, reversed |
| Effusion | AP 0.860 vs PA 0.637 (n=171/102) | AP 0.861 vs PA 0.686 (n=316/271) | AP 0.917 vs PA 0.683 (n=387/378) | AP 0.933 vs PA 0.714 (n=254/199) | AP > PA, every band |
| Atelectasis | AP 0.753 vs PA 0.553 (n=146/85) | AP 0.815 vs PA 0.668 (n=286/211) | AP 0.911 vs PA 0.771 (n=327/353) | AP 0.912 vs PA 0.821 (n=193/179) | AP > PA, every band |

Consolidation is the extreme case: an AP-vs-PA gap of **0.68–0.71 recall**, more than double the
largest gap the original audit found on any axis, and the sign flips relative to Pneumothorax
(PA wins) — so this isn't "AP films are just worse," it's finding-specific, the same pattern the
audit found for age (Findings #3). This is a materially larger, previously undocumented
disparity axis, and it holds within every age band, so it is not explained by the age
composition either.

## Finding 2 — the Cardiomegaly age effect is *not* a view-position artifact

The audit's specific hypothesis (Findings #4) was that AP films — bedside, on sicker/older
patients — magnify the cardiac silhouette and could explain why Cardiomegaly recall falls
monotonically with age. The per-cell worst/best breakdown from the audit has too few positives
to split by view reliably (both land under the n=20 guard rail), so the coarser age-band x view
cut (sex pooled) is the one to read:

| Age band | AP | PA |
|---|---|---|
| <30 | 0.929 (n=28) | 0.903 (n=31) |
| 30-49 | 0.780 (n=50) | 0.825 (n=63) |
| 50-64 | 0.727 (n=55) | 0.840 (n=81) |
| 65+ | 0.667 (n=36) | 0.743 (n=35) |

The decline with age is present in **both** AP-only and PA-only subsets, at similar magnitude
(AP: 0.929 → 0.667, a 0.262 drop; PA: 0.903 → 0.743, a 0.160 drop). If AP positioning were doing
the work, the PA-only subset should show little to no age effect. It doesn't. **The hypothesis
in the audit report is not supported by this data** — view position is not the mechanism behind
the Cardiomegaly age gap. Whatever is causing it (body habitus, chest geometry changing with age,
something else) is still open.

## What this changes about the mitigation plan

- **The age-stratified retrain (`fairness_audit_report.md`, mitigation #2) is still justified**
  for Cardiomegaly specifically — the effect it targets is not a view-position confound, so
  retraining against it addresses something real.
- **View position itself now needs to be added as its own audited axis, not just diagnosed
  away.** It produces larger gaps than sex or age for 4 of 5 Tier 1 findings and was not in the
  original audit's scope at all. This is the omission the mitigation round should supplement:
  before or alongside the age-stratified retrain, extend `fairness_eval.ipynb` to report recall
  by (finding x view position) as a first-class result, not a footnote.
- Consolidation's 0.7-point swing is large enough that it's worth checking whether it reflects a
  genuine detection difference or a labeling/prevalence artifact specific to AP framing (e.g.
  consolidation is easier to read on AP due to positioning) before treating it as a fairness
  finding — this audit only establishes that the gap exists and is real (guard rails passed), not
  its cause.

## Limitations

- Same single-model, single-seed limitation as the underlying audit.
- Sex x age x view three-way cells mostly fail the n>=20 guard rail (Cardiomegaly worst/best
  cells above are explicitly flagged unreliable at that granularity) — conclusions here use the
  coarser two-way cuts where the three-way cells were too small.
- This is a diagnosis of an existing model, not a mitigation. No retraining happened.

## Files

| | |
|---|---|
| `eval_checkpoint.py` | Regenerates per-image predictions from a trained checkpoint without retraining |
| `view_position_analysis.py` | This analysis, reproducible end to end |
| `predictions/simple_cnn_full_test_predictions.csv` | Regenerated input — gitignored, matches the audited run (test mean AUC 0.7225) |
