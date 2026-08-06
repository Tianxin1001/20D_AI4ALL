# Age-stratified split — mitigation #2, re-measured

Follow-up to `fairness_audit_report.md` ("Next step — mitigation", #2) and
`view_position_stratification_report.md`, which ruled out view position as the explanation for
the Cardiomegaly age gap and left the age-stratified retrain as the next real test. This is that
retrain, re-measured with the audit's own methodology.

**Model:** `simple_cnn` — identical architecture, hyperparameters, and seed (42) to the audited
`simple_cnn_full` baseline (4 blocks, width 1.0, avg pooling). The only intended variable that
changed is the split: `make_age_stratified_split.py` rebuilds train/val/test stratified jointly
on age band and sex (confirmed balanced — age-band share is ~18/36/33/13% across train/val/test,
vs. the baseline's sex-only stratification).

**Deviation, stated up front:** trained at **128px, not 224px**, to fit a same-day time budget —
epoch 1 at 224px measured 95.5 min (real, post-setup compute, not stall time); at 128px it was
10.6 min. This is not an uncontrolled shortcut: `resolution_ablation_report.md` already measured
128 vs. 224 on this exact architecture and found a negligible +0.004 AUC difference, because the
four max-pool stages absorb the extra detail either way. Test mean AUC came in at 0.7133 here
vs. 0.7225 for the 224px baseline — a 0.009 gap, consistent with that prior finding plus a
different (harder or easier) test-set composition, not a resolution effect.

## Results

| | Baseline (`simple_cnn_full`) | Mitigated (age-stratified, 128px) |
|---|---|---|
| Split | sex-only stratified | age + sex stratified |
| Test set | 17,131 images / 4,621 patients | 16,531 images / 4,621 patients |
| Test mean AUC | 0.7225 | 0.7133 |
| Best val mean AUC | 0.7101 (epoch 15) | 0.7327 (epoch 15) |

Same threshold methodology as the audit: per-class threshold calibrated to 80% pooled recall on
each model's own test set (not shared across models, since the test-set patients differ), Wilson
intervals, same two guard rails (>=20 positives/cell, Tier 1 = lift-over-random >= 0.20).

### Gaps that are Tier 1 in both models — the only directly comparable set

| Finding | Baseline gap | Mitigated gap | Δ | Direction |
|---|---|---|---|---|
| Pneumothorax | 0.265 | 0.122 | **-0.143** | narrowed |
| Atelectasis | 0.274 | 0.185 | **-0.089** | narrowed |
| Effusion | 0.162 | 0.139 | -0.023 | narrowed |
| Consolidation | 0.158 | 0.224 | +0.066 | **widened** |

Pneumothorax's gap more than halved and Atelectasis narrowed meaningfully — real movement, not
noise-sized. Effusion barely moved. Consolidation got worse, in the opposite direction from what
a mitigation aimed at age/sex balance should do — worth flagging rather than omitting.

### Cardiomegaly: no longer measurable, not resolved

Cardiomegaly was the audit's largest gap (0.338) and the one `view_position_stratification_report.md`
confirmed was *not* a view-position artifact. It doesn't appear in the mitigated model's Tier 1
list at all. Checked directly: its lift-over-random is still high (0.437, well above the 0.20
cutoff) — this is **not** a case of the operating point becoming meaningless. It fails the other
guard rail: the F <30 cell has only 14 positives in this test set (vs. the 20 required), because
the age-stratified split put fewer young-female Cardiomegaly cases in test. **This is an
artifact of the split changing test-set composition, not evidence the gap closed.** The honest
statement is "not measurable here," not "fixed."

### Emphysema: a new gap the baseline split couldn't see

Emphysema wasn't analysable in the original audit (didn't clear the guard rail on the sex-only
split) and is now Tier 1 with the **largest gap of any finding in this run**: 0.275
(M 30-49 — 0.643, n=28, vs. M 50-64 — 0.918, n=97; lift 0.348, min cell 28, both guard rails
cleared cleanly). Same pattern as `fairness_audit_report.md`'s Mass result on the v2 model: a
change made for one reason (here, split composition; there, model capacity) revealed a
disparity the previous setup was structurally blind to. "No gap measured" was never the same
claim as "no gap exists."

## Interpretation

Mixed, not clean. The age-stratified split measurably helped two of the four gaps it could be
tested against (Pneumothorax substantially, Atelectasis meaningfully), left one flat, and made
one worse. It also cost the ability to measure the audit's single largest finding
(Cardiomegaly) and surfaced a previously invisible one (Emphysema) that's now the largest gap in
this run. A team reporting "the mitigation worked" without the Cardiomegaly and Emphysema
caveats would be overstating it in one direction and understating it in another.

## Limitations

- **Single seed**, same as every prior run in this project — none of these deltas have an error
  bar around them yet.
- **128px, not 224px** — justified by the resolution ablation, but it is a second changed
  variable alongside the split, even if prior evidence says its effect is near zero.
- **Test-set composition differs between the two models** (different patients, by design of
  re-stratifying), which is what caused the Cardiomegaly guard-rail failure. A cleaner
  before/after would hold the test set fixed and only change training — not possible here since
  the split itself is the intervention being tested.
- Consolidation's regression and Emphysema's new gap are both first-run findings — same caveat
  the audit applied to itself: no follow-up done yet to explain *why*.

## Files

| | |
|---|---|
| `make_age_stratified_split.py` | Builds the age+sex-stratified split |
| `data/data_split_age_stratified.csv` | The split used for this run (gitignored, local) |
| `checkpoints/simple_cnn_age_stratified_128.pt` | Trained weights (gitignored) |
| `predictions/simple_cnn_age_stratified_128_test_predictions.csv` | Per-image predictions (gitignored) |
| `remeasure_fairness.py` | Reusable before/after comparison, run against any two prediction files |
| `reports/simple_cnn_age_stratified_128_results.csv` | Per-epoch validation metrics and the final test row |
