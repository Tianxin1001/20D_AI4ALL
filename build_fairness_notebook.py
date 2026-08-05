"""Generates fairness_eval.ipynb.

The notebook is the deliverable; this script exists so it can be regenerated after edits
without hand-editing notebook JSON, and so the analysis is reviewable as plain Python in
git diffs.

    python build_fairness_notebook.py
"""
import json

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").split("\n")})


md("""
# Fairness audit — per-class recall by sex and age band

**AI4ALL Ignite 2026 · Chest X-ray Abnormality Detection: Making It Fair for Everyone**

> ### ⚠️ Status: analysis complete, results pending
>
> `PREDICTIONS` currently points at `smoke_full_test_predictions.csv` — a **one-epoch**
> pipeline-validation run, not a trained model. Every code cell has been executed end to end
> against it, so the analysis is verified to work; **the numbers it produces are not results and
> must not be quoted.**
>
> When the full training run finishes, change `PREDICTIONS` to
> `predictions/simple_cnn_full_test_predictions.csv`, re-run all cells, and complete the
> Findings table at the bottom. Nothing else needs to change.

## Research question

> Does the model achieve equal **recall** across sex and age groups — and if not, where does it
> fail, and does that failure track the age-correlated prevalence pattern found in EDA?

## Why recall, not accuracy or AUC

- **Accuracy** is meaningless here. Hernia is 227 / 112,120 images; a model that always predicts
  "no disease" scores 99.8% accuracy and catches nobody.
- **AUC** measures ranking ability and needs no threshold, which makes it good for comparing
  models — but it does not tell you how many patients were *missed*.
- **Recall** (sensitivity) = of the patients who truly have this finding, what fraction did the
  model flag? In medicine a false negative — a missed diagnosis — is the costly error. That is
  the quantity a fairness audit has to disaggregate.

## Inputs

`predictions/<run>_test_predictions.csv`, written by `train.py`: one row per test image, with
Patient ID, sex, age, view position, and both the true label and predicted probability for all
14 findings.
""")

code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Point this at the full-training-run predictions once available.
PREDICTIONS = 'predictions/smoke_full_test_predictions.csv'

AGE_BINS   = [0, 30, 50, 65, 200]
AGE_LABELS = ['<30', '30-49', '50-64', '65+']
MIN_CELL   = 20      # minimum positives per cell before we are willing to quote a recall
TARGET_RECALL = 0.80 # pooled operating point at which subgroups are compared

pd.set_option('display.width', 160)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
""")

code("""
df = pd.read_csv(PREDICTIONS)
CLASSES = [c[5:] for c in df.columns if c.startswith('true_')]

df['age_band'] = pd.cut(df['Patient Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
df['group'] = df['Patient Gender'].astype(str) + ' ' + df['age_band'].astype(str)
GROUPS = [f'{s} {b}' for b in AGE_LABELS for s in ['M', 'F']]

print(f'{len(df):,} test images from {df["Patient ID"].nunique():,} patients')
print(f'{len(CLASSES)} findings\\n')
print(pd.crosstab(df['age_band'], df['Patient Gender'], margins=True))
""")

md("""
---
## Step 1 — What can this test set actually answer?

Before computing anything, establish which subgroup cells are large enough to support a claim.
Recall is a proportion estimated from the *positive* cases in a cell; a cell with 8 positives
gives an estimate with a confidence interval so wide it cannot distinguish a real gap from
chance.

This step is the guard rail for the entire analysis. It also explains why the audit could not be
run on the 5,606-image sample: the earlier depth ablation showed per-class AUC swinging by up to
0.258 across runs on an 829-image test set, and splitting that eight ways would have left single
digits per cell.
""")

code("""
rows = []
for c in CLASSES:
    counts = {g: int(df.loc[df['group'] == g, f'true_{c}'].sum()) for g in GROUPS}
    rows.append({'finding': c, 'total_pos': int(df[f'true_{c}'].sum()),
                 **counts, 'min_cell': min(counts.values())})

cells = pd.DataFrame(rows).sort_values('total_pos', ascending=False).reset_index(drop=True)
cells['verdict'] = np.where(cells['min_cell'] >= MIN_CELL, 'all cells usable',
                    np.where(cells['min_cell'] >= 5, 'some cells too small', 'not analysable'))

ANALYSABLE = cells.loc[cells['min_cell'] >= MIN_CELL, 'finding'].tolist()
print(f'{len(ANALYSABLE)} of {len(CLASSES)} findings have >={MIN_CELL} positives in every cell:')
print('  ' + ', '.join(ANALYSABLE) + '\\n')
cells
""")

md("""
---
## Step 2 — Choosing a decision threshold

The model outputs probabilities; recall requires a yes/no decision, which requires a threshold.

Two choices matter, and both are stated explicitly because they shape every number that follows:

**Per class, not one global threshold.** A single 0.5 cutoff across all 14 findings would drive
recall to zero on the rare classes, since the model rarely assigns them high probability. Each
finding gets its own threshold.

**Set on the pooled test set to hit a common operating point, then held fixed across
subgroups.** Ideally the threshold would be picked on validation data, but `train.py` only
persists test predictions. Calibrating on pooled test data is a limitation to declare — however,
it does not bias the *disparity* measurement, which is what this audit is about: every subgroup
is judged at the identical threshold. The question is not "how good is the model" but "at one
operating point, is performance equal across groups?"
""")

code("""
def threshold_for_recall(y_true, y_prob, target):
    \"\"\"Lowest threshold achieving at least `target` pooled recall.\"\"\"
    pos = y_prob[y_true == 1]
    if len(pos) == 0:
        return np.nan
    return float(np.quantile(pos, 1 - target))


thresholds, thr_rows = {}, []
for c in CLASSES:
    y, p = df[f'true_{c}'].values, df[f'prob_{c}'].values
    t = threshold_for_recall(y, p, TARGET_RECALL)
    thresholds[c] = t
    pred = (p >= t).astype(int)
    thr_rows.append({
        'finding': c, 'threshold': round(t, 4),
        'pooled_recall': round(pred[y == 1].mean(), 3),
        'pooled_precision': round(y[pred == 1].mean(), 3) if pred.sum() else np.nan,
        'flagged_rate': round(pred.mean(), 3),
    })

pd.DataFrame(thr_rows).set_index('finding')
""")

md("""
`flagged_rate` is worth reading alongside recall. Where it approaches 1.0, the model is
achieving recall by flagging almost everything — recall parity across subgroups would then be
trivially satisfied and uninformative. Note any such class before drawing conclusions from it.
""")

md("""
---
## Step 3 — Recall by subgroup

For each finding and each (sex × age band) cell: recall, the positive count behind it, and a
**Wilson confidence interval**.

The interval is not decoration. With 27 positives, a recall of 0.70 carries an interval roughly
±0.17 wide — so a 10-point gap against another cell is not evidence of anything. Reporting recall
without an interval is the single easiest way to manufacture a false bias finding.
""")

code("""
def wilson(k, n, z=1.96):
    \"\"\"Wilson score interval for a binomial proportion — well behaved at small n,
    unlike the normal approximation, which can return bounds outside [0, 1].\"\"\"
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


records = []
for c in CLASSES:
    t = thresholds[c]
    for g in GROUPS:
        sub = df[df['group'] == g]
        pos = sub[sub[f'true_{c}'] == 1]
        n = len(pos)
        k = int((pos[f'prob_{c}'] >= t).sum())
        lo, hi = wilson(k, n)
        records.append({'finding': c, 'group': g, 'n_positive': n,
                        'recall': k / n if n else np.nan, 'ci_low': lo, 'ci_high': hi,
                        'reliable': n >= MIN_CELL})

recall_df = pd.DataFrame(records)
recall_df.loc[recall_df['finding'].isin(ANALYSABLE)].head(16)
""")

code("""
mat = (recall_df[recall_df['finding'].isin(ANALYSABLE)]
       .pivot(index='finding', columns='group', values='recall')
       .reindex(columns=GROUPS)
       .reindex(ANALYSABLE))
counts = (recall_df[recall_df['finding'].isin(ANALYSABLE)]
          .pivot(index='finding', columns='group', values='n_positive')
          .reindex(columns=GROUPS).reindex(ANALYSABLE))

fig, ax = plt.subplots(figsize=(10, 0.52 * len(mat) + 2.4))
norm = TwoSlopeNorm(vmin=np.nanmin(mat.values), vcenter=TARGET_RECALL,
                    vmax=max(np.nanmax(mat.values), TARGET_RECALL + 1e-6))
im = ax.imshow(mat.values, cmap='RdYlGn', norm=norm, aspect='auto')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v, n = mat.values[i, j], counts.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}\\nn={int(n)}', ha='center', va='center', fontsize=7.5)

ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, fontsize=9)
ax.set_yticks(range(len(mat)));    ax.set_yticklabels(mat.index, fontsize=9)
ax.set_title(f'Recall by sex and age band, at a pooled operating point of '
             f'{TARGET_RECALL:.0%}\\nGreen = above the pooled rate, red = below. '
             f'n = positive cases behind each estimate.', fontsize=11, loc='left', pad=12)
fig.colorbar(im, ax=ax, shrink=0.6, label='Recall')
fig.tight_layout()
fig.savefig('reports/figures/fig5_recall_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
""")

md("""
---
## Step 4 — Where are the gaps, and which are real?

A gap is only worth reporting if the two cells' confidence intervals do not overlap. Overlapping
intervals mean the data cannot distinguish the two groups, however large the point difference
looks.
""")

code("""
gaps = []
for c in ANALYSABLE:
    sub = recall_df[(recall_df['finding'] == c) & recall_df['reliable']]
    best, worst = sub.loc[sub['recall'].idxmax()], sub.loc[sub['recall'].idxmin()]
    gaps.append({
        'finding': c,
        'worst_group': worst['group'], 'worst_recall': round(worst['recall'], 3),
        'worst_n': int(worst['n_positive']),
        'best_group': best['group'], 'best_recall': round(best['recall'], 3),
        'best_n': int(best['n_positive']),
        'gap': round(best['recall'] - worst['recall'], 3),
        # Non-overlapping Wilson intervals -> the gap survives sampling uncertainty
        'significant': bool(worst['ci_high'] < best['ci_low']),
    })

gap_df = pd.DataFrame(gaps).sort_values('gap', ascending=False).reset_index(drop=True)
print(f\"{gap_df['significant'].sum()} of {len(gap_df)} findings show a gap with \"
      f\"non-overlapping confidence intervals\\n\")
gap_df
""")

code("""
# Collapse to one axis at a time — larger cells, so tighter intervals and clearer signal.
def marginal(by):
    out = []
    for c in ANALYSABLE:
        t = thresholds[c]
        for key, sub in df.groupby(by, observed=True):
            pos = sub[sub[f'true_{c}'] == 1]
            n = len(pos)
            if n < MIN_CELL:
                continue
            k = int((pos[f'prob_{c}'] >= t).sum())
            lo, hi = wilson(k, n)
            out.append({'finding': c, str(by): str(key), 'n_positive': n,
                        'recall': k / n, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(out)


by_sex = marginal('Patient Gender')
by_age = marginal('age_band')

print('Recall by sex, pooled over age:')
print(by_sex.pivot(index='finding', columns='Patient Gender', values='recall')
      .assign(gap=lambda x: (x['M'] - x['F']).round(3)).round(3).to_string(), '\\n')
print('Recall by age band, pooled over sex:')
print(by_age.pivot(index='finding', columns='age_band', values='recall')
      .reindex(columns=AGE_LABELS).round(3).to_string())
""")

md("""
---
## Step 5 — Does the pattern track age-correlated prevalence?

EDA established that disease prevalence rises with age (`reports/data_quality_report.md`, §4:
Hernia mean age 63.2 vs. overall 46.9). If recall drops in the age bands where a finding is
*rarer*, the mechanism is plausibly training-data scarcity for that group. If recall drops where
the finding is *common*, something else is happening — and that is the more interesting result,
because scarcity cannot explain it.
""")

code("""
prev = []
for c in ANALYSABLE:
    for b in AGE_LABELS:
        sub = df[df['age_band'] == b]
        prev.append({'finding': c, 'age_band': b,
                     'prevalence': sub[f'true_{c}'].mean(),
                     'n_images': len(sub)})
prev_df = pd.DataFrame(prev)

joined = by_age.merge(prev_df, on=['finding', 'age_band'])
corr = (joined.groupby('finding')
        .apply(lambda g: g['recall'].corr(g['prevalence']), include_groups=False)
        .rename('recall_vs_prevalence_corr').sort_values())

print('Correlation between subgroup recall and subgroup prevalence, per finding:')
print('  negative = recall is WORSE where the finding is more common\\n')
print(corr.round(2).to_string())
""")

code("""
fig, ax = plt.subplots(figsize=(7, 5))
for c in ANALYSABLE:
    g = joined[joined['finding'] == c].sort_values('age_band')
    ax.plot(g['prevalence'], g['recall'], marker='o', ms=4, lw=1, alpha=0.75, label=c)
ax.axhline(TARGET_RECALL, color='#888780', ls='--', lw=1)
ax.set_xlabel('Prevalence of the finding within the age band')
ax.set_ylabel('Recall within the age band')
ax.set_title('Is low recall explained by the finding being rare in that group?',
             fontsize=11, loc='left')
ax.legend(fontsize=7, ncol=2, frameon=False)
fig.tight_layout()
fig.savefig('reports/figures/fig6_recall_vs_prevalence.png', dpi=200, bbox_inches='tight')
plt.show()
""")

md("""
---
## Step 6 — Sensitivity check: patients, not images

Everything above treats each image as independent. It is not: 17,131 test images come from 4,621
patients, and one patient contributes up to dozens of images. That inflates effective sample size
and makes confidence intervals narrower than they should be.

Recomputing at patient level — a patient counts as detected if *any* of their positive images was
flagged — is a stricter reading. If the gaps survive, they are not an artefact of repeat imaging.
""")

code("""
pat_rows = []
for c in ANALYSABLE:
    t = thresholds[c]
    pos = df[df[f'true_{c}'] == 1].copy()
    pos['flagged'] = (pos[f'prob_{c}'] >= t).astype(int)
    per_patient = pos.groupby('Patient ID').agg(flagged=('flagged', 'max'),
                                                group=('group', 'first'))
    for g, sub in per_patient.groupby('group', observed=True):
        n = len(sub)
        if n < MIN_CELL:
            continue
        k = int(sub['flagged'].sum())
        lo, hi = wilson(k, n)
        pat_rows.append({'finding': c, 'group': g, 'n_patients': n,
                         'recall_patient': k / n, 'ci_low': lo, 'ci_high': hi})

patient_df = pd.DataFrame(pat_rows)
cmp = (patient_df.merge(recall_df, on=['finding', 'group'], suffixes=('_pat', '_img'))
       [['finding', 'group', 'n_patients', 'n_positive', 'recall_patient', 'recall']]
       .rename(columns={'n_positive': 'n_images', 'recall': 'recall_image'}))
cmp['delta'] = (cmp['recall_patient'] - cmp['recall_image']).round(3)
cmp.round(3).head(20)
""")

md("""
---
## Findings

*Fill in once run against the full-training-run predictions.*

| | |
|---|---|
| Findings with a statistically distinguishable gap | |
| Largest gap by sex | |
| Largest gap by age band | |
| Does it track age-correlated prevalence? | |
| Survives patient-level aggregation? | |

## Limitations

- **Thresholds are calibrated on pooled test data**, because `train.py` persists test predictions
  only. This does not bias the subgroup comparison — all groups share one threshold — but it does
  mean the absolute operating point is optimistic. Saving validation predictions would remove the
  caveat.
- **Labels are NLP-mined** from radiology reports with an estimated ~10% error rate, and label
  noise is not necessarily uniform across subgroups. A recall gap could partly reflect a labelling
  gap.
- **Hernia (10 positives in the test set) and other small classes are excluded** from the
  subgroup breakdown rather than reported with unusable intervals.
- **Sex is recorded binary** in this dataset; the audit inherits that limitation.
- **View position (AP/PA) is a known confound** — AP films are taken bedside on sicker patients,
  and AP share differs by age. A recall gap by age may partly be a gap by view position. Repeating
  the analysis stratified by view position is the natural next step.
- **Single trained model, single seed.** Run-to-run variance was measured in
  `reports/depth_ablation_report.md` and is not negligible.

## Next step — mitigation

Once the gaps are located, test at least one mitigation and re-measure:

1. **Age-stratified split ablation** — rebuild the train/test split stratified on age band as
   well as sex, retrain, and check whether the gap narrows. Reliable and directly interpretable.
2. **Confidence-weighted loss** — down-weight images whose labels are most likely NLP errors.
   More novel, but needs a defensible proxy for label uncertainty, which this dataset does not
   supply directly.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("fairness_eval.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote fairness_eval.ipynb — {len(CELLS)} cells "
      f"({sum(c['cell_type'] == 'code' for c in CELLS)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in CELLS)} markdown)")
