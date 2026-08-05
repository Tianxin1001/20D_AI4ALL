"""
View-position stratification — diagnosis for fairness_audit_report.md.

The audit found age effects with no single direction (Atelectasis recall rises
with age, Cardiomegaly falls) and, for Cardiomegaly/Pleural_Thickening, recall
*lowest* in the groups with the *most* training examples of the finding —
ruling out data scarcity as the explanation (see "Findings #4" in
fairness_audit_report.md). The report proposed AP/PA view position as an
uncontrolled confound: AP films are taken bedside on sicker, disproportionately
older patients, and AP systematically magnifies the cardiac silhouette used to
diagnose Cardiomegaly. preprocessing.py already retains View Position for
exactly this follow-up ("EDA note: ... Keep it for post-hoc per-view
evaluation").

This does not retrain or mitigate anything — it re-slices the existing test
predictions to check whether the age/sex gaps in the audit survive once view
position is controlled for, which decides whether the age-stratified retrain
(fairness_audit_report.md, "Next step — mitigation", #2) is targeting a real
effect or a proxy.

Same methodology as fairness_eval.ipynb: per-class threshold calibrated to 80%
pooled recall on the pooled test set (held fixed across every slice below),
Wilson score intervals, and the two guard rails (>=20 positives per cell,
lift-over-random >=0.20 for "Tier 1").

    python view_position_analysis.py
"""
import numpy as np
import pandas as pd

PREDICTIONS = 'predictions/simple_cnn_full_test_predictions.csv'
AGE_BINS = [0, 30, 50, 65, 200]
AGE_LABELS = ['<30', '30-49', '50-64', '65+']
MIN_CELL = 20
TARGET_RECALL = 0.80

# Tier 1 findings from fairness_audit_report.md, ranked by lift-over-random.
TIER1 = ['Pneumothorax', 'Consolidation', 'Effusion', 'Cardiomegaly', 'Atelectasis']

# The worst/best (sex, age_band) cell per Tier-1 finding, from the audit's Findings #1,
# so we can check directly whether each one survives a view-position split.
AUDIT_GAPS = {
    'Cardiomegaly':  {'worst': ('M', '65+'), 'best': ('M', '<30')},
    'Atelectasis':   {'worst': ('M', '<30'), 'best': ('F', '65+')},
    'Pneumothorax':  {'worst': ('M', '30-49'), 'best': ('F', '65+')},
    'Effusion':      {'worst': ('F', '<30'), 'best': ('F', '65+')},
    'Consolidation': {'worst': ('F', '50-64'), 'best': ('M', '50-64')},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_for_recall(y_true, y_prob, target):
    pos = y_prob[y_true == 1]
    return float(np.quantile(pos, 1 - target)) if len(pos) else np.nan


def recall_cell(sub, cls, t):
    pos = sub[sub[f'true_{cls}'] == 1]
    n = len(pos)
    k = int((pos[f'prob_{cls}'] >= t).sum())
    lo, hi = wilson(k, n)
    return {'n_positive': n, 'recall': k / n if n else np.nan,
            'ci_low': lo, 'ci_high': hi, 'reliable': n >= MIN_CELL}


def main():
    df = pd.read_csv(PREDICTIONS)
    df['age_band'] = pd.cut(df['Patient Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    print(f'{len(df):,} test images from {df["Patient ID"].nunique():,} patients\n')

    # --- Premise check: does AP share actually vary with age and sex? ---
    print('=== Premise: View Position distribution by age band ===')
    vp_age = pd.crosstab(df['age_band'], df['View Position'], normalize='index')
    print((vp_age * 100).round(1).to_string(), '\n')

    print('=== Premise: View Position distribution by sex ===')
    vp_sex = pd.crosstab(df['Patient Gender'], df['View Position'], normalize='index')
    print((vp_sex * 100).round(1).to_string(), '\n')

    # Thresholds held identical to the audited run: pooled, per-class, 80% target.
    thresholds = {c: threshold_for_recall(df[f'true_{c}'].values, df[f'prob_{c}'].values, TARGET_RECALL)
                  for c in TIER1}

    # --- For each Tier-1 finding: does the audit's worst/best cell gap survive
    # a view-position split, or does it collapse to a view-position effect? ---
    for cls in TIER1:
        t = thresholds[cls]
        gap = AUDIT_GAPS[cls]
        print(f'=== {cls} (threshold={t:.4f}) ===')

        for label, (sex, band) in [('worst', gap['worst']), ('best', gap['best'])]:
            cell = df[(df['Patient Gender'] == sex) & (df['age_band'] == band)]
            overall = recall_cell(cell, cls, t)
            print(f'  audit {label} cell {sex} {band}: recall={overall["recall"]:.3f} '
                  f'[{overall["ci_low"]:.3f}, {overall["ci_high"]:.3f}] n={overall["n_positive"]}')
            for view in ['AP', 'PA']:
                sub = cell[cell['View Position'] == view]
                r = recall_cell(sub, cls, t)
                flag = '' if r['reliable'] else '  (< 20 positives — not reliable)'
                print(f'    {view}: recall={r["recall"]:.3f} '
                      f'[{r["ci_low"]:.3f}, {r["ci_high"]:.3f}] n={r["n_positive"]}{flag}')

        # Coarser, better-powered cut: recall by age band collapsed to AP vs PA only
        # (sex pooled), so cells stay above the guard rail.
        print(f'  --- recall by age band x view position (sex pooled) ---')
        for band in AGE_LABELS:
            row = []
            for view in ['AP', 'PA']:
                sub = df[(df['age_band'] == band) & (df['View Position'] == view)]
                r = recall_cell(sub, cls, t)
                row.append(f'{view}={r["recall"]:.3f}(n={r["n_positive"]})' if r['reliable']
                           else f'{view}=n/a(n={r["n_positive"]})')
            print(f'    {band:>6}: ' + '  '.join(row))
        print()


if __name__ == '__main__':
    main()
