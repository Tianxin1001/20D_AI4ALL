"""
Re-measures the fairness audit on a new model and compares it against the baseline.

Mitigation #2 from fairness_audit_report.md: rebuild the split stratified by age band as well as
sex, retrain, and re-measure the same recall gaps. This applies the audit's exact methodology
(per-model threshold calibrated to 80% pooled recall, Wilson intervals, >=20-positives guard
rail, Tier 1 = lift-over-random >= 0.20) to any predictions CSV, so it can be run on the
baseline (simple_cnn_full) and the age-stratified retrain and compared directly.

Threshold is recalibrated per model rather than reused, because each model's test set is a
different set of patients (the split changed) — the audit's own definition of a fair comparison
is "the same operating point" (80% pooled recall) on each model's own test set, not a shared
absolute threshold.

    python remeasure_fairness.py \
        --baseline predictions/simple_cnn_full_test_predictions.csv \
        --mitigated predictions/simple_cnn_age_stratified_test_predictions.csv
"""
import argparse

import numpy as np
import pandas as pd

AGE_BINS = [0, 30, 50, 65, 200]
AGE_LABELS = ['<30', '30-49', '50-64', '65+']
MIN_CELL = 20
TARGET_RECALL = 0.80
TIER1_LIFT_CUTOFF = 0.20


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


def analyse(path, label):
    df = pd.read_csv(path)
    df['age_band'] = pd.cut(df['Patient Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df['group'] = df['Patient Gender'].astype(str) + ' ' + df['age_band'].astype(str)
    groups = [f'{s} {b}' for b in AGE_LABELS for s in ['M', 'F']]
    classes = [c[5:] for c in df.columns if c.startswith('true_')]

    print(f'\n=== {label}: {path} ===')
    print(f'{len(df):,} test images from {df["Patient ID"].nunique():,} patients')

    rows = []
    for c in classes:
        y, p = df[f'true_{c}'].values, df[f'prob_{c}'].values
        t = threshold_for_recall(y, p, TARGET_RECALL)
        pred = (p >= t).astype(int)
        counts = {g: int(df.loc[df['group'] == g, f'true_{c}'].sum()) for g in groups}
        rows.append({
            'finding': c, 'threshold': t, 'min_cell': min(counts.values()),
            'lift_over_random': pred[y == 1].mean() - pred.mean() if y.sum() else np.nan,
        })
    meta = pd.DataFrame(rows).set_index('finding')
    analysable = meta.index[meta['min_cell'] >= MIN_CELL].tolist()
    tier1 = meta.index[(meta['min_cell'] >= MIN_CELL) &
                       (meta['lift_over_random'] >= TIER1_LIFT_CUTOFF)].tolist()
    print(f'{len(analysable)} analysable findings, {len(tier1)} Tier 1: {tier1}')

    records = []
    for c in tier1:
        t = meta.loc[c, 'threshold']
        for g in groups:
            sub = df[df['group'] == g]
            pos = sub[sub[f'true_{c}'] == 1]
            n = len(pos)
            k = int((pos[f'prob_{c}'] >= t).sum())
            lo, hi = wilson(k, n)
            records.append({'finding': c, 'group': g, 'n_positive': n,
                            'recall': k / n if n else np.nan, 'ci_low': lo, 'ci_high': hi,
                            'reliable': n >= MIN_CELL})
    recall_df = pd.DataFrame(records)

    summary = []
    for c in tier1:
        sub = recall_df[(recall_df['finding'] == c) & recall_df['reliable']]
        if sub.empty:
            continue
        worst = sub.loc[sub['recall'].idxmin()]
        best = sub.loc[sub['recall'].idxmax()]
        summary.append({'finding': c, 'worst_group': worst['group'], 'worst_recall': worst['recall'],
                        'worst_n': worst['n_positive'], 'best_group': best['group'],
                        'best_recall': best['recall'], 'best_n': best['n_positive'],
                        'gap': best['recall'] - worst['recall']})
    summary_df = pd.DataFrame(summary).set_index('finding').sort_values('gap', ascending=False)
    print(summary_df.round(3).to_string())
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Compare fairness gaps before vs. after a mitigation")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mitigated", required=True)
    args = parser.parse_args()

    base = analyse(args.baseline, "BASELINE (simple_cnn_full, sex-only stratified split)")
    mit = analyse(args.mitigated, "MITIGATED (age+sex-stratified split)")

    print("\n=== Gap comparison: baseline vs. age-stratified retrain ===")
    common = base.index.intersection(mit.index)
    comp = pd.DataFrame({
        'baseline_gap': base.loc[common, 'gap'],
        'mitigated_gap': mit.loc[common, 'gap'],
    })
    comp['delta'] = comp['mitigated_gap'] - comp['baseline_gap']
    comp['direction'] = np.where(comp['delta'] < 0, 'narrowed', 'widened')
    print(comp.round(3).sort_values('delta').to_string())

    dropped = set(base.index) - set(common)
    added = set(mit.index) - set(common)
    if dropped:
        print(f"\nTier 1 in baseline only (lost reliability after retrain): {sorted(dropped)}")
    if added:
        print(f"Tier 1 in mitigated only (gained reliability after retrain): {sorted(added)}")


if __name__ == "__main__":
    main()
