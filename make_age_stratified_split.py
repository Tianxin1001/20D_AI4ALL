"""
Age-stratified train/val/test split — mitigation #2 from fairness_audit_report.md.

make_data_split.py stratifies patient-level splits by sex only. The fairness audit
found recall gaps by age band that view_position_analysis.py ruled out as a view-position
artifact for Cardiomegaly, so the next test is whether an age-stratified split (equal age-band
representation in train/val/test, not just equal sex representation) changes anything.

Same 70/15/15 patient-level split logic and seed as make_data_split.py, stratified jointly on
Gender x age band instead of Gender alone. Age is parsed here (not in make_data_split.py, which
leaves it raw) only to build the stratification key; the output CSV still carries raw
Patient Age through unchanged, so preprocessing.py's own parsing is unaffected.

    python make_age_stratified_split.py --csv_path data/Data_Entry_2017.csv \
        --output data/data_split_age_stratified.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AGE_BINS = [0, 30, 50, 65, 200]
AGE_LABELS = ['<30', '30-49', '50-64', '65+']


def parse_age(raw):
    parts = raw.astype(str).str.extract(r'(\d+)\s*([YMD])?')
    value = pd.to_numeric(parts[0], errors='coerce')
    unit = parts[1]
    return np.select([unit == 'Y', unit == 'M', unit == 'D'],
                      [value, value / 12, value / 365], default=value)


def make_age_sex_stratified_split(df, test_size=0.30, val_fraction=0.50, seed=42):
    df = df.copy()
    df['_age_years'] = parse_age(df['Patient Age'])
    df = df[df['_age_years'] <= 100]  # same 16-record outlier exclusion as preprocessing.py

    patient_df = (df.groupby('Patient ID')
                  .agg(Gender=('Patient Gender', 'first'), Age=('_age_years', 'first'))
                  .reset_index())
    patient_df['age_band'] = pd.cut(patient_df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    patient_df['stratum'] = patient_df['Gender'].astype(str) + '_' + patient_df['age_band'].astype(str)

    # A handful of strata may have too few patients for a 3-way stratified split; sklearn
    # requires >= 2 members per class at each split. Report and drop patients in such strata
    # from stratification (they still get assigned, just via a coarser fallback: Gender only).
    counts = patient_df['stratum'].value_counts()
    rare = counts[counts < 4].index.tolist()
    if rare:
        print(f"[WARNING] Strata with <4 patients, falling back to Gender-only stratification "
              f"for {patient_df['stratum'].isin(rare).sum()} patients: {rare}")
        patient_df.loc[patient_df['stratum'].isin(rare), 'stratum'] = \
            patient_df.loc[patient_df['stratum'].isin(rare), 'Gender'].astype(str)

    train_ids, temp_ids = train_test_split(
        patient_df['Patient ID'], test_size=test_size,
        random_state=seed, stratify=patient_df['stratum'],
    )
    temp_df = patient_df[patient_df['Patient ID'].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df['Patient ID'], test_size=val_fraction,
        random_state=seed, stratify=temp_df['stratum'],
    )

    df = df.copy()
    df['Split'] = 'train'
    df.loc[df['Patient ID'].isin(val_ids), 'Split'] = 'val'
    df.loc[df['Patient ID'].isin(test_ids), 'Split'] = 'test'
    return df.drop(columns=['_age_years']), patient_df


def main():
    parser = argparse.ArgumentParser(description="Generate an age+sex-stratified data_split.csv")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df, patient_df = make_age_sex_stratified_split(df, seed=args.seed)

    keep_cols = ['Image Index', 'Patient ID', 'Finding Labels', 'Patient Age',
                 'Patient Gender', 'View Position', 'Split']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(args.output, index=False)

    sets = {s: set(df[df['Split'] == s]['Patient ID']) for s in ['train', 'val', 'test']}
    print(f"Split sizes — "
          f"train: {len(df[df['Split']=='train']):,} imgs / {len(sets['train']):,} patients | "
          f"val: {len(df[df['Split']=='val']):,} imgs / {len(sets['val']):,} patients | "
          f"test: {len(df[df['Split']=='test']):,} imgs / {len(sets['test']):,} patients")

    overlaps = {p: sets[a] & sets[b] for p, (a, b) in
                {'train/val': ('train', 'val'), 'train/test': ('train', 'test'),
                 'val/test': ('val', 'test')}.items()}
    if any(overlaps.values()):
        for pair, overlap in overlaps.items():
            if overlap:
                print(f"[WARNING] {pair} patient overlap: {len(overlap)}")
    else:
        print("Verified patient-level isolation: zero overlap across splits.")

    # Confirm age-band balance actually improved vs. sex-only stratification.
    patient_split = patient_df.merge(
        df.groupby('Patient ID')['Split'].first().reset_index(), on='Patient ID')
    print("\nAge-band share by split (target: roughly equal columns per row):")
    print((pd.crosstab(patient_split['age_band'], patient_split['Split'], normalize='columns') * 100)
          .round(1).to_string())

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
