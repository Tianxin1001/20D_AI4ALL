"""
Generates a patient-level train/val/test split CSV from a raw NIH metadata file
(either sample_labels.csv or Data_Entry_2017.csv).

Pulled out of notebooks/NIH_ChestXray_EDA_Full.ipynb (cells 29-30) so the same
70/15/15 patient-level, gender-stratified split logic can run over either the
sample or full dataset instead of being duplicated per-notebook.

Age cleaning/parsing is intentionally NOT done here — preprocessing.py's
get_dataloaders() already parses the Y/M/D age suffix and drops age>100 outliers
at load time, so the split CSV just carries the raw metadata columns through.
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def make_patient_split(df, test_size=0.30, val_fraction=0.50, seed=42):
    """Assigns a 'Split' column (train/val/test) at the patient level, stratified
    by gender. Mirrors the full-dataset EDA split so results/leakage guarantees
    are consistent regardless of which metadata file is used."""
    patient_df = df.groupby('Patient ID').agg(Gender=('Patient Gender', 'first')).reset_index()

    train_ids, temp_ids = train_test_split(
        patient_df['Patient ID'], test_size=test_size,
        random_state=seed, stratify=patient_df['Gender'],
    )
    temp_df = patient_df[patient_df['Patient ID'].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df['Patient ID'], test_size=val_fraction,
        random_state=seed, stratify=temp_df['Gender'],
    )

    df = df.copy()
    df['Split'] = 'train'
    df.loc[df['Patient ID'].isin(val_ids), 'Split'] = 'val'
    df.loc[df['Patient ID'].isin(test_ids), 'Split'] = 'test'
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a patient-level data_split.csv")
    parser.add_argument("--csv_path", required=True, help="Raw metadata CSV (sample_labels.csv or Data_Entry_2017.csv)")
    parser.add_argument("--output", required=True, help="Path to write the resulting split CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df = make_patient_split(df, seed=args.seed)

    keep_cols = ['Image Index', 'Patient ID', 'Finding Labels', 'Patient Age',
                 'Patient Gender', 'View Position', 'Split']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(args.output, index=False)

    sets = {s: set(df[df['Split'] == s]['Patient ID']) for s in ['train', 'val', 'test']}
    print(f"Split sizes — "
          f"train: {len(df[df['Split']=='train']):,} imgs / {len(sets['train']):,} patients | "
          f"val: {len(df[df['Split']=='val']):,} imgs / {len(sets['val']):,} patients | "
          f"test: {len(df[df['Split']=='test']):,} imgs / {len(sets['test']):,} patients")

    overlaps = {
        'train/val': sets['train'] & sets['val'],
        'train/test': sets['train'] & sets['test'],
        'val/test': sets['val'] & sets['test'],
    }
    if any(overlaps.values()):
        for pair, overlap in overlaps.items():
            if overlap:
                print(f"[WARNING] {pair} patient overlap: {len(overlap)}")
    else:
        print("Verified patient-level isolation: zero overlap across splits.")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
