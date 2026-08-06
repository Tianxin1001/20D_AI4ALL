"""
Generates the presentation figures requested in the Week 11 check-in (July 30):

  fig1_label_counts.png       — how many images carry each label
  fig2_patient_structure.png  — unique patients vs. images, and images per patient
  fig3_label_consistency.png  — do repeat visits of the same patient carry the same labels?
  fig4_one_patient.png        — every X-ray from a single patient, side by side

Figures 1 and 2 use the full-dataset counts documented in
reports/data_quality_report.md (112,120 images / 30,805 patients), since the full
Data_Entry_2017.csv is not checked in. Figures 3 and 4 are computed from whichever
metadata CSV and image directory are passed in, and label which dataset they used.

Usage:
    python make_eda_figures.py \
        --csv_path nih_sample/sample/sample_labels.csv \
        --image_dir nih_sample/sample/images \
        --output_dir reports/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BLUE, GRAY, CORAL, TEAL, PLUM = "#378ADD", "#888780", "#D85A30", "#0F6E56", "#993556"

# Source: reports/data_quality_report.md (full dataset, 112,120 images)
FULL_LABEL_COUNTS = {
    "No Finding": 60361, "Infiltration": 19894, "Effusion": 13317, "Atelectasis": 11559,
    "Nodule": 6331, "Mass": 5782, "Pneumothorax": 5302, "Consolidation": 4667,
    "Pleural_Thickening": 3385, "Cardiomegaly": 2776, "Emphysema": 2516, "Edema": 2303,
    "Fibrosis": 1686, "Pneumonia": 1431, "Hernia": 227,
}
FULL_TOTAL_IMAGES, FULL_TOTAL_PATIENTS = 112120, 30805
FULL_MULTI_PATIENTS, FULL_MULTI_IMAGES, FULL_MAX_PER_PATIENT = 13302, 94617, 184


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.yaxis.label.set_color("#52514e")
    ax.xaxis.label.set_color("#52514e")


def fig1_label_counts(out):
    items = sorted(FULL_LABEL_COUNTS.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [GRAY if n == "No Finding" else BLUE for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, vals, color=colors, edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 900, f"{v:,}",
                ha="center", fontsize=8.5, color="#52514e")
    ax.set_yscale("log")
    ax.set_ylabel("Image count (log scale)")
    ax.set_title(f"Images per label — full dataset (n={FULL_TOTAL_IMAGES:,})\n"
                 f"Most common finding outnumbers the rarest by "
                 f"{FULL_LABEL_COUNTS['Infiltration'] // FULL_LABEL_COUNTS['Hernia']}×",
                 fontsize=12, color="#0b0b0b", loc="left")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {out}")


def fig2_patient_structure(df, out):
    per_patient = df["Patient ID"].value_counts()
    dist = per_patient.value_counts().sort_index().head(12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(["Images", "Unique patients"], [FULL_TOTAL_IMAGES, FULL_TOTAL_PATIENTS],
                color=[BLUE, TEAL], edgecolor="none", width=0.55)
    for i, v in enumerate([FULL_TOTAL_IMAGES, FULL_TOTAL_PATIENTS]):
        axes[0].text(i, v + 2500, f"{v:,}", ha="center", fontsize=11, color="#52514e")
    axes[0].set_ylim(0, FULL_TOTAL_IMAGES * 1.18)
    axes[0].set_title(f"{FULL_TOTAL_IMAGES / FULL_TOTAL_PATIENTS:.1f} images per patient on average\n"
                      f"{FULL_MULTI_IMAGES / FULL_TOTAL_IMAGES:.1%} of images come from repeat patients",
                      fontsize=11, color="#0b0b0b", loc="left")
    axes[0].set_ylabel("Count")
    style(axes[0])

    axes[1].bar(dist.index.astype(str), dist.values, color=PLUM, edgecolor="none")
    axes[1].set_xlabel("Number of images from the same patient")
    axes[1].set_ylabel("Number of patients")
    axes[1].set_title(f"Images per patient (sample dataset, max in full data = {FULL_MAX_PER_PATIENT})",
                      fontsize=11, color="#0b0b0b", loc="left")
    style(axes[1])

    fig.suptitle("Why the split must be by patient, not by image", fontsize=13,
                 color="#0b0b0b", x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {out}")


def fig3_label_consistency(df, out, dataset_name):
    multi = df.groupby("Patient ID").filter(lambda g: len(g) > 1)
    g = multi.groupby("Patient ID")["Finding Labels"]
    n_unique = g.nunique()
    identical = int((n_unique == 1).sum())
    changed = int((n_unique > 1).sum())

    healthy_then_sick = 0
    for _, labels in g:
        vals = list(labels)
        if any(v == "No Finding" for v in vals) and any(v != "No Finding" for v in vals):
            healthy_then_sick += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(["Same labels\nevery visit", "Labels change\nbetween visits"],
                [identical, changed], color=[TEAL, CORAL], edgecolor="none", width=0.5)
    total = identical + changed
    for i, v in enumerate([identical, changed]):
        axes[0].text(i, v + total * 0.02, f"{v:,}\n({v/total:.0%})",
                     ha="center", fontsize=10, color="#52514e")
    axes[0].set_ylim(0, total * 0.75)
    axes[0].set_ylabel("Patients with >1 image")
    axes[0].set_title(f"Labels are NOT fixed per patient\n"
                      f"{changed/total:.0%} of repeat patients change label between visits",
                      fontsize=11, color="#0b0b0b", loc="left")
    style(axes[0])

    axes[1].bar(["Crosses the\nhealthy/disease line"], [healthy_then_sick],
                color=PLUM, edgecolor="none", width=0.3)
    axes[1].text(0, healthy_then_sick + total * 0.02,
                 f"{healthy_then_sick:,}\n({healthy_then_sick/total:.0%} of repeat patients)",
                 ha="center", fontsize=10, color="#52514e")
    axes[1].set_xlim(-0.6, 0.6)
    axes[1].set_ylim(0, total * 0.75)
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Same patient recorded as both\n'No Finding' and diseased on different visits",
                      fontsize=11, color="#0b0b0b", loc="left")
    style(axes[1])

    fig.suptitle(f"Do repeat visits carry the same label?  ({dataset_name})",
                 fontsize=13, color="#0b0b0b", x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {out}")
    return identical, changed, healthy_then_sick


def fig4_one_patient(df, image_dir, out, dataset_name, max_images=6):
    counts = df["Patient ID"].value_counts()
    eligible = counts[(counts >= 4) & (counts <= max_images)]
    chosen = None
    for pid in eligible.index:
        rows = df[df["Patient ID"] == pid]
        if rows["Finding Labels"].nunique() > 1 and all(
                os.path.exists(os.path.join(image_dir, f)) for f in rows["Image Index"]):
            chosen = pid
            break
    if chosen is None:
        print("  [skip] fig4 — no suitable patient with multiple differing labels found")
        return None

    rows = df[df["Patient ID"] == chosen].sort_values("Image Index")
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, rows.iterrows()):
        ax.imshow(Image.open(os.path.join(image_dir, r["Image Index"])).convert("L"), cmap="gray")
        label = r["Finding Labels"].replace("|", "\n")
        ax.set_title(label, fontsize=8, color="#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c3c2b7")

    raw_age = str(rows["Patient Age"].iloc[0])
    m = pd.Series([raw_age]).str.extract(r"(\d+)\s*([YMD])?")
    years = int(m[0][0])
    unit = m[1][0]
    age = f"{years}" if unit in (None, "Y") or pd.isna(unit) else f"{years}{unit} old"
    sex = rows["Patient Gender"].iloc[0]
    fig.suptitle(f"All {n} X-rays from patient {chosen} ({sex}, age {age}) — {dataset_name}\n"
                 f"Same anatomy across every visit; labels differ. An image-level split "
                 f"would put these in both train and test.",
                 fontsize=11, color="#0b0b0b", x=0.008, ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return chosen


def main():
    p = argparse.ArgumentParser(description="Generate EDA figures for the final presentation")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", default="reports/figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    dataset_name = ("sample dataset, n=5,606" if len(df) < 20000
                    else f"full dataset, n={len(df):,}")

    print(f"Loaded {len(df):,} rows / {df['Patient ID'].nunique():,} patients from {args.csv_path}")
    print("Writing figures:")
    fig1_label_counts(os.path.join(args.output_dir, "fig1_label_counts.png"))
    fig2_patient_structure(df, os.path.join(args.output_dir, "fig2_patient_structure.png"))
    ident, chg, cross = fig3_label_consistency(
        df, os.path.join(args.output_dir, "fig3_label_consistency.png"), dataset_name)
    fig4_one_patient(df, args.image_dir,
                     os.path.join(args.output_dir, "fig4_one_patient.png"), dataset_name)

    print(f"\nLabel consistency ({dataset_name}):")
    print(f"  repeat patients with identical labels every visit: {ident:,}")
    print(f"  repeat patients whose labels change:               {chg:,}")
    print(f"  repeat patients crossing No Finding <-> disease:   {cross:,}")


if __name__ == "__main__":
    main()
