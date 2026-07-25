# Data Quality Report — NIH Chest X-ray Full Dataset
**AI4ALL Ignite Project | Data & EDA Lead**

---

## Dataset Overview

| Metric | Value |
|---|---|
| Total images | 112,120 |
| Unique patients | 30,805 |
| Disease labels | 15 |
| Overall mean age | 46.9 years |
| Gender split | Male 56.5% / Female 43.5% |
| View position | PA 60.0% / AP 40.0% |

---

## 1. Age Errors (Cleaned)

**16 records** contain impossible ages over 100 years. These are data entry errors inherited from the source hospital records — not real patients.

| Patient ID | Age | Finding |
|---|---|---|
| 5567 | 412 | Effusion, Pneumonia |
| 11973 | 414 | Edema |
| 12238 | 148 | No Finding |
| 13950 | 148 | No Finding |
| 14520 | 150 | Infiltration, Mass |
| 15558 | 149 | No Finding |
| 18366 | 152 | Pneumothorax |
| 19346 | 151 | Infiltration |
| 20900 | 411 | No Finding |
| 21047 | 412 | Mass, Pleural Thickening |
| 21275 | 413 | No Finding |
| 22811 | 412 | No Finding |
| 25206 | 153 | Infiltration, Mass |
| 26028 | 154 | Atelectasis |
| 26871 | 155 | No Finding |
| 27989 | 155 | No Finding |

**Action taken:** All 16 records have `Age_clean` set to `NaN`. Images are retained (labels are still valid) but excluded from any age-based analysis or age-stratified evaluation.

**Note on the sample dataset:** The sample `sample_labels.csv` stores age with unit suffixes (e.g. `060Y`, `013M`, `001D`). The full `Data_Entry_2017.csv` stores age as a plain integer — no infant unit-suffix ambiguity exists in the full dataset. Minimum clean age is 1 year.

---

## 2. Label Quality

- Labels are **NLP-mined from radiology reports**, not radiologist-verified. Estimated error rate ~10%.
- **Pneumothorax** is a known high-noise label: frequently captures already-treated cases where a chest drain is visible rather than the acute condition.
- `No Finding` co-occurring with another label: **0 rows** — integrity check passed.
- Duplicate Image Index rows: **0** — no duplicate images.

---

## 3. Class Imbalance

Severe imbalance across all 15 labels:

| Label | Count | % of images |
|---|---|---|
| No Finding | 60,361 | 53.8% |
| Infiltration | 19,894 | 17.7% |
| Effusion | 13,317 | 11.9% |
| Atelectasis | 11,559 | 10.3% |
| Nodule | 6,331 | 5.6% |
| Mass | 5,782 | 5.2% |
| Pneumothorax | 5,302 | 4.7% |
| Consolidation | 4,667 | 4.2% |
| Pleural_Thickening | 3,385 | 3.0% |
| Emphysema | 2,516 | 2.2% |
| Cardiomegaly | 2,776 | 2.5% |
| Edema | 2,303 | 2.1% |
| Fibrosis | 1,686 | 1.5% |
| Pneumonia | 1,431 | 1.3% |
| Hernia | 227 | 0.2% |

**Hernia (227 images)** remains the rarest class even in the full dataset. Class weighting and oversampling are necessary but model reliability on Hernia should be explicitly disclaimed.

---

## 4. Age-correlated Prevalence

Clear age gradient confirmed across labels (overall mean: 46.9 years):

| Label | Mean Age | Diff from Mean |
|---|---|---|
| Hernia | 63.2 | +16.3 |
| Fibrosis | 52.7 | +5.8 |
| Atelectasis | 50.5 | +3.6 |
| Pleural_Thickening | 50.5 | +3.6 |
| Emphysema | 50.3 | +3.4 |
| Effusion | 49.8 | +2.9 |
| Nodule | 49.5 | +2.6 |
| Mass | 48.8 | +1.9 |
| Cardiomegaly | 47.3 | +0.4 |
| Consolidation | 46.7 | -0.2 |
| Pneumothorax | 46.5 | -0.4 |
| Infiltration | 46.2 | -0.7 |
| No Finding | 45.8 | -1.1 |
| Edema | 45.6 | -1.3 |
| Pneumonia | 44.9 | -2.0 |

**Mitigation:** Age-stratified train/test splits implemented (see Section 6).

---

## 5. Patient Leakage Risk

| Metric | Value |
|---|---|
| Patients with >1 image | 13,302 / 30,805 (43.2%) |
| Images from multi-image patients | 94,617 / 112,120 (**84.4%**) |
| Max images per patient | 184 |

**84.4% of images share a patient ID with at least one other image.** Splitting by image rather than patient ID would cause severe data leakage — the model would see the same patient in both training and test sets.

**Mitigation:** Patient-level split applied (see Section 6). Zero patient overlap verified across all three splits.

---

## 6. Train / Validation / Test Split

**Method:** Patient-level stratified split, stratified by gender to maintain sex balance.

| Split | Images | Patients | % |
|---|---|---|---|
| Train | ~78,484 | ~21,564 | 70% |
| Validation | ~16,818 | ~4,621 | 15% |
| Test | ~16,818 | ~4,620 | 15% |

- Patient overlap between splits: **0** (verified)
- Gender balance maintained across all three splits
- Output saved to `data_split.csv`

---

## 7. Summary of Actions Taken

| Issue | Action |
|---|---|
| 16 impossible age values (>100 yrs) | Set `Age_clean` to NaN; images retained |
| NLP label noise (~10% error rate) | Flagged; label smoothing planned during training |
| 84.4% images share a patient ID | Patient-level 70/15/15 split applied |
| Severe class imbalance (Hernia: 227) | Class weighting to be applied during training |
| Age-correlated prevalence | Age-stratified evaluation planned |
| View position confound (AP vs PA) | Flagged for analysis; not removed |
