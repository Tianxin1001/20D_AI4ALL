# Chest X-ray Abnormality Detection: Making It Fair for Everyone

Trained a multi-label CNN to detect 14 thoracic findings across all 112,120 images of the NIH
ChestX-ray14 dataset, then audited whether it detects disease equally well for every patient —
measuring recall separately for each sex and age group rather than reporting a single accuracy
figure. Built during the AI4ALL Ignite accelerator.

## Problem Statement <!--- do not change this line -->

A model can be accurate overall and still fail a specific group of patients. If a hospital
deploys it, the same people are failed every time — and aggregate metrics make that invisible.
If a model detects pneumothorax in 89% of women aged 30-49 and 64% of men in the same age band,
a single reported AUC will never show it.

Chest radiography is the highest-volume imaging exam in medicine, and automated triage is
already being deployed. The people most likely to be harmed by an unevenly reliable model are
those already least well served by healthcare systems. So the question this project asks is not
"can we detect disease" but **"do we detect it equally, and if not, where does it fail?"**

We chose NIH ChestX-ray14 specifically because it ships patient sex and age. Without
demographics, there is no fairness audit.

## Key Results <!--- do not change this line -->

1. **Trained a from-scratch CNN to 0.7647 test mean AUC** across 14 findings on the full
   112,120-image dataset — with 1.6M parameters, versus 7M for the pretrained CheXNet benchmark
   (0.841 published).

2. **Found recall gaps of up to 26 percentage points between patient groups.** Four of seven
   reliably-measurable findings show a gap whose 95% confidence intervals do not overlap. The
   largest: Pneumothorax, an acute and time-critical finding, is caught in **89% of women aged
   30-49 but 64% of men in the same age band**.

3. **No group is uniformly underserved.** Men are recalled worse on Cardiomegaly and
   Pneumothorax; women are recalled worse on Mass and Nodule. Across age, Atelectasis improves
   with age while Cardiomegaly declines. The bias is finding-specific — a less quotable result
   than a single headline disparity, and the one the data supports.

4. **Improving the model closed two gaps, and revealed a third.** Comparing our two trained
   models is effectively an intervention study. The Cardiomegaly gap vanished once the model got
   better at the task, meaning it was a symptom of model weakness. Mass had *no measurable gap*
   in the weaker model only because that model flagged 68% of all images indiscriminately —
   **"no measurable gap" from a poor model is not reassurance.**

5. **Data volume mattered ~30× more than architecture tuning.** Moving from a 5,606-image sample
   to the full dataset gained +0.131 mean AUC. Three controlled ablations on input resolution
   and network depth gained between −0.014 and +0.004.

6. **Per-image analysis flatters performance on older patients.** Recomputing recall per patient
   rather than per image drops Infiltration for women 65+ from 0.798 to 0.575. Older patients
   contribute many images each, so image-level averaging is weighted towards them.

## Methodologies <!--- do not change this line -->

Every preprocessing decision traces to a specific finding in exploratory analysis:

| EDA finding | Consequence |
|---|---|
| 84.4% of images share a patient ID | Patient-level 70/15/15 split, verified zero overlap on every run |
| 88× class imbalance (Infiltration 19,894 vs Hernia 227) | `pos_weight` capped at 20, plus rare-class oversampling |
| Labels NLP-mined from reports, ~10% error rate | Label smoothing at 0.1 |
| Disease prevalence rises with age | The baseline the fairness results are read against |

**Multi-label, not multi-class.** A patient can have several findings at once, so the model
produces 14 independent sigmoid outputs trained with `BCEWithLogitsLoss` — never softmax, which
would force the findings to compete.

**Architecture** was chosen by controlled ablation rather than assumption. We isolated input
resolution from architecture to explain a performance gap between two models, found resolution
contributed nothing, and separately established that the network's global average pooling was
diluting small focal lesions. Adding max pooling alongside it gained +0.065 on small findings
against +0.036 elsewhere.

**The fairness audit** measures recall — not accuracy, which is meaningless when Hernia is 0.2%
of images, and not AUC, which measures ranking but never says how many patients were missed.
Two guard rails run before any number is quoted: findings are excluded unless every sex × age
cell holds at least 20 positive cases, and every recall is checked against what random flagging
at the same rate would achieve. Every estimate carries a Wilson confidence interval, and a gap is
only reported when intervals do not overlap.

## Data Sources <!--- do not change this line -->

[NIH Chest X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) —
112,120 frontal-view radiographs from 30,805 unique patients, with 14 disease labels plus patient
sex, age, and view position.

Labels were extracted from radiology reports using NLP, not verified by radiologists, with an
estimated ~10% error rate. This is documented in `reports/data_quality_report.md` and treated as
a limitation throughout rather than ignored.

## Technologies Used <!--- do not change this line -->

- Python, PyTorch, torchvision
- scikit-learn, pandas, NumPy
- Matplotlib
- Streamlit — interactive demo showing measured reliability per patient group
- Kaggle GPU notebooks — full-dataset training

## Repository guide

| | |
|---|---|
| `preprocessing.py` | Dataset, patient-level splits, class weighting, augmentation |
| `models.py` | `SimpleCNN` (configurable depth, width, pooling) and `CheXNet` |
| `train.py` | Training loop, early stopping, per-image prediction export |
| `make_data_split.py` | Patient-level, sex-stratified 70/15/15 split |
| `fairness_eval.ipynb` | **The fairness audit** |
| `make_eda_figures.py` | Presentation figures |
| `streamlit_app.py` | Demo app |
| `KAGGLE_SETUP.md` | Reproducing the full-dataset run |

Reports, in reading order:

1. `reports/data_quality_report.md` — what the data actually looks like
2. `reports/full_dataset_run_report.md` — training the model
3. `reports/fairness_audit_report.md` — **the main result**
4. `reports/resolution_ablation_report.md`, `reports/depth_ablation_report.md` — architecture
   experiments, including a negative result and a correction to an earlier over-reading of it

## Limitations

This model is **not clinically usable**. Precision is low on most findings, labels are
NLP-derived, and view position (AP films are taken bedside on sicker patients) is a confound we
identified but did not remove. The audit measures relative disparity between groups, not fitness
for deployment. Sex is recorded binary in this dataset and the audit inherits that limitation.

## Authors <!--- do not change this line -->

This project was completed in collaboration with:

- Tianxin Dong ([Tianxin1001](https://github.com/Tianxin1001))
- Krisha Rathod ([Krisha052](https://github.com/Krisha052))
- Junaid Pathan ([junaid-pathan](https://github.com/junaid-pathan))
- Belema Roberts ([RBelex-007](https://github.com/RBelex-007))
