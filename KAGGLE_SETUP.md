# Running the full-dataset training on Kaggle

The full NIH release is ~42 GB and will not train in reasonable time on a laptop
(Junaid's overheated on the 5,606-image sample). Kaggle hosts the dataset publicly, so it
mounts read-only into the notebook — nothing to download or upload — and provides a free GPU.

## 1. Create the notebook

1. Go to kaggle.com → **Create** → **New Notebook**
2. Right sidebar → **Add Input** → search `NIH Chest X-rays` → add **`nih-chest-xrays/data`**
   (112,120 images, the official release)
3. Right sidebar → **Session options** → Accelerator: **GPU T4 x2** (or P100)
4. Session options → Internet: **On** (needed to clone the repo)

## 2. Confirm the mounted paths

Paths occasionally differ between dataset versions, so check before relying on them.

```python
!ls /kaggle/input/data/ | head
!ls /kaggle/input/data/images_001/images | head -3
!wc -l /kaggle/input/data/Data_Entry_2017.csv
```

Expect `Data_Entry_2017.csv` plus `images_001/` … `images_012/`, and 112,121 lines
(112,120 rows + header). If the folder is named differently, adjust `DATA` below.

## 3. Set up the code

```python
!git clone -b tianxin/simple-cnn-224 https://github.com/Tianxin1001/20D_AI4ALL.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q scikit-learn
```

## 4. Build the patient-level split

`--seed 42` matches every run done so far, so results stay comparable.

```python
DATA = '/kaggle/input/data'
!python make_data_split.py \
    --csv_path {DATA}/Data_Entry_2017.csv \
    --output /kaggle/working/data_split_full.csv \
    --seed 42
```

Expect roughly `train: 78,484 / val: 16,818 / test: 16,818` and
`Verified patient-level isolation: zero overlap across splits.`

## 5. Smoke test — one epoch first

Do not start the full run blind. One epoch confirms the recursive image index resolves the
nested `images_001/images/` layout, which is the failure mode that only appears on the full
dataset.

```python
!python train.py --model simple_cnn \
    --csv_path /kaggle/working/data_split_full.csv \
    --image_dir {DATA} \
    --img_size 224 --batch_size 64 --num_workers 4 \
    --epochs 1 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --predictions_dir /kaggle/working/predictions \
    --run_name smoke_full
```

Two lines to check:

- `Indexed 112,120 images across 12 directories under /kaggle/input/data` — the path fix working
- `Using device: cuda`

Note the reported epoch time before continuing.

## 6. Full run

```python
!python train.py --model simple_cnn \
    --csv_path /kaggle/working/data_split_full.csv \
    --image_dir {DATA} \
    --img_size 224 --batch_size 64 --lr 1e-4 \
    --dropout 0.2 --label_smoothing 0.1 \
    --epochs 15 --patience 4 --num_workers 4 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --predictions_dir /kaggle/working/predictions \
    --run_name simple_cnn_full
```

Roughly 15–25 minutes per epoch, so 4–6 hours for 15 epochs. A Kaggle session runs at most
12 hours and the weekly GPU quota is 30 hours, so this fits — but see the note below.

Differences from the sample runs, and why:

| Flag | Sample runs | Full run | Reason |
|---|---|---|---|
| `--batch_size` | 16 | 64 | GPU has the memory; fewer, larger steps are faster |
| `--num_workers` | 2 | 4 | Kaggle gives 4 CPUs; PNG decoding is the bottleneck |
| `--patience` | 15 (disabled) | 4 | Early stopping was disabled only to make the ablation curves comparable |

## 7. Download the outputs

Everything under `/kaggle/working/` is downloadable from the notebook's **Output** tab.
Three files matter:

| File | Why |
|---|---|
| `predictions/simple_cnn_full_test_predictions.csv` | **The fairness audit input** — per-image probabilities joined to Patient ID, sex, age, view position |
| `experiments/simple_cnn_full.csv` | Per-epoch metrics plus the final test row |
| `checkpoints/simple_cnn_full.pt` | Trained weights, for the Streamlit app and any re-analysis |

Commit the run log to `reports/`. The predictions CSV is large (~17,000 rows) and the
checkpoint is binary — keep both out of git and share via Drive.

## Notes

**If the session times out mid-run.** Checkpoints are written every time validation AUC
improves, so the best model up to that point survives in `/kaggle/working/checkpoints/`.
There is no resume-from-checkpoint flag in `train.py` yet; if this becomes a problem, lower
`--epochs` rather than risk losing the run.

**Keep the tab alive.** Kaggle stops interactive sessions after ~20 minutes of inactivity.
Either use **Save Version → Save & Run All (Commit)** to run it headless in the background,
which is the safer option for a multi-hour job, or leave the tab open and interact
occasionally.

**Quota.** GPU time is capped at 30 hours per week and a failed 5-hour run still consumes
quota, which is the real reason step 5 exists.

**If CheXNet gets re-approved.** Only `--model chexnet` changes; the pipeline is identical.
Budget 25–35 minutes per epoch instead, and drop to 12 epochs to stay inside one session.
