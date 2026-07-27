import os
import random
import pathlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Standard NIH 14 Chest X-ray disease classes.
# 'No Finding' is intentionally excluded — it is represented as an all-zero label vector.
# EDA confirmed 15 total labels (14 diseases + No Finding) across 112,120 images.
NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
assert len(NIH_CLASSES) == 14, "Expected 14 NIH disease classes"


# ==========================================
# 1. Robust Data Mocking (For Testing)
# ==========================================
def generate_mock_data(output_dir="mock_data", num_samples=150):
    """
    Generates a mock dataset for immediate pipeline testing.
    Creates a folder of mock images (random noise) and a metadata CSV.

    EDA fix: splits are now assigned at patient level, not image level.
    EDA finding: 84.4% of images in the full dataset share a patient ID —
    splitting by image causes severe data leakage.
    """
    set_seed(42)
    output_path = pathlib.Path(output_dir)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} mock 2D images in {image_dir}...")
    for i in range(num_samples):
        img_name = f"{i:08d}_000.png"
        img_path = image_dir / img_name
        img_arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(img_path)

    csv_path = output_path / "mock_metadata.csv"

    # Assign images to patients first — each patient gets 1-5 images
    # This mirrors the real dataset where patients have multiple follow-up scans
    patient_ids = list(range(10000, 10040))  # 40 unique patients
    image_to_patient = []
    for i in range(num_samples):
        image_to_patient.append(random.choice(patient_ids))

    # --- Patient-level split (EDA fix) ---
    # Assign the split to each unique patient, then propagate to their images.
    # This guarantees zero patient overlap across train/val/test.
    unique_patients = list(set(image_to_patient))
    random.shuffle(unique_patients)
    n_train = int(len(unique_patients) * 0.70)
    n_val   = int(len(unique_patients) * 0.15)

    patient_split_map = {}
    for i, pid in enumerate(unique_patients):
        if i < n_train:
            patient_split_map[pid] = 'train'
        elif i < n_train + n_val:
            patient_split_map[pid] = 'val'
        else:
            patient_split_map[pid] = 'test'

    # Generate labels: "No Finding" very common, Hernia extremely rare
    findings_pool = []
    for _ in range(num_samples):
        r = random.random()
        if r < 0.55:
            findings_pool.append("No Finding")
        elif r < 0.85:
            weights = [1.0] * len(NIH_CLASSES)
            weights[NIH_CLASSES.index('Hernia')] = 0.02
            selected = random.choices(NIH_CLASSES, weights=weights, k=1)
            findings_pool.append(selected[0])
        else:
            weights = [1.0] * len(NIH_CLASSES)
            weights[NIH_CLASSES.index('Hernia')] = 0.02
            selected = list(set(random.choices(NIH_CLASSES, weights=weights, k=2)))
            findings_pool.append("|".join(selected))

    # Force a few Hernia samples into train to verify oversampling
    findings_pool[0] = "Hernia"
    image_to_patient[0] = unique_patients[0]   # ensure in train
    findings_pool[1] = "Hernia"
    image_to_patient[1] = unique_patients[0]

    data = []
    for i in range(num_samples):
        pid = image_to_patient[i]
        data.append({
            'Image Index':    f"{i:08d}_000.png",
            'Finding Labels': findings_pool[i],
            'Patient ID':     pid,
            'Patient Age':    random.randint(20, 80),
            'Patient Gender': random.choice(['M', 'F']),
            'View Position':  random.choice(['PA', 'AP']),
            'Split':          patient_split_map[pid]
        })

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Mock metadata CSV saved to {csv_path}")
    print(f"Patient-level split: "
          f"train={sum(v=='train' for v in patient_split_map.values())} patients, "
          f"val={sum(v=='val' for v in patient_split_map.values())} patients, "
          f"test={sum(v=='test' for v in patient_split_map.values())} patients\n")
    return csv_path, image_dir


# ==========================================
# 2. Custom Dataset Class
# ==========================================
class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the NIH Chest X-ray dataset.
    Parses multi-label findings separated by '|' and returns multi-hot encoded labels.

    EDA addition: label_smoothing parameter to mitigate NLP-mined label noise.
    EDA finding: labels were extracted via NLP from radiology reports with an
    estimated ~10% error rate. Label smoothing reduces overconfidence on noisy labels.
    Recommended value: 0.1 (i.e. positive targets become 0.9 instead of 1.0).
    """
    def __init__(self, dataframe, image_dir, classes=None, transform=None, label_smoothing=0.0):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = pathlib.Path(image_dir)
        self.classes = classes if classes is not None else NIH_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform
        # label_smoothing=0.1 recommended based on EDA estimated ~10% NLP error rate
        self.label_smoothing = label_smoothing

        self.labels = []
        for _, row in self.df.iterrows():
            label_vec = np.zeros(len(self.classes), dtype=np.float32)
            findings_str = str(row['Finding Labels'])
            if findings_str and findings_str != 'No Finding':
                findings = [f.strip() for f in findings_str.split('|')]
                for finding in findings:
                    if finding in self.class_to_idx:
                        # Apply label smoothing: positive target = 1.0 - smoothing
                        # This prevents the model from being overconfident on noisy NLP labels
                        label_vec[self.class_to_idx[finding]] = 1.0 - self.label_smoothing
            self.labels.append(label_vec)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['Image Index']

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# ==========================================
# 3. Clinically Plausible Augmentations
# ==========================================
def get_transforms(img_size=224):
    """
    Returns torchvision transform pipelines.

    EDA fix: horizontal flip removed from training transforms.
    EDA finding: chest X-rays are anatomically asymmetric — the heart sits left,
    liver shadow sits right, and aortic knob is left-sided. Flipping at p=0.5
    trains the model on anatomically impossible images, which can hurt performance
    on laterality-dependent findings (pneumothorax, effusion, cardiomegaly).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Mild rotation (±10 deg): plausible for minor patient positioning shifts
        transforms.RandomRotation(degrees=10),
        # Horizontal flip REMOVED — chest X-rays are anatomically asymmetric.
        # Flipping creates impossible anatomy and harms laterality-dependent findings.
        # Original p=0.5 was too aggressive; removed entirely per EDA recommendation.
        # Brightness/contrast: simulates varying X-ray exposure and scanner calibration
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        # Standard ImageNet normalisation (compatible with DenseNet-121 / ResNet backbones)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ==========================================
# 4. Class Weighting & Oversampling
# ==========================================
def compute_class_weights(dataset):
    """
    Computes pos_weight per class for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = (total_negatives + eps) / (total_positives + eps)

    EDA context: label counts range from Infiltration (19,894) to Hernia (227).
    This ratio-based weighting ensures rare classes like Hernia receive higher loss weight.
    """
    labels = dataset.labels
    total_samples = len(labels)
    pos_counts = labels.sum(axis=0)
    neg_counts = total_samples - pos_counts
    eps = 1e-5
    pos_weights = (neg_counts + eps) / (pos_counts + eps)
    return torch.tensor(pos_weights, dtype=torch.float32)


def get_multilabel_sampler(dataset, baseline_weight=0.1):
    """
    WeightedRandomSampler to oversample rare classes in multi-label setting.
    Sample weight = sum of rarity scores of all positive findings in the image.
    'No Finding' images receive a small baseline weight.

    EDA context: Hernia has only 227 images in the full dataset (0.2% of images).
    Note: oversampling alone cannot overcome fundamental data scarcity for Hernia —
    model reliability on Hernia should be explicitly disclaimed regardless.
    """
    labels = dataset.labels
    class_counts = labels.sum(axis=0)
    class_rarity = 1.0 / (class_counts + 1.0)

    sample_weights = []
    for label_vec in labels:
        active_indices = np.where(label_vec > 0.0)[0]  # works with smoothed labels too
        if len(active_indices) > 0:
            weight = class_rarity[active_indices].sum()
        else:
            weight = baseline_weight * class_rarity.min()
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ==========================================
# 5. Flexible DataLoaders
# ==========================================
def get_dataloaders(csv_path, image_dir, batch_size=16, img_size=224,
                    use_oversampling=True, label_smoothing=0.1, num_workers=0):
    """
    Creates Train, Val, and Test DataLoaders from a pre-split CSV.

    EDA fix 1: expects 'Split' column from data_split.csv produced by the EDA notebook.
    The EDA already performed a verified patient-level 70/15/15 split with zero
    patient overlap — do NOT re-split here.

    EDA fix 2: filters out the 16 records with impossible ages (>100 years) identified
    in the EDA. These are data entry errors; images are removed to keep metadata clean.

    EDA addition: label_smoothing parameter passed through to NIHChestXrayDataset.
    Recommended value: 0.1, based on estimated ~10% NLP label error rate.

    EDA note: View Position (PA/AP) is retained in the dataframe but not passed
    to the model. Keep it for post-hoc per-view evaluation — AP images correlate
    with sicker patients and may show different performance characteristics.
    """
    df = pd.read_csv(csv_path)

    if 'Split' not in df.columns:
        raise ValueError(
            "CSV must contain a 'Split' column. "
            "Use data_split.csv generated by the EDA notebook (patient-level split)."
        )

    # --- Parse Patient Age: raw CSV stores it as text with a Y/M/D unit
    # suffix (e.g. "058Y", "013M", "007D"). Convert months/days to a
    # year-equivalent so infants aren't misread as N years old.
    if 'Patient Age' in df.columns:
        age_parts = df['Patient Age'].astype(str).str.extract(r'(\d+)([YMD])')
        age_value = pd.to_numeric(age_parts[0], errors='coerce')
        age_unit = age_parts[1]
        df['Patient Age'] = np.select(
            [age_unit == 'Y', age_unit == 'M', age_unit == 'D'],
            [age_value, age_value / 12, age_value / 365],
            default=age_value,
        )

    # --- EDA fix: remove age outliers (16 records with age > 100) ---
    if 'Patient Age' in df.columns:
        n_before = len(df)
        df = df[df['Patient Age'] <= 100].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"Removed {n_removed} records with impossible age > 100 yrs "
                  f"(EDA identified 16 such records in the full dataset).")

    # Log View Position distribution for post-hoc analysis (not fed into model)
    if 'View Position' in df.columns:
        vp = df['View Position'].value_counts()
        print(f"View position breakdown — PA: {vp.get('PA', 0):,} | "
              f"AP: {vp.get('AP', 0):,}  "
              f"(AP correlates with patient severity — track per-view eval)")

    train_df = df[df['Split'] == 'train'].copy()
    val_df   = df[df['Split'] == 'val'].copy()
    test_df  = df[df['Split'] == 'test'].copy()

    print(f"\nSplit sizes — train: {len(train_df):,} | "
          f"val: {len(val_df):,} | test: {len(test_df):,}")

    train_transform, val_transform = get_transforms(img_size)

    train_dataset = NIHChestXrayDataset(
        train_df, image_dir, transform=train_transform, label_smoothing=label_smoothing)
    val_dataset = NIHChestXrayDataset(
        val_df, image_dir, transform=val_transform, label_smoothing=0.0)
    test_dataset = NIHChestXrayDataset(
        test_df, image_dir, transform=val_transform, label_smoothing=0.0)

    # Class weights computed on training data only — avoids validation leakage
    class_weights = compute_class_weights(train_dataset)

    train_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available()
    }

    if use_oversampling and len(train_dataset) > 0:
        train_kwargs['sampler'] = get_multilabel_sampler(train_dataset)
        train_kwargs['shuffle'] = False  # mutually exclusive with sampler
    else:
        train_kwargs['shuffle'] = True

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, class_weights


# ==========================================
# 6. Quick Verification Main Block
# ==========================================
if __name__ == "__main__":
    print("=" * 52)
    print("NIH Chest X-ray Preprocessing Pipeline Verification")
    print("=" * 52)

    set_seed(42)
    mock_dir = "mock_data"

    # 1. Generate mock data with patient-level splits
    csv_path, image_dir = generate_mock_data(output_dir=mock_dir, num_samples=150)

    # 2. Load without oversampling
    print("--- Loading WITHOUT oversampling ---")
    train_loader_no_os, val_loader, test_loader, weights_no_os = get_dataloaders(
        csv_path=csv_path, image_dir=image_dir,
        batch_size=16, use_oversampling=False, label_smoothing=0.1
    )

    # 3. Load with oversampling
    print("\n--- Loading WITH oversampling ---")
    train_loader_os, _, _, weights_os = get_dataloaders(
        csv_path=csv_path, image_dir=image_dir,
        batch_size=16, use_oversampling=True, label_smoothing=0.1
    )

    # 4. Batch verification
    images, labels = next(iter(train_loader_os))
    print("\nBatch Verification:")
    print(f"  Image tensor shape : {images.shape}  (expected [batch, 3, 224, 224])")
    print(f"  Label tensor shape : {labels.shape}  (expected [batch, 14])")
    print(f"  Image value range  : min={images.min():.3f}, max={images.max():.3f}")
    print(f"  Label value range  : min={labels.min():.3f}, max={labels.max():.3f}  "
          f"(max should be {1.0-0.1:.1f} with label_smoothing=0.1)")

    # 5. Class weights
    print("\nComputed BCE pos_weight per class:")
    for cls, wt in zip(NIH_CLASSES, weights_os.numpy()):
        print(f"  {cls:<20}: {wt:.3f}")

    # 6. Rare class amplification check
    print("\nRare Class Amplification Verification:")
    train_ds = train_loader_os.dataset
    raw_class_counts = train_ds.labels.sum(axis=0)

    sampled_labels = []
    for _ in range(20):
        for _, lbls in train_loader_os:
            sampled_labels.append(lbls.numpy())
            if len(sampled_labels) * 16 >= 1000:
                break
        if len(sampled_labels) * 16 >= 1000:
            break

    sampled_labels = np.concatenate(sampled_labels, axis=0)
    oversampled_counts = sampled_labels.sum(axis=0)
    raw_freq = raw_class_counts / len(train_ds)
    os_freq  = oversampled_counts / len(sampled_labels)

    print(f"  Train dataset size:        {len(train_ds)}")
    print(f"  Oversampled sample size:   {len(sampled_labels)}")
    print(f"\n  {'Class':<20} | {'Raw %':>8} | {'Oversampled %':>14} | {'Increase':>10}")
    print("  " + "-" * 62)
    for i, cls in enumerate(NIH_CLASSES):
        rel = (os_freq[i] / (raw_freq[i] + 1e-5)) if raw_freq[i] > 0 else 0
        print(f"  {cls:<20} | {raw_freq[i]*100:>7.2f}% | {os_freq[i]*100:>13.2f}% | {rel:>8.2f}x")

    # 7. Patient leakage verification
    print("\nPatient Leakage Verification:")
    df_check = pd.read_csv(csv_path)
    for s1, s2 in [('train','val'), ('train','test'), ('val','test')]:
        p1 = set(df_check[df_check['Split']==s1]['Patient ID'])
        p2 = set(df_check[df_check['Split']==s2]['Patient ID'])
        overlap = len(p1 & p2)
        status = '✓' if overlap == 0 else '✗ LEAKAGE'
        print(f"  {s1} ∩ {s2}: {overlap} patients  {status}")

    print("\nPipeline verification complete.")
