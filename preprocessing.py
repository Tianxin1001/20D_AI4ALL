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

# Standard NIH 14 Chest X-ray classes
NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# ==========================================
# 1. Robust Data Mocking (For Testing)
# ==========================================
def generate_mock_data(output_dir="mock_data", num_samples=150):
    """
    Generates a mock dataset for immediate pipeline testing.
    Creates a folder of mock images (random noise) and a metadata CSV.
    """
    set_seed(42)
    output_path = pathlib.Path(output_dir)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} mock 2D images in {image_dir}...")
    for i in range(num_samples):
        img_name = f"{i:08d}_000.png"
        img_path = image_dir / img_name
        
        # Tiny 128x128 grayscale noise images to save space/time
        img_arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(img_path)
        
    csv_path = output_path / "mock_metadata.csv"
    
    # Realistic Patient IDs to simulate patient stratification
    patient_ids = [random.randint(10000, 99999) for _ in range(30)]
    
    # Generate labels, making "Hernia" extremely rare and "No Finding" very common
    findings_pool = []
    for _ in range(num_samples):
        r = random.random()
        if r < 0.55:
            findings_pool.append("No Finding")
        elif r < 0.85:
            # Single finding, Hernia has very low probability (0.02 vs 1.0)
            weights = [1.0] * len(NIH_CLASSES)
            weights[NIH_CLASSES.index('Hernia')] = 0.02
            selected = random.choices(NIH_CLASSES, weights=weights, k=1)
            findings_pool.append(selected[0])
        else:
            # Multiple findings, Hernia has very low probability
            weights = [1.0] * len(NIH_CLASSES)
            weights[NIH_CLASSES.index('Hernia')] = 0.02
            selected = random.choices(NIH_CLASSES, weights=weights, k=2)
            selected = list(set(selected))  # Ensure unique findings
            findings_pool.append("|".join(selected))
            
    # Assign splits: 70% train, 15% val, 15% test
    splits = ['train'] * int(num_samples * 0.7) + \
             ['val'] * int(num_samples * 0.15) + \
             ['test'] * (num_samples - int(num_samples * 0.7) - int(num_samples * 0.15))
    random.shuffle(splits)
    
    # Force a few Hernia samples in the train split to verify oversampling
    findings_pool[0] = "Hernia"
    splits[0] = "train"
    findings_pool[1] = "Hernia"
    splits[1] = "train"
    
    data = []
    for i in range(num_samples):
        img_name = f"{i:08d}_000.png"
        patient_id = random.choice(patient_ids)
        finding = findings_pool[i]
        split = splits[i]
        
        data.append({
            'Image Index': img_name,
            'Finding Labels': finding,
            'Patient ID': patient_id,
            'Split': split
        })
        
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Mock metadata CSV saved to {csv_path}\n")
    return csv_path, image_dir

# ==========================================
# 2. Custom Dataset Class
# ==========================================
class NIHChestXrayDataset(Dataset):
    """
    A clean, modular PyTorch Dataset class for the NIH Chest X-ray dataset.
    Parses multi-label findings separated by '|' and returns multi-hot encoded labels.
    """
    def __init__(self, dataframe, image_dir, classes=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = pathlib.Path(image_dir)
        self.classes = classes if classes is not None else NIH_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform
        
        # Pre-parse and store targets for sampler/weights calculation
        self.labels = []
        for _, row in self.df.iterrows():
            label_vec = np.zeros(len(self.classes), dtype=np.float32)
            findings_str = str(row['Finding Labels'])
            if findings_str and findings_str != 'No Finding':
                findings = [f.strip() for f in findings_str.split('|')]
                for finding in findings:
                    if finding in self.class_to_idx:
                        label_vec[self.class_to_idx[finding]] = 1.0
            self.labels.append(label_vec)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        image_path = self.image_dir / image_name
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Convert to RGB (standard for pre-trained CNNs)
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
    - Train Pipeline: Restricted to clinically plausible augmentations.
    - Val/Test Pipeline: Resizing and normalization only.
    """
    # Clinically plausible augmentations for train set
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Mild rotation (±10 deg) is clinically plausible for minor patient alignment shifts
        transforms.RandomRotation(degrees=10),
        # Horizontal flip (doubles patient orientation variability, though left-right is anatomical)
        transforms.RandomHorizontalFlip(p=0.5),
        # Brightness/Contrast adjustments simulate varying exposure/scanner calibration
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        # Standard ImageNet normalization (compatible with pre-trained backbones)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Standard resize and normalize for validation/test sets
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
    Computes class weights dynamically for BCEWithLogitsLoss to handle imbalance.
    pos_weight = (total_negatives + eps) / (total_positives + eps) for each class.
    """
    labels = dataset.labels
    total_samples = len(labels)
    
    # Compute active counts per class
    pos_counts = labels.sum(axis=0)
    neg_counts = total_samples - pos_counts
    
    # Handle numerical stability (if a class is missing in this split)
    eps = 1e-5
    pos_weights = (neg_counts + eps) / (pos_counts + eps)
    
    return torch.tensor(pos_weights, dtype=torch.float32)

def get_multilabel_sampler(dataset, baseline_weight=0.1):
    """
    Constructs a WeightedRandomSampler to oversample rare classes in a multi-label setting.
    We compute sample weights based on the sum of active class rarity weights.
    If a sample has 'No Finding', it is assigned a small baseline weight.
    """
    labels = dataset.labels
    class_counts = labels.sum(axis=0)
    
    # Class rarity is inversely proportional to class frequency
    class_rarity = 1.0 / (class_counts + 1.0)
    
    sample_weights = []
    for label_vec in labels:
        active_indices = np.where(label_vec == 1.0)[0]
        if len(active_indices) > 0:
            # Sum the rarity scores of all positive findings in this image
            weight = class_rarity[active_indices].sum()
        else:
            # Baseline weight for images with 'No Finding'
            weight = baseline_weight * class_rarity.min()
        sample_weights.append(weight)
        
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# ==========================================
# 5. Flexible DataLoaders
# ==========================================
def get_dataloaders(csv_path, image_dir, batch_size=16, img_size=224, use_oversampling=True, num_workers=0):
    """
    Creates and returns Train, Val, and Test DataLoaders matching the splits.
    Also returns the class weights computed from the training split.
    """
    df = pd.read_csv(csv_path)
    
    if 'Split' not in df.columns:
        raise ValueError("CSV metadata must contain a 'Split' column.")
        
    train_df = df[df['Split'] == 'train'].copy()
    val_df = df[df['Split'] == 'val'].copy()
    test_df = df[df['Split'] == 'test'].copy()
    
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = NIHChestXrayDataset(train_df, image_dir, transform=train_transform)
    val_dataset = NIHChestXrayDataset(val_df, image_dir, transform=val_transform)
    test_dataset = NIHChestXrayDataset(test_df, image_dir, transform=val_transform)
    
    # Calculate loss weights on training data only to avoid validation leakage
    class_weights = compute_class_weights(train_dataset)
    
    # Configure Loader Arguments
    train_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False
    }
    
    if use_oversampling and len(train_dataset) > 0:
        sampler = get_multilabel_sampler(train_dataset)
        train_kwargs['sampler'] = sampler
        train_kwargs['shuffle'] = False  # Sampler and shuffle are mutually exclusive
    else:
        train_kwargs['shuffle'] = True
        
    train_loader = DataLoader(train_dataset, **train_kwargs)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, class_weights

# ==========================================
# 6. Quick Verification Main Block
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("NIH Chest X-ray Preprocessing Pipeline Verification")
    print("==================================================")
    
    # Set seed
    set_seed(42)
    
    # Setup paths
    mock_dir = "mock_data"
    
    # 1. Generate Mock Data
    csv_path, image_dir = generate_mock_data(output_dir=mock_dir, num_samples=150)
    
    # 2. Instantiate Loaders with and without oversampling to verify behaviors
    batch_size = 16
    
    print("--- Loading WITHOUT oversampling ---")
    train_loader_no_os, val_loader, test_loader, weights_no_os = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        batch_size=batch_size,
        use_oversampling=False
    )
    
    print("\n--- Loading WITH oversampling ---")
    train_loader_os, _, _, weights_os = get_dataloaders(
        csv_path=csv_path,
        image_dir=image_dir,
        batch_size=batch_size,
        use_oversampling=True
    )
    
    # Check a single batch
    images, labels = next(iter(train_loader_os))
    
    print("\nBatch Verification:")
    print(f"  Batch Image Tensor Shape : {images.shape} (Expected: [batch_size, 3, 224, 224])")
    print(f"  Batch Label Tensor Shape : {labels.shape} (Expected: [batch_size, 14])")
    print(f"  Batch Image Value Range  : Min={images.min().item():.3f}, Max={images.max().item():.3f}")
    
    # Show Class Weight details
    print("\nComputed BCE pos_weight per class:")
    for cls, wt in zip(NIH_CLASSES, weights_os.numpy()):
        print(f"  {cls:<20}: {wt:.3f}")
        
    # Verify rare class frequency amplification under oversampling
    print("\nRare Class Amplification Verification:")
    # Retrieve all targets in train dataset
    train_ds = train_loader_os.dataset
    raw_class_counts = train_ds.labels.sum(axis=0)
    
    # Count frequency in a large sample of batches to verify oversampler
    sampled_labels = []
    # Sample 20 batches (with replacement) to get robust counts
    for _ in range(20):
        for _, lbls in train_loader_os:
            sampled_labels.append(lbls.numpy())
            if len(sampled_labels) * batch_size >= 1000:
                break
        if len(sampled_labels) * batch_size >= 1000:
            break
            
    sampled_labels = np.concatenate(sampled_labels, axis=0)
    oversampled_counts = sampled_labels.sum(axis=0)
    
    # Calculate percentage frequency
    raw_freq = raw_class_counts / len(train_ds)
    os_freq = oversampled_counts / len(sampled_labels)
    
    print(f"  Total train dataset size: {len(train_ds)}")
    print(f"  Oversampled validation sample size: {len(sampled_labels)}")
    print(f"\n  Comparison (Raw Dataset Freq % vs. Oversampled Freq %):")
    print(f"  {'Class':<20} | {'Raw Freq %':<12} | {'Oversampled %':<15} | {'Relative Increase':<18}")
    print("-" * 75)
    for i, cls in enumerate(NIH_CLASSES):
        rel_increase = (os_freq[i] / (raw_freq[i] + 1e-5)) if raw_freq[i] > 0 else 0
        print(f"  {cls:<20} | {raw_freq[i]*100:>10.2f}% | {os_freq[i]*100:>13.2f}% | {rel_increase:>15.2f}x")
        
    print("\nPipeline run successful! Ready for production integration.")
