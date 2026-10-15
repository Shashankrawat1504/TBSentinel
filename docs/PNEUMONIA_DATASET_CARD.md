# RSNA Pneumonia Dataset Card

## Dataset Overview

The RSNA Pneumonia Detection Challenge dataset contains chest X-ray images with labels for pneumonia detection. This dataset has been integrated into TBSentinel to enable multi-disease classification (Normal / Tuberculosis / Pneumonia).

## Dataset Details

- **Source**: RSNA Pneumonia Detection Challenge (Kaggle)
- **Image Modality**: Chest X-ray (Digital Radiography)
- **Original Classes**:
  - Normal: Healthy chest X-rays
  - Lung Opacity: Pneumonia cases
  - No Lung Opacity / Not Normal: Ambiguous or other conditions (excluded)

## Dataset Structure

```
data/raw/Pneumonia/chest x ray/
├── Training/
│   ├── Images/          # Training images (PNG)
│   └── Masks/           # Segmentation masks (not used)
├── Test/                # Test images (PNG)
├── stage2_train_metadata.csv  # Training metadata
└── stage2_test_metadata.csv   # Test metadata
```

## Metadata Fields

- `patientId`: Unique patient identifier
- `x`, `y`, `width`, `height`: Bounding box coordinates (for detection task)
- `Target`: Binary label (0 = Normal/No Opacity, 1 = Lung Opacity)
- `class`: Class name (Normal, Lung Opacity, No Lung Opacity / Not Normal)
- `age`: Patient age
- `sex`: Patient sex (M/F)
- `modality`: Imaging modality (CR = Computed Radiography)
- `position`: View position (PA = Posteroanterior, AP = Anteroposterior)

## Processing Pipeline

### 1. Data Cleaning

- Loaded from `stage2_train_metadata.csv`
- Deduplicated by patient ID (26,684 unique patients)
- Filtered to keep only "Normal" and "Lung Opacity" classes
- Excluded "No Lung Opacity / Not Normal" (ambiguous cases)
- Removed samples with missing images (3,634 images not found in training directory)

### 2. Label Mapping

For Pneumonia-only classification:
- Normal → Label 0
- Lung Opacity (Pneumonia) → Label 1

For multi-class integration:
- Normal → Label 0
- Tuberculosis → Label 1 (from TB dataset)
- Pneumonia → Label 2

### 3. Dataset Statistics

**Original Dataset:**
- Total unique patients: 26,684
- Normal: 8,851 samples
- Lung Opacity: 6,012 samples
- No Lung Opacity / Not Normal: 11,821 samples (excluded)

**After Cleaning:**
- Total samples: 11,229
- Normal: 6,427 samples
- Pneumonia: 4,802 samples

**Train/Val/Test Split:**
- Train: 7,859 samples (Normal: 4,499, Pneumonia: 3,360)
- Val: 1,685 samples (Normal: 964, Pneumonia: 721)
- Test: 1,685 samples (Normal: 964, Pneumonia: 721)

## Multi-Class Integration

### Combined Dataset Statistics

The multi-class dataset combines:
- Normal samples from Pneumonia dataset
- TB samples from TBX11K dataset
- Pneumonia samples from Pneumonia dataset

**Balanced Multi-Class Dataset:**
- Total samples: 12,000 (4,000 per class)
- Train: 8,399 samples (~2,800 per class)
- Val: 1,801 samples (~600 per class)
- Test: 1,800 samples (600 per class)

### Class Mapping

```
0: Normal (healthy) - from RSNA Pneumonia dataset
1: Tuberculosis (TB) - from TBX11K dataset
2: Pneumonia (Lung Opacity) - from RSNA Pneumonia dataset
```

## Usage

### Binary Pneumonia Classification

```python
from src.data.dataset import TBXRayDataset, build_eval_transform

dataset = TBXRayDataset(
    annotations="data/processed/pneumonia_train.csv",
    image_root="data/raw/Pneumonia/chest x ray/Training/Images",
    transform=build_eval_transform(image_size=300),
    label_names={0: "normal", 1: "pneumonia"},
)
```

### Multi-Class Classification (Normal/TB/Pneumonia)

```python
from src.data.dataset import TBXRayDataset, MULTICLASS_LABEL_NAMES, build_eval_transform

dataset = TBXRayDataset(
    annotations="data/processed/multiclass_train.csv",
    image_root="",  # Image paths are absolute in multi-class dataset
    transform=build_eval_transform(image_size=300),
    label_names=MULTICLASS_LABEL_NAMES,
)
```

## Data Splits

### Pneumonia-Only Splits

- `data/processed/pneumonia_train.csv` - Training split
- `data/processed/pneumonia_val.csv` - Validation split
- `data/processed/pneumonia_test.csv` - Test split
- `data/processed/pneumonia_cleaned.csv` - Combined metadata

### Multi-Class Splits

- `data/processed/multiclass_train.csv` - Training split
- `data/processed/multiclass_val.csv` - Validation split
- `data/processed/multiclass_test.csv` - Test split
- `data/processed/multiclass_class_mapping.txt` - Class mapping reference

## Preprocessing Scripts

### 1. Prepare Pneumonia Data

```bash
python scripts/prepare_pneumonia_data.py
```

This script:
- Loads RSNA Pneumonia metadata
- Filters to Normal and Lung Opacity classes
- Assigns image paths
- Creates numeric labels
- Generates train/val/test splits
- Saves processed CSV files

### 2. Create Multi-Class Dataset

```bash
python scripts/create_multiclass_dataset.py
```

This script:
- Loads TB data from processed TB splits
- Loads Pneumonia data (Normal and Pneumonia classes)
- Balances classes (4,000 samples per class)
- Combines into unified dataset
- Creates stratified train/val/test splits
- Saves multi-class CSV files

## Important Notes

1. **Missing Images**: 3,634 images from the metadata were not found in the training directory. These were excluded from the processed dataset.

2. **Class Balance**: The original Pneumonia dataset is imbalanced (more Normal than Pneumonia). The multi-class dataset is balanced to ensure fair training.

3. **Label Consistency**: The multi-class dataset uses a unified label mapping (0=Normal, 1=TB, 2=Pneumonia) across all three diseases.

4. **Image Paths**: Pneumonia images use absolute paths in the processed CSVs, while TB images use relative paths from the TB image root.

5. **Clinical Features**: The Pneumonia dataset includes age and sex information, which could be used for future multimodal integration.

## References

- RSNA Pneumonia Detection Challenge: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
- Original Paper: "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (2017)

## License

The RSNA Pneumonia dataset is provided by the Radiological Society of North America (RSNA) for research purposes. Please refer to the original dataset license for usage terms.
