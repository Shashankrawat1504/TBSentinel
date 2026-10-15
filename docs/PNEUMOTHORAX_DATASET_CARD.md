# Pneumothorax Dataset Card

## Dataset Overview

**Dataset Name:** SIIM-ACR Pneumothorax Detection  
**Source:** Kaggle - SIIM-ACR Pneumothorax Detection Challenge  
**Task:** Binary classification (Pneumothorax present vs absent)  
**Images:** Chest X-rays (DICOM converted to PNG)  
**Use in TBSentinel:** Confirmed pneumothorax cases as the 4th class (label=3)

---

## Dataset Structure

### Raw Dataset Location
```
data/raw/Pneumothorax/
├── png_images/              # 12,047 PNG images (converted from DICOM)
├── png_masks/               # Segmentation masks (not used in TBSentinel)
├── stage_1_train_images.csv  # Training metadata (10,675 samples)
└── stage_1_test_images.csv   # Test metadata (1,372 samples)
```

### Metadata Columns
- `new_filename`: Renamed image filename (e.g., "12345_train_0_.png")
- `ImageId`: Original DICOM SOP Instance UID
- `has_pneumo`: Binary label (0 = no pneumothorax, 1 = pneumothorax present)

---

## Statistics

### Original Dataset
- **Total samples:** 12,047
- **Training samples:** 10,675
- **Test samples:** 1,372
- **Pneumothorax-positive:** 2,669 (22.1%)
- **Pneumothorax-negative:** 9,378 (77.9%)

### Used in TBSentinel (Confirmed Pneumothorax Only)
- **Total pneumothorax-positive:** 2,669
- **Training split:** 1,867 (70%)
- **Validation split:** 401 (15%)
- **Test split:** 401 (15%)

**Note:** We only use `has_pneumo=1` (confirmed pneumothorax) cases. The `has_pneumo=0` (normal) cases are excluded to avoid confounding with existing Normal sources from RSNA Pneumonia dataset.

---

## Class Label Mapping

For the 4-class system:
```
0: normal        (from RSNA Pneumonia)
1: tb            (from TBX11K)
2: pneumonia     (from RSNA Pneumonia)
3: pneumothorax  (from SIIM-ACR Pneumothorax)
```

---

## Data Preparation

### Processing Script
`scripts/prepare_pneumothorax_data.py`

### Processing Steps
1. Load metadata from `stage_1_train_images.csv` and `stage_1_test_images.csv`
2. Filter to keep only pneumothorax-positive cases (`has_pneumo=1`)
3. Assign image paths from `png_images/` directory
4. Verify all image files exist
5. Assign label=3 (pneumothorax) to all samples
6. Create train/val/test splits (70/15/15) with stratification
7. Save processed CSVs to `data/processed/`

### Output Files
- `data/processed/pneumothorax_train.csv` (1,867 samples)
- `data/processed/pneumothorax_val.csv` (401 samples)
- `data/processed/pneumothorax_test.csv` (401 samples)
- `data/processed/pneumothorax_cleaned.csv` (2,669 samples, combined)

---

## Integration into Multi-Class4 Dataset

### Combined Dataset Script
`scripts/create_multiclass4_dataset.py`

### Final 4-Class Dataset Statistics
- **Total training samples:** 10,286
  - Normal: 4,499 (43.7%)
  - TB: 560 (5.4%)
  - Pneumonia: 3,360 (32.7%)
  - Pneumothorax: 1,867 (18.2%)
- **Total validation samples:** 2,206
  - Normal: 964 (43.7%)
  - TB: 120 (5.4%)
  - Pneumonia: 721 (32.7%)
  - Pneumothorax: 401 (18.2%)
- **Total test samples:** 2,206
  - Normal: 964 (43.7%)
  - TB: 120 (5.4%)
  - Pneumonia: 721 (32.7%)
  - Pneumothorax: 401 (18.2%)

### Output Files
- `data/processed/multiclass4_train.csv`
- `data/processed/multiclass4_val.csv`
- `data/processed/multiclass4_test.csv`
- `data/processed/multiclass4_class_mapping.txt`

---

## Dataset Characteristics

### Image Properties
- **Format:** PNG (converted from DICOM)
- **Bit depth:** 8-bit grayscale (converted to RGB for model)
- **Resolution:** Variable (resized to 300x300 during training)
- **Orientation:** Mostly anterior-posterior (AP) chest X-rays

### Label Quality
- **Ground truth:** Radiologist-confirmed pneumothorax
- **Label source:** Expert annotations from SIIM-ACR challenge
- **Ambiguity:** Minimal - binary classification task with clear labels

---

## Limitations and Risks

### 1. Class Imbalance
- Pneumothorax is a minority class (18.2% in combined dataset)
- This imbalance may affect model performance
- Consider class weighting or focal loss during training

### 2. Source Domain Confounding
- Pneumothorax comes from SIIM-ACR (adult patients)
- Normal and Pneumonia come from RSNA (pediatric patients)
- TB comes from TBX11K (adult patients)
- Source identity is confounded with class boundaries
- Metrics may not be clinically transferable

### 3. Normal Exclusion
- We exclude pneumothorax-negative cases from this dataset
- This is intentional to avoid adding another Normal source
- The Normal class is already well-represented from RSNA

### 4. Test Set Composition
- Pneumothorax test set is a held-out split from labeled data
- Not an official test set (unlike some other challenges)
- Should be treated as validation data rather than benchmark

### 5. Age Demographics
- SIIM-ACR dataset primarily contains adult patients
- Different from RSNA (pediatric) and TBX11K (adult)
- Age distribution may affect model generalization

---

## Recommended Usage

### Training Considerations
1. **Class Weighting:** Use inverse frequency weighting to handle imbalance
2. **Focal Loss:** Consider focal loss to focus on hard examples
3. **Data Augmentation:** Apply to address minority class under-representation
4. **Stratified Sampling:** Ensure all classes represented in each batch

### Evaluation Metrics
- **Primary:** Macro F1-score (balanced across all classes)
- **Secondary:** Per-class recall (sensitivity for each disease)
- **Confusion Matrix:** Analyze confusion between similar classes
- **Class-wise Metrics:** Monitor pneumothorax specifically due to minority status

---

## References

1. **SIIM-ACR Pneumothorax Detection Challenge**
   - https://www.kaggle.com/c/siim-acr-pneumothorax-detection
   - Original challenge paper and methodology

2. **Original Dataset Paper**
   - (Add citation when available)

3. **Clinical Background**
   - Pneumothorax: Collection of air in pleural space
   - Can be life-threatening if large
   - Typically diagnosed on chest X-ray
   - May require urgent intervention

---

## Contact

For questions about this dataset preparation or integration, refer to:
- Dataset preparation script: `scripts/prepare_pneumothorax_data.py`
- Multi-class integration: `scripts/create_multiclass4_dataset.py`
- Dataset module: `src/data/dataset.py`

---

**Last Updated:** 2026-10-11  
**Version:** 1.0  
**Prepared by:** TBSentinel ML Engineering Team
