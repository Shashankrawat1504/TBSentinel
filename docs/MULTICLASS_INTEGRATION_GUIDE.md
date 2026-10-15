# Multi-Disease Integration Guide

This guide explains how to use TBSentinel for multi-disease classification (Normal / Tuberculosis / Pneumonia).

## Overview

TBSentinel now supports both binary classification (TB vs Non-TB) and multi-class classification (Normal / TB / Pneumonia). The system has been designed to be backward-compatible with existing TB workflows while enabling multi-disease detection.

## Datasets

### 1. TB Dataset (Binary)
- **Source**: TBX11K
- **Classes**: Non-TB (0), TB (1)
- **Files**: `tb_train.csv`, `tb_val.csv`, `tb_test.csv`

### 2. Pneumonia Dataset (Binary)
- **Source**: RSNA Pneumonia Detection Challenge
- **Classes**: Normal (0), Pneumonia (1)
- **Files**: `pneumonia_train.csv`, `pneumonia_val.csv`, `pneumonia_test.csv`

### 3. Multi-Class Dataset
- **Source**: Combined TBX11K + RSNA Pneumonia
- **Classes**: Normal (0), TB (1), Pneumonia (2)
- **Files**: `multiclass_train.csv`, `multiclass_val.csv`, `multiclass_test.csv`
- **Balance**: 4,000 samples per class (12,000 total)

## Label Mapping

### Binary TB Classification
```
0: non_tb
1: tb
```

### Binary Pneumonia Classification
```
0: normal
1: pneumonia
```

### Multi-Class Classification
```
0: normal (healthy)
1: tuberculosis (TB)
2: pneumonia (Lung Opacity)
```

## Configuration

### Binary Classification (Default)

**`configs/model/model.yaml`:**
```yaml
num_classes: 2
```

**`configs/data/data.yaml`:**
```yaml
classification:
  num_classes: 2
  class_to_index:
    non_tb: 0
    tb: 1
multiclass:
  enabled: false
```

### Multi-Class Classification

**`configs/model/model.yaml`:**
```yaml
num_classes: 3
```

**`configs/data/data.yaml`:**
```yaml
classification:
  num_classes: 2  # Keep as is for backward compatibility
multiclass:
  enabled: true
  num_classes: 3
  class_to_index:
    normal: 0
    tb: 1
    pneumonia: 2
  label_names: [normal, tb, pneumonia]
  train_csv: ${data.processed_dir}/multiclass_train.csv
  val_csv: ${data.processed_dir}/multiclass_val.csv
  test_csv: ${data.processed_dir}/multiclass_test.csv
```

## Training

### Binary TB Classification (Existing Workflow)

```bash
python -m src.training.train_centralized
```

This uses the default TB dataset with 2 classes.

### Multi-Class Classification

To train the multi-class model:

1. **Update Configuration:**
   - Set `num_classes: 3` in `configs/model/model.yaml`
   - Set `multiclass.enabled: true` in `configs/data/data.yaml`

2. **Update Training Script:**
   Modify `src/training/train_centralized.py` to:
   - Check if multiclass is enabled
   - Load the appropriate CSV files
   - Use `MULTICLASS_LABEL_NAMES` for the dataset

3. **Run Training:**
   ```bash
   python -m src.training.train_centralized
   ```

### Code Example for Multi-Class Training

```python
from src.data.dataset import TBXRayDataset, MULTICLASS_LABEL_NAMES, build_train_transform, build_eval_transform
from src.models.image_model import TBImageClassifier

# Load multi-class dataset
train_dataset = TBXRayDataset(
    annotations="data/processed/multiclass_train.csv",
    image_root="",  # Absolute paths in multi-class dataset
    transform=build_train_transform(image_size=300),
    label_names=MULTICLASS_LABEL_NAMES,
)

val_dataset = TBXRayDataset(
    annotations="data/processed/multiclass_val.csv",
    image_root="",
    transform=build_eval_transform(image_size=300),
    label_names=MULTICLASS_LABEL_NAMES,
)

# Initialize model with 3 classes
model = TBImageClassifier(
    num_classes=3,
    pretrained=True,
    dropout=0.3,
)
```

## Dataset Class Usage

### Binary Classification

```python
from src.data.dataset import TBXRayDataset, DEFAULT_LABEL_NAMES

dataset = TBXRayDataset(
    annotations="data/processed/tb_train.csv",
    image_root="data/raw/TB/chest x ray/imgs",
    label_names=DEFAULT_LABEL_NAMES,  # {0: "non_tb", 1: "tb"}
)
```

### Multi-Class Classification

```python
from src.data.dataset import TBXRayDataset, MULTICLASS_LABEL_NAMES

dataset = TBXRayDataset(
    annotations="data/processed/multiclass_train.csv",
    image_root="",  # Absolute paths
    label_names=MULTICLASS_LABEL_NAMES,  # {0: "normal", 1: "tb", 2: "pneumonia"}
)
```

## Model Retraining Requirements

### From Binary to Multi-Class

When switching from binary (2 classes) to multi-class (3 classes):

1. **Model Head**: The final classification layer must be updated from 2 to 3 output neurons
2. **Full Retraining**: The entire model should be retrained with the new dataset
3. **Checkpoints**: Old binary checkpoints are not compatible with multi-class models
4. **Metrics**: Update evaluation metrics to support multi-class (accuracy, macro-F1, confusion matrix)

### Model Code Changes

The `TBImageClassifier` already supports variable `num_classes`, so no code changes are needed in the model architecture. Simply update the configuration:

```python
model = TBImageClassifier(
    num_classes=3,  # Changed from 2 to 3
    pretrained=True,
    dropout=0.3,
)
```

## Federated Learning

Federated Learning is compatible with both binary and multi-class classification. The federated splits can be created for the multi-class dataset using a similar approach to the TB federated splits.

### Creating Federated Splits for Multi-Class

```python
from scripts.create_federated_splits import create_federated_splits

create_federated_splits(
    train_csv="data/processed/multiclass_train.csv",
    output_dir="data/federated_splits/multiclass",
    num_clients=3,
    partition_strategy="dirichlet",
    non_iid_alpha=0.5,
)
```

## Grad-CAM Explainability

The Grad-CAM explainer automatically adapts to the number of classes. No changes are needed in the XAI module.

```python
from src.xai.gradcam import TBGradCAMExplainer

explainer = TBGradCAMExplainer(
    checkpoint_path="artifacts/checkpoints/multiclass_model_best.pt",
    class_names=["normal", "tb", "pneumonia"],  # Updated for 3 classes
)
```

## Inference

### Binary Classification Inference

```python
from src.inference.service import InferenceService

service = InferenceService(
    checkpoint_path="artifacts/checkpoints/tb_efficientnet_b3_best.pt",
    num_classes=2,
    class_names=("non_tb", "tb"),
)
```

### Multi-Class Classification Inference

```python
from src.inference.service import InferenceService

service = InferenceService(
    checkpoint_path="artifacts/checkpoints/multiclass_model_best.pt",
    num_classes=3,
    class_names=("normal", "tb", "pneumonia"),
)
```

## Migration Steps

### Step 1: Prepare Multi-Class Dataset

```bash
python scripts/prepare_pneumonia_data.py
python scripts/create_multiclass_dataset.py
```

### Step 2: Update Configuration

Edit `configs/model/model.yaml`:
```yaml
num_classes: 3
```

Edit `configs/data/data.yaml`:
```yaml
multiclass:
  enabled: true
```

### Step 3: Update Training Script

Modify `src/training/train_centralized.py` to:
- Check `cfg.data.multiclass.enabled`
- Load appropriate CSV files
- Use correct label names

### Step 4: Train New Model

```bash
python -m src.training.train_centralized
```

### Step 5: Update Inference Service

Modify `src/inference/service.py` to:
- Load correct number of classes
- Use appropriate class names

### Step 6: Update Frontend

Update the Streamlit dashboard to:
- Show 3-class predictions
- Update UI labels
- Display multi-class confusion matrix

## Backward Compatibility

The system maintains backward compatibility:

1. **Binary TB Classification**: Works exactly as before
2. **Existing Checkpoints**: Compatible with binary models
3. **Configuration**: Default settings use binary classification
4. **Dataset Class**: Automatically adapts to number of classes

## Testing

### Test Binary Classification

```bash
pytest tests/test_image_model.py -v
```

### Test Multi-Class Classification

```python
from src.data.dataset import TBXRayDataset, MULTICLASS_LABEL_NAMES

dataset = TBXRayDataset(
    annotations="data/processed/multiclass_train.csv",
    label_names=MULTICLASS_LABEL_NAMES,
)

assert len(dataset.class_distribution) == 3
assert "normal" in dataset.class_distribution
assert "tb" in dataset.class_distribution
assert "pneumonia" in dataset.class_distribution
```

## Important Notes

1. **Model Compatibility**: Binary and multi-class models use different checkpoint files. Ensure you use the correct checkpoint for the task.

2. **Metric Changes**: Multi-class classification requires different metrics (macro-F1, per-class accuracy, confusion matrix).

3. **Loss Function**: Consider using CrossEntropyLoss with class weights if the dataset is imbalanced.

4. **Data Balance**: The provided multi-class dataset is balanced (4,000 samples per class). If using different data, ensure proper balancing.

5. **Clinical Interpretation**: Multi-class predictions require careful clinical interpretation, especially for borderline cases.

## Troubleshooting

### Issue: Label Mismatch Error

**Problem**: `ValueError: dataset contains unsupported labels`

**Solution**: Ensure the label names in the dataset match the configured `label_names` parameter.

### Issue: Model Output Dimension Error

**Problem**: Model output shape doesn't match number of classes

**Solution**: Ensure `num_classes` in model configuration matches the dataset's label count.

### Issue: Checkpoint Loading Error

**Problem**: Loading a binary checkpoint for multi-class model

**Solution**: Binary checkpoints are incompatible with multi-class models. Train a new model with the multi-class dataset.

## Next Steps

1. **Update Training Script**: Modify `src/training/train_centralized.py` to support multi-class configuration
2. **Update Metrics**: Implement multi-class metrics (macro-F1, per-class precision/recall)
3. **Update Frontend**: Modify Streamlit dashboard to display 3-class predictions
4. **Create Federated Splits**: Generate federated partitions for multi-class dataset
5. **Train Multi-Class Model**: Run centralized and federated training
6. **Update Inference**: Modify inference service to support multi-class predictions
7. **Test End-to-End**: Verify multi-class pipeline works correctly

## References

- [Pneumonia Dataset Card](PNEUMONIA_DATASET_CARD.md)
- [TB Dataset Documentation](../README.md)
- [Hydra Configuration](https://hydra.cc/)
