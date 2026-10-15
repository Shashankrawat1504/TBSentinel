# TBSentinel Multi-class Classification

## Task

Three-class chest X-ray classification replacing the earlier binary TB / non-TB
task:

| Index | Class | Source |
| --- | --- | --- |
| 0 | Normal | RSNA Pneumonia Detection Challenge, class `Normal` |
| 1 | Tuberculosis | TBX11K, `imgs/tb` only |
| 2 | Pneumonia | RSNA Pneumonia Detection Challenge, class `Lung Opacity` |

The binary TB pipeline (`configs/config.yaml`, `tb_*.csv`, `tb_efficientnet_b3_*.pt`)
is unchanged and still runs with its own two-class configuration.

## Dataset

Built by `scripts/create_multiclass_dataset.py`, which validates every image path
and refuses to write a dataset with leaked, duplicate, or incomplete splits.

| Split | Total | Normal | TB | Pneumonia |
| --- | ---: | ---: | ---: | ---: |
| train | 8,419 | 4,499 | 560 | 3,360 |
| val | 1,805 | 964 | 120 | 721 |
| test | 1,805 | 964 | 120 | 721 |

12,029 distinct images, no image appears in more than one split. Normal and
Pneumonia reuse the split files produced by `scripts/prepare_pneumonia_data.py`
(70/15/15, seed 42, `patientId` unique per image). The 800 TBX11K TB images are
split 70/15/15 with the same seed.

Excluded on purpose:

- TBX11K `imgs/health` (3,800) and `imgs/sick` (3,800): labelled non-TB, but
  neither is "Normal" (`sick` carries other lung disease), so they cannot be
  assigned to any of the three classes without lying.
- TBX11K `imgs/test` (3,302) and `imgs/extra`: no ground-truth labels exist.
- An earlier version of these CSVs assigned every TBX11K image (including the
  unlabelled test split) to the TB class, leaving only ~7% real TB images. Those
  files and the checkpoint trained on them are archived in
  `artifacts/deprecated/poisoned_multiclass_labels/` for audit.

## Training

`python -m src.training.train_centralized_multiclass` (config
`configs/config_multiclass.yaml`): EfficientNet-B3, pretrained, 300 px, batch 8,
AdamW lr `1e-4`, weight decay `1e-4`, cosine schedule to `1e-6` over 15 epochs,
inverse-frequency class weights `[0.289, 2.324, 0.387]`, mixed precision,
gradient clip 1.0, seed 42. Early stopping monitors validation macro-F1
(patience 4) because AUC is undefined for three classes.

Validation history (macro-F1 is the selection metric):

| Epoch | Val accuracy | Val macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.9247 | 0.9420 |
| 2 | 0.9440 | 0.9554 |
| 3 | 0.9219 | 0.9424 |
| 4 | 0.9413 | **0.9569** |
| 5 | 0.9435 | 0.9562 |
| 6 | 0.9341 | 0.9377 |
| 7 | 0.9368 | 0.9539 |
| 8 | 0.9396 | 0.9558 |

Early stopping fired at epoch 8 (4 epochs without improvement); the best
checkpoint is epoch 4.

## Results

Best validation checkpoint (epoch 4), 1,805 validation images:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9413 |
| Macro-F1 | 0.9569 |
| Normal F1 / recall | 0.9460 / 0.9637 |
| TB F1 / recall | 1.0000 / 1.0000 |
| Pneumonia F1 / recall | 0.9246 / 0.9015 |

Confusion matrix, rows = true, order `normal, tb, pneumonia`:
`[[929, 0, 35], [0, 120, 0], [71, 0, 650]]`.

Held-out test split, 1,805 images, evaluated once with
`scripts/evaluate_multiclass_checkpoint.py`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9551 |
| Macro-F1 | 0.9671 |
| Macro AUC (one-vs-rest) | 0.9909 |

| Class | Precision | Recall | F1 | AUC (OvR) | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 0.9437 | 0.9741 | 0.9587 | 0.9868 | 964 |
| tb | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 120 |
| pneumonia | 0.9638 | 0.9223 | 0.9426 | 0.9860 | 721 |

Confusion matrix, rows = true, order `normal, tb, pneumonia`:
`[[939, 0, 25], [0, 120, 0], [56, 0, 665]]`.

## Limitations

- **Source confounding is unresolved.** Normal and Pneumonia come from RSNA
  (pediatric) while TB comes from TBX11K (adult, different acquisition). The TB
  class is perfectly separable (F1 and AUC of 1.0000), which is the expected
  signature of dataset identity rather than disease signal. Treat the TB numbers
  as an interface smoke test, not a performance estimate. The informative part
  of this run is the Normal/Pneumonia boundary, which is a within-source
  comparison and produces 0.94–0.96 F1.
- The task cannot express "abnormal but not TB or pneumonia", which is why
  TBX11K `sick` is excluded rather than folded into Normal.
- TB support is small (120 validation images, 120 test images), so TB
  confidence intervals are wide.
- TBX11K ships no labelled test split; the TB test rows are held-out labelled
  images, not an official test set.
- Single seed, 8 trained epochs; no cross-validation or seed variance study.

## Artifacts and reproduction

- Best model: `artifacts/checkpoints/multiclass_efficientnet_b3_best.pt` (epoch 4)
- Resumable model: `artifacts/checkpoints/multiclass_efficientnet_b3_last.pt`
- Test metrics JSON: `artifacts/outputs/multiclass_test_metrics.json`
- Splits: `data/processed/multiclass_{train,val,test}.csv`
- Label contract and provenance: `data/processed/multiclass_class_mapping.txt`
- Superseded (mislabeled) artifacts: `artifacts/deprecated/poisoned_multiclass_labels/`

```bash
python scripts/create_multiclass_dataset.py                # rebuild + validate splits
python -m src.training.train_centralized_multiclass        # train (15 epochs, early stop)
python scripts/evaluate_multiclass_checkpoint.py           # held-out test metrics
# Continue an interrupted run instead of restarting from scratch:
python -m src.training.train_centralized_multiclass \
    train.resume_from=artifacts/checkpoints/multiclass_efficientnet_b3_last.pt
```

Resuming restores model, optimizer, scheduler, best score, and patience counter,
so an interrupted run continues at the next epoch and cannot overwrite a better
best checkpoint with a worse epoch. Checkpoints are `fsync`-ed before the atomic
rename because a hard kill previously left a truncated `last.pt`.

## Handoff for the inference and frontend phase

`src/inference/service.py` already derives `num_classes` from the checkpoint's
stored config, so the three-class checkpoint loads without code changes
(verified on CPU: Grad-CAM initialises with `['normal', 'tb', 'pneumonia']` and
`predict` returns three probabilities). Still to update:

1. `configs/inference/inference.yaml`: point `checkpoint_path` at
   `multiclass_efficientnet_b3_best.pt` and set
   `class_names: [normal, tb, pneumonia]`.
2. `frontend/app.py`: the result card is binary — the badge keys off class 0 vs
   1, so it needs a three-way badge plus per-class probability bars. The API
   response shape is unchanged (`predicted_class_name`, `confidence`,
   `probabilities`), so no backend contract change is required.
3. Environment: the system Python has an incompatible `fastapi 0.115` +
   `starlette 1.7` pair, which breaks `src/inference/api.py` import and the
   `tests/test_inference_api.py` collection. Use the project virtualenv
   (`.venv`, `fastapi 0.138.2` + `starlette 1.3.1`) or pin `starlette<0.42`.
