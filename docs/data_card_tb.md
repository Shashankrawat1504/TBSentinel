# TB Dataset Data Card

## Summary

TBSentinel currently holds two independently sourced TB modalities:

1. Chest X-rays and lesion annotations from the TBX11K collection.
2. Clinical/genomic specimen records and sequenced drug-susceptibility-test
   (DST) observations from files named as TB Portals March 2026 exports.

These sources are **not currently linked at patient level**. The raw tree
contains no crosswalk, and an exact check of image filename stems against
genomic identifiers found no matches. The cleaned multimodal CSV is therefore
schema-ready but contains **zero paired rows** until an authorized, verified
mapping is supplied. No synthetic join or filename-based guess is used.

## Source and description

### Chest X-rays

- Local root: `data/raw/TB/chest x ray/`
- Image directory: `data/raw/TB/chest x ray/imgs/`
- Dataset identity: TBX11K, as identified by the included `README.md`.
- Included source metadata: official train/validation/test lists, COCO-style
  JSON annotations, and 800 XML annotation files.
- The included README describes TBX11K and cites Liu et al., *Rethinking
  Computer-Aided Tuberculosis Diagnosis*, CVPR 2020. It also describes
  additional images from DA, DB, Montgomery, and Shenzhen datasets.

### Genomics and DST

- Primary table:
  `data/raw/TB/Genomic/TB_Portals_Genomics_March_2026.csv`
- Repeated test/observation table:
  `data/raw/TB/Genomic/TB_Portals_Sequenced_DST_March_2026.csv`
- These are retained as distinct source tables. The DST file contains
  repeated observations per specimen and is not merged into the master CSV
  without an explicit observation aggregation policy.
- The locally available files do not include a separate source dictionary or
  a TBX11K-to-TB-Portals crosswalk.

## Modalities and observed sample counts

| Modality/table | Observed count | Notes |
|---|---:|---|
| X-ray image files | 12,278 | 12,174 PNG and 104 JPG; readable, 512 x 512 pixels |
| TBX11K official train/validation | 8,400 | 6,600 train and 1,800 validation entries |
| TBX11K unknown-label test list | 3,302 | `test/unknown_*.png`; labels are not included |
| Extra X-ray images | 576 | Source/split directories exist, but local directory names do not give labels |
| Primary genomics records | 9,333 | 40 source columns; 33 non-identifier feature fields |
| Sequenced DST observations | 44,364 | 179 source columns; 9,221 distinct `specimen_id` values |
| Verified cross-modal pairs | **0** | No mapping file or matching image/specimen identifier found |

The primary genomic file has 9,224 distinct `specimen_identifier` values; 109
rows repeat an identifier. It has no exact duplicate rows. These are specimen
records, not verified unique people.

## Labels and class distribution

For the official TBX11K training/validation list, the folder categories are
used as documented by the supplied data layout:

| Binary label | Source categories | Count |
|---|---|---:|
| `0` (non-TB) | `health` + `sick` | 7,600 |
| `1` (TB) | `tb` | 800 |

The test folder and the additional-source folders are not included in this
class distribution because their labels are not encoded in the local directory
structure. `outcome` in the genomic source is a treatment outcome; it is **not**
a TB-positive/negative label. The genomic source also does not provide a
negative-TB cohort, so it cannot independently support a binary TB diagnostic
label or a feature-versus-X-ray-label correlation.

## Day 5–6 supervised image splits

The current experiment follows **image-first development**. Only TBX11K images
from the official train and validation lists are used for supervised
development:

| Output CSV | Source list | Images | Non-TB (label 0) | TB (label 1) |
|---|---|---:|---:|---:|
| `data/processed/tb_train.csv` | `TBX11K_train.txt` | 6,600 | 6,000 | 600 |
| `data/processed/tb_val.csv` | `TBX11K_val.txt` | 1,800 | 1,600 | 200 |
| `data/processed/tb_test.csv` | `all_test.txt` | 3,302 | not labeled | not labeled |

The official train and validation lists are disjoint and their union matches
`TBX11K_trainval.txt` exactly. Every listed path was found under the image
directory. The supplied test list is `all_test.txt` (there is no separate
`TBX11K_test.txt` in the local lists); it contains 3,302 `test/` images whose
labels are unknown. Those rows are preserved for inference/evaluation
bookkeeping with an empty `label` and `label_name=unknown`; they are never used
as supervised examples. The 576 `extra/` images are excluded from all
supervised splits.

Each CSV contains `image_path`, `label`, `label_name`, and `split`. The image
path is relative to `data/raw/TB/chest x ray/imgs`, which keeps the CSV
portable and directly consumable by `TBXRayDataset` when given that image
root.

Generate the split files and simulated federated clients from the project
root:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_tb_image_splits.py
.\.venv\Scripts\python.exe scripts\create_federated_splits.py
```

## Federated non-IID simulation

`create_federated_splits.py` partitions **only** `tb_train.csv` among three
simulated clients. It uses a per-class Dirichlet allocation with
`alpha=0.5` and deterministic `seed=42`; each training image appears in
exactly one client CSV:

- `data/federated_splits/client_0_train.csv`
- `data/federated_splits/client_1_train.csv`
- `data/federated_splits/client_2_train.csv`

With the configured seed, the generated client counts are:

| Client | Samples | Non-TB | TB |
|---|---:|---:|---:|
| `client_0` | 5,571 | 5,466 | 105 |
| `client_1` | 536 | 534 | 2 |
| `client_2` | 493 | 0 | 493 |

This seed yields a very strong non-IID partition, including a TB-only client;
review local class coverage and convergence before training. Change the seed
only intentionally and record it for reproducibility. Validation and test
CSVs remain global and are not copied into client data.

## Image-first, multimodal-ready design

`TBXRayDataset` returns `(image_tensor, label)` from labeled image CSV rows,
using Albumentations for separate training and deterministic validation/test
transforms. `MultimodalTBDataset` keeps that exact image-only return behavior
when tabular features are absent. It can accept row-aligned numeric features
later and then return `(image_tensor, tabular_tensor, label)`. No tabular or
genomic features are enabled in the current configuration or used in training:
the image/genomic pair count remains **zero** until a verified crosswalk is
available. The image-only implementation therefore prepares an extension
point without inventing pairs or changing the image dataset interface.

## Centralized image-only baseline

The baseline uses an ImageNet-pretrained EfficientNet-B3 with a two-class
dropout head. The image branch is currently trained alone; the optional
tabular encoder/fusion head is disabled until verified pairs exist. Training
reads `tb_train.csv` and model selection reads `tb_val.csv`; the unknown-label
`tb_test.csv` is deliberately never opened by the training script.
Class-weighted cross-entropy compensates for the 10:1 non-TB/TB training
imbalance. AdamW, cosine annealing, mixed precision on CUDA, seeded data
loading/augmentation, early stopping on validation ROC-AUC, best/last
checkpoints, and offline-by-default Weights & Biases logging are configured
with Hydra.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.training.train_centralized
```

The default best checkpoint path is
`artifacts/checkpoints/tb_efficientnet_b3_best.pt`. A completed training run
prints the best validation accuracy, macro-F1, ROC-AUC, sensitivity,
specificity, confusion matrix, selected epoch, and checkpoint path. Validation
scores are experiment outputs, not dataset properties; report only values
produced by a completed run with its seed/configuration recorded.

## Genomic feature fields

The 33 source feature fields retained for the master table are:

`sex`, `age_of_onset`, `type_of_resistance`, `outcome`,
`specimen_collection_date`, `specimen_collection_site`,
`ncbi_sourceorganism`, `main_lineage`, `sub_lineage`, `mixed`,
`spoligotype_lineage`, `spoligotype_octal`, `drug_resistance_type`,
`num_drug_resistant_variants`, `num_other_variants`, `gene_snp_mutations`,
`ethambutol`, `isoniazid`, `rifampicin`, `streptomycin`, `pyrazinamide`,
`amikacin`, `capreomycin`, `cycloserine`, `kanamycin`, `bedaquiline`,
`clofazimine`, `delamanid`, `ethionamide`, `levofloxacin`, `linezolid`,
`moxifloxacin`, and `para_aminosalicylic_acid`.

The seven excluded identifier/accession fields are `condition_id`,
`specimen_id`, `specimen_identifier`, `sra_id`, `ncbi_sra`,
`ncbi_bioproject`, and `ncbi_biosample`. The preparation script uses
`specimen_identifier` as its default explicit mapping key; it is not presumed
to be a unique person identifier.
With the current vocabulary and low-cardinality one-hot encoding, the
schema-ready CSV has 121 genomic feature columns (124 total columns including
the three required identifiers/label fields).

## Cleaning and preparation

Implemented in `scripts/prepare_tb_data.py`:

1. Reads the observed TB Portals genomics file and image root using paths
   relative to the project or explicit command-line arguments.
2. Normalizes source column names to lowercase `snake_case`; it raises an
   error if distinct columns collide after normalization.
3. Requires an explicit mapping CSV with `patient_id` and `image_filename`
   columns before creating any non-empty multimodal rows. A `label` column is
   optional for image categories with an established folder label.
4. Resolves mapped images only inside the image root, checks that files are
   supported and decodable, and excludes invalid, unlabeled, or unmatched
   mapping rows. Conflicts between an explicit label and a known image-folder
   label are errors.
5. Uses only the mapped rows and checks that genomic join keys are unambiguous;
   it does not expand duplicated genomic IDs into a many-to-many join.
6. Converts the known numeric feature fields to numeric values and median
   imputes missing values only when a matched-row median can be calculated.
   Missing categorical values are encoded as `unknown`, never as evidence of
   a negative result.
7. One-hot encodes low-cardinality fields (at most 20 observed values) and
   preserves high-cardinality fields, including mutation text, as strings.
8. Writes `patient_id`, `image_filename` (relative to the image root), and
   integer `label` followed by the encoded genomic feature columns to
   `data/processed/tb_cleaned.csv`.

Running the script without a crosswalk writes a header-only CSV with the
expected feature schema and an explicit warning. No patient rows are dropped
from a purported cross-modal dataset because no such paired dataset exists
yet. Run it from the project root with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_tb_data.py
```

When an approved crosswalk becomes available, pass it explicitly with
`--mapping-csv <path>`; its `patient_id` values must match the selected
`--genomic-id-column` (default `specimen_identifier`).

## Known limitations and biases

- The TBX11K X-rays and TB Portals specimens have different source datasets and
  no shared patient/image key. Their distributions cannot be treated as
  paired or representative of one population.
- The TBX11K official binary labeled subset is class-imbalanced (7,600 non-TB
  versus 800 TB). The 3,302 test images have no released labels in this copy.
- The 576 additional-source images have no local labels, and their source
  populations/devices may differ from TBX11K.
- The supplied X-ray README describes 11,200 TBX11K images plus 1,106 images
  from DA/DB/Montgomery/Shenzhen (12,306 total); the unique paths across the
  local `all_train.txt`, `all_val.txt`, and `all_test.txt` lists total 12,278.
  The resulting 28-image discrepancy needs source-level clarification.
- Image-level patient IDs are absent, so image counts cannot establish
  independent patient counts or detect patient-level train/validation leakage.
- The genomic table repeats 109 specimen identifiers. The biological meaning
  and deduplication policy require source metadata before patient-level use.
- The separate DST table is observation-level and contains repeated
  specimens; aggregation/selection is not defined here.
- Genomic missingness is concentrated in `gene_snp_mutations` (3,089 rows),
  `spoligotype_lineage` (677), `mixed` (390), and `spoligotype_octal` (390).
  Missing variant data must not be interpreted as absence of a mutation.
- `outcome` and resistance fields are clinical/genomic descriptors, not
  diagnostic labels. Their use as inputs requires a clearly defined task and
  leakage review.
- Public dataset provenance does not establish permission to combine records.
  Obtain applicable approvals, data-use terms, and a privacy-preserving
  crosswalk before attempting linkage or federated training.

## Intended multimodal use

After an authorized crosswalk is provided, one master row should represent a
verified patient/specimen-image association. The image path in the CSV is
relative to `data/raw/TB/chest x ray/imgs`; its binary label is derived from
the documented TBX11K folder or a separately verified mapping label. The
specimen key is joined only through the explicit mapping, and the stable
split/federated partition must be patient-level to avoid leakage. Keep official
test labels unavailable, do not merge DST observations without an explicit
policy, and fit categorical/numeric preprocessing on the training subset in
the eventual modeling pipeline.

## Citation information in the included X-ray README

- Liu, Y., Wu, Y.-H., Ban, Y., Wang, H., and Cheng, M.-M. “Rethinking
  Computer-Aided Tuberculosis Diagnosis.” CVPR, 2020.
- The README also lists DA/DB and Montgomery/Shenzhen source papers and
  requests citation when those data are used. Consult the included
  `data/raw/TB/chest x ray/README.md` for their supplied citation details.

## Explainable AI (XAI) Setup: Grad-CAM

To provide clinical transparency for chest X-ray predictions, TBSentinel integrates
a production-ready Gradient-weighted Class Activation Mapping (Grad-CAM) module.

- **Architecture:** EfficientNet-B3 binary image baseline (`tb_efficientnet_b3_best.pt`).
- **Target Convolutional Layer:** `model.backbone.features[-1]` (torchvision `Conv2dNormActivation` block at index 8, 1536 channels).
- **Resolution & Colormap:** 300x300 input resolution, OpenCV `COLORMAP_JET` overlay (with configurable alpha opacity 0.5).
- **Validation Outcome Categorization:**
  - True Positives (TP, 181 validation cases): Demonstrates model activation aligning with pulmonary cavitary lesions, consolidations, and apical infiltrates.
  - True Negatives (TN, 1,597 validation cases): Confirms absence of localized hyper-activations on healthy pulmonary fields or diffuse non-specific markings.
  - False Positives (FP, 3 validation cases): Highlights non-TB pathologies (e.g., severe non-TB pneumonia or fibrosis) triggering false-positive alerts.
  - False Negatives (FN, 19 validation cases): Reveals subtle or atypical lesions missed by the classifier, aiding clinician differential review.
- **Artifacts Directory:** `artifacts/explanations/` contains categorized 3-panel figures, standalone overlays, structured metadata (`explanation_summary.json` / `csv`), and a unified clinical overview collage (`xai_clinical_overview.png`).
- **Reproduction Command:** `python scripts/generate_explanations.py` or `python -m src.xai.explain`.

