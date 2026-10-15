"""Create the combined 4-class dataset for Normal / TB / Pneumonia / Pneumothorax.

Classes and sources:

- Normal       (label=0): RSNA Pneumonia "Normal" chest X-rays
- Tuberculosis (label=1): TBX11K ``imgs/tb`` chest X-rays only
- Pneumonia    (label=2): RSNA "Lung Opacity" chest X-rays
- Pneumothorax (label=3): SIIM-ACR Pneumothorax confirmed pneumothorax cases

The RSNA component reuses the splits written by
``scripts/prepare_pneumonia_data.py`` (``pneumonia_{train,val,test}.csv``) so the
multi-class experiment stays consistent with the pneumonia phase. The 800 TBX11K
TB images are split 70/15/15 with a fixed seed and merged into the same splits.
The Pneumothorax component uses the splits from
``scripts/prepare_pneumothorax_data.py`` (``pneumothorax_{train,val,test}.csv``).

Deliberately excluded:

- TBX11K ``imgs/health`` and ``imgs/sick``: labeled non-TB images. They are not
  TB, and they are not "Normal" either (``sick`` carries other lung disease),
  so they cannot be assigned to any of the classes without lying.
- TBX11K ``imgs/test`` and ``imgs/extra``: no ground-truth labels exist.
- Pneumothorax normal cases (has_pneumo=0): excluded to avoid confounding with
  existing Normal sources from RSNA.

Known limitations, reported rather than hidden:

1. Normal/Pneumonia come from RSNA (pediatric) while TB comes from TBX11K
   (adult, different acquisition). Source identity is therefore confounded with
   the TB class boundary.
2. Pneumothorax comes from SIIM-ACR (adult), adding another source domain.
3. TBX11K has no labeled test split; the TB test rows are held-out images from
   the labeled pool, not an official test set.
4. Class imbalance is significant: Normal and Pneumonia are majority classes,
   TB and Pneumothorax are minority classes.

Usage:
    python scripts/create_multiclass4_dataset.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_TB_DIR = DATA_DIR / "raw" / "TB" / "chest x ray"
TB_IMAGE_ROOT = RAW_TB_DIR / "imgs"

# Label contract for 4-class system
MULTICLASS4_LABELS: dict[str, int] = {
    "normal": 0,
    "tb": 1,
    "pneumonia": 2,
    "pneumothorax": 3,
}
DISEASE_TYPES: dict[str, str] = {
    "normal": "normal",
    "tb": "tuberculosis",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}
RSNA_CLASS_TO_CLASS_NAME: dict[str, str] = {
    "Normal": "normal",
    "Lung Opacity": "pneumonia",
}

SPLITS: tuple[str, ...] = ("train", "val", "test")
TB_VAL_SIZE = 0.15
TB_TEST_SIZE = 0.15
SEED = 42
PATH_CHECK_SAMPLE_FRACTION = 1.0  # 1.0 = verify every image path exists

LOGGER = logging.getLogger(__name__)


def assign_rsna_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map RSNA rows to the multi-class label contract.

    Args:
        df: Rows from a ``pneumonia_{split}.csv`` file with a ``class`` column.

    Returns:
        Frame with ``image_path``, ``label``, and ``disease_type`` columns.
    """
    unknown_classes = sorted(set(df["class"]).difference(RSNA_CLASS_TO_CLASS_NAME))
    if unknown_classes:
        raise ValueError(f"Unexpected RSNA classes: {unknown_classes}")
    class_names = df["class"].map(RSNA_CLASS_TO_CLASS_NAME)
    result = pd.DataFrame(
        {
            "image_path": df["image_path"].astype(str),
            "label": class_names.map(MULTICLASS4_LABELS).astype(int),
            "disease_type": class_names.map(DISEASE_TYPES),
        }
    )
    return result.reset_index(drop=True)


def load_rsna_split(split: str) -> pd.DataFrame:
    """Load one prepared RSNA split and relabel it for four classes."""
    csv_path = PROCESSED_DIR / f"pneumonia_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prepared pneumonia split is missing: {csv_path}. "
            "Run scripts/prepare_pneumonia_data.py first."
        )
    df = pd.read_csv(csv_path)
    missing = {"image_path", "class"}.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    labeled = assign_rsna_labels(df)
    LOGGER.info(
        "RSNA %s: %d images %s",
        split,
        len(labeled),
        labeled["label"].value_counts().sort_index().to_dict(),
    )
    return labeled


def load_tb_images() -> pd.DataFrame:
    """Return the labeled TBX11K TB images from the prepared binary CSVs.

    Only rows whose prepared ``label_name`` is ``tb`` are used. The unlabeled
    TBX11K test split and the extra/health/sick folders are never read.
    """
    frames = []
    for split in ("train", "val"):
        csv_path = PROCESSED_DIR / f"tb_{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Prepared TB split is missing: {csv_path}. "
                "Run scripts/prepare_tb_data.py first."
            )
        df = pd.read_csv(csv_path)
        missing = {"image_path", "label_name"}.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        tb_rows = df[df["label_name"] == "tb"].copy()
        frames.append(tb_rows)
    positive = pd.concat(frames, ignore_index=True)
    absolute = [
        str(Path(value))
        if Path(value).is_absolute()
        else str(TB_IMAGE_ROOT / Path(value))
        for value in positive["image_path"].astype(str)
    ]
    tb_frame = pd.DataFrame(
        {
            "image_path": absolute,
            "label": MULTICLASS4_LABELS["tb"],
            "disease_type": DISEASE_TYPES["tb"],
        }
    )
    duplicated = tb_frame["image_path"].duplicated().sum()
    if duplicated:
        raise ValueError(f"Prepared TB splits contain {duplicated} duplicate images.")
    LOGGER.info("TBX11K tb images available: %d", len(tb_frame))
    return tb_frame


def load_pneumothorax_split(split: str) -> pd.DataFrame:
    """Load one prepared Pneumothorax split and relabel it for four classes."""
    csv_path = PROCESSED_DIR / f"pneumothorax_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Prepared pneumothorax split is missing: {csv_path}. "
            "Run scripts/prepare_pneumothorax_data.py first."
        )
    df = pd.read_csv(csv_path)
    missing = {"image_path", "label", "disease_type"}.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    
    # Verify all are labeled as pneumothorax (label=3)
    if (df["label"] != 3).any():
        raise ValueError(f"{csv_path} contains labels other than pneumothorax (3)")
    
    # Select only needed columns
    result = df[["image_path", "label", "disease_type"]].copy()
    LOGGER.info("Pneumothorax %s: %d images", split, len(result))
    return result


def split_tb_images(
    tb_frame: pd.DataFrame,
    val_size: float = TB_VAL_SIZE,
    test_size: float = TB_TEST_SIZE,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Split the TB images into train/val/test with a fixed seed."""
    if not 0.0 < val_size + test_size < 1.0:
        raise ValueError("val_size + test_size must be in (0, 1).")
    train_frame, remaining = train_test_split(
        tb_frame,
        test_size=val_size + test_size,
        random_state=seed,
        shuffle=True,
    )
    relative_test = test_size / (val_size + test_size)
    val_frame, test_frame = train_test_split(
        remaining,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
    )
    return {"train": train_frame, "val": val_frame, "test": test_frame}


def combine_splits(
    rsna_splits: dict[str, pd.DataFrame],
    tb_splits: dict[str, pd.DataFrame],
    pneumothorax_splits: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Concatenate the RSNA, TB, and Pneumothorax components per split and shuffle once."""
    combined: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        merged = pd.concat(
            [rsna_splits[split], tb_splits[split], pneumothorax_splits[split]],
            ignore_index=True
        )
        shuffled = merged.sample(frac=1, random_state=SEED).reset_index(drop=True)
        combined[split] = shuffled
    return combined


def validate_dataset(
    splits: dict[str, pd.DataFrame],
    check_paths: bool = True,
) -> None:
    """Fail loudly on label drift, leakage, or missing image files."""
    expected_labels = set(MULTICLASS4_LABELS.values())
    for split, df in splits.items():
        observed = set(df["label"].unique())
        if observed != expected_labels:
            raise ValueError(
                f"{split} split labels {sorted(observed)} do not match "
                f"the contract {sorted(expected_labels)}."
            )
        if df["image_path"].duplicated().any():
            raise ValueError(f"{split} split contains duplicate image paths.")
        mismatched = df.groupby("label")["disease_type"].nunique()
        if (mismatched > 1).any():
            raise ValueError(f"{split} split maps one label to several diseases.")

    train_paths = set(splits["train"]["image_path"])
    for split in ("val", "test"):
        overlap = train_paths.intersection(splits[split]["image_path"])
        if overlap:
            raise ValueError(
                f"{len(overlap)} images appear in both train and {split}."
            )
    val_test_overlap = set(splits["val"]["image_path"]).intersection(
        splits["test"]["image_path"]
    )
    if val_test_overlap:
        raise ValueError(f"{len(val_test_overlap)} images appear in both val and test.")

    if check_paths:
        for split, df in splits.items():
            sample = df
            if PATH_CHECK_SAMPLE_FRACTION < 1.0:
                sample = df.sample(
                    n=max(1, int(len(df) * PATH_CHECK_SAMPLE_FRACTION)),
                    random_state=SEED,
                )
            missing = [
                path for path in sample["image_path"] if not Path(path).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{split} split references {len(missing)} missing files, "
                    f"first: {missing[0]}"
                )
            LOGGER.info(
                "Verified %d/%d %s image paths exist",
                len(sample),
                len(df),
                split,
            )


def save_multiclass4_data(
    splits: dict[str, pd.DataFrame],
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write one CSV per split with the shared 4-class schema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        output_path = output_dir / f"multiclass4_{split}.csv"
        df.to_csv(output_path, index=False)
        LOGGER.info("Saved %d rows to %s", len(df), output_path)


def save_class_mapping(
    splits: dict[str, pd.DataFrame],
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write the label contract, provenance, and caveats next to the CSVs."""
    mapping_path = output_dir / "multiclass4_class_mapping.txt"
    lines = [
        "Multi-class4 Label Mapping (4-class system)",
        "=" * 50,
        "",
    ]
    for class_name, index in MULTICLASS4_LABELS.items():
        lines.append(f"{index}: {class_name}")
    lines += [
        "",
        "Sources:",
        "- normal (0): RSNA Pneumonia dataset, class 'Normal'",
        "- tb (1): TBX11K 'imgs/tb' only",
        "- pneumonia (2): RSNA Pneumonia dataset, class 'Lung Opacity'",
        "- pneumothorax (3): SIIM-ACR Pneumothorax dataset, confirmed pneumothorax cases",
        "",
        f"Split policy: TB images split {int((1 - TB_VAL_SIZE - TB_TEST_SIZE) * 100)}/"
        f"{int(TB_VAL_SIZE * 100)}/{int(TB_TEST_SIZE * 100)} with seed {SEED}; "
        "RSNA splits reused from scripts/prepare_pneumonia_data.py; "
        "Pneumothorax splits reused from scripts/prepare_pneumothorax_data.py.",
        "",
        "Excluded: TBX11K imgs/health, imgs/sick (labeled non-TB but not Normal),",
        "          TBX11K imgs/test, imgs/extra (unlabeled).",
        "          Pneumothorax normal cases (has_pneumo=0) to avoid confounding.",
        "",
        "Caveats:",
        "- Normal and Pneumonia come from RSNA (pediatric); TB comes from TBX11K",
        "  (adult); Pneumothorax comes from SIIM-ACR (adult). Source identity is",
        "  confounded with class boundaries, so these metrics are not a clinically",
        "  transferable estimate.",
        "- TBX11K has no labeled test split; the TB test rows above are held-out",
        "  images from the labeled pool, not an official test set.",
        "- Significant class imbalance: Normal and Pneumonia are majority classes,",
        "  TB and Pneumothorax are minority classes.",
        "",
        "Split sizes:",
    ]
    for split, df in splits.items():
        counts = df["label"].value_counts().sort_index().to_dict()
        reverse = {index: name for name, index in MULTICLASS4_LABELS.items()}
        readable = {reverse[label]: count for label, count in counts.items()}
        lines.append(f"- {split}: {len(df)} {readable}")
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved class mapping to: %s", mapping_path)


def main() -> None:
    """Build, validate, and write the 4-class dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    started = time.perf_counter()
    LOGGER.info("=" * 80)
    LOGGER.info("Starting 4-class dataset creation (Normal/TB/Pneumonia/Pneumothorax)")
    LOGGER.info("=" * 80)

    # Load RSNA (Normal + Pneumonia)
    rsna_splits = {split: load_rsna_split(split) for split in SPLITS}
    
    # Load and split TB
    tb_splits = split_tb_images(load_tb_images())
    for split, df in tb_splits.items():
        LOGGER.info("TB %s: %d images", split, len(df))
    
    # Load Pneumothorax (already split)
    pneumothorax_splits = {split: load_pneumothorax_split(split) for split in SPLITS}

    # Combine all sources
    splits = combine_splits(rsna_splits, tb_splits, pneumothorax_splits)
    validate_dataset(splits)
    
    for split, df in splits.items():
        counts = df["label"].value_counts().sort_index().to_dict()
        reverse = {index: name for name, index in MULTICLASS4_LABELS.items()}
        readable = {reverse[label]: count for label, count in counts.items()}
        LOGGER.info("%s: %d rows, labels %s", split, len(df), readable)

    save_multiclass4_data(splits)
    save_class_mapping(splits)

    LOGGER.info("Finished in %.1fs", time.perf_counter() - started)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
