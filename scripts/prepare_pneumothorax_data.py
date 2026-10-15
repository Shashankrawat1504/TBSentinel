"""Prepare Pneumothorax dataset for TBSentinel multi-disease classification.

This script processes the SIIM-ACR Pneumothorax dataset to create:
1. Cleaned Pneumothorax-only dataset (confirmed pneumothorax cases)
2. Train/val/test splits compatible with TBSentinel
3. Multi-class ready format for Normal/TB/Pneumonia/Pneumothorax integration

Classes:
- Pneumothorax: Confirmed pneumothorax cases (has_pneumo=1)
- Normal samples from this dataset are NOT used to avoid confounding with existing Normal sources

Label mapping for 4-class system:
- Normal -> 0
- Tuberculosis -> 1
- Pneumonia -> 2
- Pneumothorax -> 3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PNEUMOTHORAX_DIR = DATA_DIR / "raw" / "Pneumothorax"
PROCESSED_DIR = DATA_DIR / "processed"

LOGGER = logging.getLogger(__name__)


def load_pneumothorax_metadata() -> pd.DataFrame:
    """Load and process Pneumothorax metadata from train and test CSVs.

    Returns:
        DataFrame with columns: new_filename, ImageId, has_pneumo, split
    """
    LOGGER.info("Loading Pneumothorax metadata")

    train_path = RAW_PNEUMOTHORAX_DIR / "stage_1_train_images.csv"
    test_path = RAW_PNEUMOTHORAX_DIR / "stage_1_test_images.csv"

    LOGGER.info("Loading training metadata from: %s", train_path)
    train_df = pd.read_csv(train_path)
    train_df["split"] = "train"

    LOGGER.info("Loading test metadata from: %s", test_path)
    test_df = pd.read_csv(test_path)
    test_df["split"] = "test"

    # Combine train and test
    df = pd.concat([train_df, test_df], ignore_index=True)

    LOGGER.info("Total samples: %d", len(df))
    LOGGER.info("Label distribution:\n%s", df["has_pneumo"].value_counts())
    LOGGER.info("Split distribution:\n%s", df["split"].value_counts())

    return df


def filter_pneumothorax_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only confirmed pneumothorax-positive cases.

    We only use has_pneumo=1 for the pneumothorax class.
    The has_pneumo=0 (normal) samples are excluded to avoid confounding
    with existing Normal sources from RSNA.

    Args:
        df: Metadata DataFrame

    Returns:
        Filtered DataFrame with only pneumothorax-positive cases
    """
    LOGGER.info("Filtering to pneumothorax-positive cases (has_pneumo=1)")

    # Keep only pneumothorax-positive
    df_filtered = df[df["has_pneumo"] == 1].copy()

    LOGGER.info("After filtering: %d pneumothorax-positive samples", len(df_filtered))
    LOGGER.info("Split distribution after filtering:\n%s", df_filtered["split"].value_counts())

    return df_filtered


def assign_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Assign image file paths to each sample.

    Args:
        df: Metadata DataFrame with new_filename column

    Returns:
        DataFrame with image_path column
    """
    images_dir = RAW_PNEUMOTHORAX_DIR / "png_images"

    LOGGER.info("Assigning image paths from: %s", images_dir)

    image_paths = []
    missing_count = 0

    for filename in df["new_filename"]:
        image_path = images_dir / filename
        if image_path.exists():
            image_paths.append(str(image_path))
        else:
            image_paths.append(None)
            missing_count += 1

    df["image_path"] = image_paths

    if missing_count > 0:
        LOGGER.warning("%d images not found in png_images directory", missing_count)

    # Remove rows with missing images
    df_clean = df[df["image_path"].notna()].copy()
    LOGGER.info("After removing missing images: %d samples", len(df_clean))

    return df_clean


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric labels for classification.

    For 4-class system:
    - Normal -> 0
    - Tuberculosis -> 1
    - Pneumonia -> 2
    - Pneumothorax -> 3

    This script assigns label=3 for pneumothorax.

    Args:
        df: Metadata DataFrame

    Returns:
        DataFrame with label column
    """
    LOGGER.info("Creating numeric labels (pneumothorax = 3)")

    # All samples in this filtered dataset are pneumothorax
    df["label"] = 3
    df["disease_type"] = "pneumothorax"

    LOGGER.info("All samples labeled as pneumothorax (label=3)")

    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    The original dataset has train/test splits, but we create our own
    stratified splits to match TBSentinel conventions.

    Args:
        df: Cleaned DataFrame
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    LOGGER.info("Creating train/val/test split (val=%.2f, test=%.2f)", val_size, test_size)

    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if len(df["label"].unique()) > 1 else None,
    )

    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_df["label"] if len(train_val_df["label"].unique()) > 1 else None,
    )

    LOGGER.info("Train: %d samples", len(train_df))
    LOGGER.info("Val: %d samples", len(val_df))
    LOGGER.info("Test: %d samples", len(test_df))

    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save processed data splits to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory for CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select relevant columns
    columns = ["image_path", "label", "disease_type", "ImageId"]

    train_output = output_dir / "pneumothorax_train.csv"
    val_output = output_dir / "pneumothorax_val.csv"
    test_output = output_dir / "pneumothorax_test.csv"

    train_df[columns].to_csv(train_output, index=False)
    val_df[columns].to_csv(val_output, index=False)
    test_df[columns].to_csv(test_output, index=False)

    LOGGER.info("Saved training data to: %s", train_output)
    LOGGER.info("Saved validation data to: %s", val_output)
    LOGGER.info("Saved test data to: %s", test_output)


def create_combined_metadata(df: pd.DataFrame, output_dir: Path) -> None:
    """Save combined metadata for reference.

    Args:
        df: Full cleaned DataFrame
        output_dir: Output directory
    """
    combined_output = output_dir / "pneumothorax_cleaned.csv"
    df.to_csv(combined_output, index=False)
    LOGGER.info("Saved combined metadata to: %s", combined_output)


def main() -> None:
    """Main processing pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Pneumothorax data preparation")
    LOGGER.info("=" * 80)

    # Load metadata
    df = load_pneumothorax_metadata()

    # Filter to pneumothorax-positive only
    df_filtered = filter_pneumothorax_positive(df)

    # Assign image paths
    df_with_paths = assign_image_paths(df_filtered)

    # Create labels
    df_labeled = create_labels(df_with_paths)

    # Save combined metadata
    create_combined_metadata(df_labeled, PROCESSED_DIR)

    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(df_labeled)

    # Save processed data
    save_processed_data(train_df, val_df, test_df, PROCESSED_DIR)

    LOGGER.info("=" * 80)
    LOGGER.info("Pneumothorax data preparation completed successfully")
    LOGGER.info("Summary:")
    LOGGER.info("  - Train: %d samples", len(train_df))
    LOGGER.info("  - Val: %d samples", len(val_df))
    LOGGER.info("  - Test: %d samples", len(test_df))
    LOGGER.info("  - Total: %d pneumothorax-positive samples", len(df_labeled))
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
