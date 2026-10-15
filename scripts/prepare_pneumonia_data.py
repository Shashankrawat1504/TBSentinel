"""Prepare RSNA Pneumonia dataset for TBSentinel multi-disease classification.

This script processes the RSNA Pneumonia dataset to create:
1. Cleaned Pneumonia-only dataset (Normal vs Pneumonia)
2. Train/val/test splits compatible with TBSentinel
3. Multi-class ready format for Normal/TB/Pneumonia integration

Classes:
- Normal: Healthy chest X-rays (from Pneumonia dataset)
- Pneumonia: Lung Opacity cases (from Pneumonia dataset)
- Excluded: "No Lung Opacity / Not Normal" (ambiguous/other conditions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PNEUMONIA_DIR = DATA_DIR / "raw" / "Pneumonia" / "chest x ray"
PROCESSED_DIR = DATA_DIR / "processed"

LOGGER = logging.getLogger(__name__)


def load_pneumonia_metadata() -> pd.DataFrame:
    """Load and process RSNA Pneumonia training metadata.

    Returns:
        DataFrame with columns: patientId, class, Target, age, sex, modality, position
    """
    metadata_path = RAW_PNEUMONIA_DIR / "stage2_train_metadata.csv"
    LOGGER.info("Loading Pneumonia metadata from: %s", metadata_path)

    df = pd.read_csv(metadata_path)

    # Deduplicate by patientId (keep first occurrence)
    df_unique = df.drop_duplicates(subset=["patientId"], keep="first")

    LOGGER.info("Total unique patients: %d", len(df_unique))
    LOGGER.info("Class distribution:\n%s", df_unique["class"].value_counts())
    LOGGER.info("Target distribution:\n%s", df_unique["Target"].value_counts())

    return df_unique


def filter_pneumonia_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only Normal and Lung Opacity (Pneumonia) classes.

    Excludes "No Lung Opacity / Not Normal" as these are ambiguous/other conditions.

    Args:
        df: Metadata DataFrame

    Returns:
        Filtered DataFrame with only Normal and Pneumonia classes
    """
    LOGGER.info("Filtering to Normal and Lung Opacity classes")

    # Keep only Normal and Lung Opacity
    df_filtered = df[df["class"].isin(["Normal", "Lung Opacity"])].copy()

    LOGGER.info("After filtering: %d samples", len(df_filtered))
    LOGGER.info("Filtered class distribution:\n%s", df_filtered["class"].value_counts())

    return df_filtered


def assign_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Assign image file paths to each patient.

    Args:
        df: Metadata DataFrame with patientId column

    Returns:
        DataFrame with image_path column
    """
    images_dir = RAW_PNEUMONIA_DIR / "Training" / "Images"

    LOGGER.info("Assigning image paths from: %s", images_dir)

    image_paths = []
    missing_count = 0

    for patient_id in df["patientId"]:
        image_path = images_dir / f"{patient_id}.png"
        if image_path.exists():
            image_paths.append(str(image_path))
        else:
            image_paths.append(None)
            missing_count += 1

    df["image_path"] = image_paths

    if missing_count > 0:
        LOGGER.warning("%d images not found in training directory", missing_count)

    # Remove rows with missing images
    df_clean = df[df["image_path"].notna()].copy()
    LOGGER.info("After removing missing images: %d samples", len(df_clean))

    return df_clean


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric labels for classification.

    Mapping:
    - Normal -> 0
    - Lung Opacity (Pneumonia) -> 1

    For future multi-class integration:
    - Normal -> 0
    - Tuberculosis -> 1
    - Pneumonia -> 2

    Args:
        df: Metadata DataFrame with class column

    Returns:
        DataFrame with label column
    """
    LOGGER.info("Creating numeric labels")

    # Map class names to labels
    label_mapping = {
        "Normal": 0,
        "Lung Opacity": 1,  # Pneumonia
    }

    df["label"] = df["class"].map(label_mapping)

    LOGGER.info("Label distribution:\n%s", df["label"].value_counts())

    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

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
        stratify=df["label"],
    )

    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    LOGGER.info("Train: %d samples", len(train_df))
    LOGGER.info("Val: %d samples", len(val_df))
    LOGGER.info("Test: %d samples", len(test_df))

    LOGGER.info("Train label distribution:\n%s", train_df["label"].value_counts())
    LOGGER.info("Val label distribution:\n%s", val_df["label"].value_counts())
    LOGGER.info("Test label distribution:\n%s", test_df["label"].value_counts())

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
    columns = ["patientId", "image_path", "class", "label", "age", "sex", "modality", "position"]

    train_output = output_dir / "pneumonia_train.csv"
    val_output = output_dir / "pneumonia_val.csv"
    test_output = output_dir / "pneumonia_test.csv"

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
    combined_output = output_dir / "pneumonia_cleaned.csv"
    df.to_csv(combined_output, index=False)
    LOGGER.info("Saved combined metadata to: %s", combined_output)


def main() -> None:
    """Main processing pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Pneumonia data preparation")
    LOGGER.info("=" * 80)

    # Load metadata
    df = load_pneumonia_metadata()

    # Filter classes
    df_filtered = filter_pneumonia_classes(df)

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
    LOGGER.info("Pneumonia data preparation completed successfully")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
