"""Prepare a conservative image/genomics master table for the TB dataset."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("prepare_tb_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "TB"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tb_cleaned.csv"
GENOMICS_RELATIVE_PATH = Path("Genomic") / "TB_Portals_Genomics_March_2026.csv"
IMAGE_RELATIVE_PATH = Path("chest x ray") / "imgs"

IDENTIFIER_COLUMNS = {
    "condition_id",
    "specimen_id",
    "specimen_identifier",
    "sra_id",
    "ncbi_sra",
    "ncbi_bioproject",
    "ncbi_biosample",
}
NUMERIC_COLUMNS = {
    "age_of_onset",
    "num_drug_resistant_variants",
    "num_other_variants",
}
KNOWN_IMAGE_LABELS = {"tb": 1, "health": 0, "sick": 0}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_ONE_HOT_CATEGORIES = 20


def normalize_column_name(name: str) -> str:
    """Normalize a column label to lowercase snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().casefold())
    return normalized.strip("_")


def normalize_category_value(value: str) -> str:
    """Normalize a categorical value for stable one-hot feature names."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().casefold())
    return normalized.strip("_") or "unknown"


def normalize_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize column labels and fail rather than silently merge collisions."""
    result = frame.copy()
    normalized = [normalize_column_name(str(column)) for column in result.columns]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{source} contains columns that collide after normalization: "
            f"{duplicates}"
        )
    result.columns = normalized
    return result


def image_label_from_path(image_path: Path, image_root: Path) -> int | None:
    """Return the known binary label encoded by a labeled image folder."""
    try:
        relative_parts = image_path.resolve().relative_to(image_root.resolve()).parts
    except ValueError:
        return None
    for part in reversed(relative_parts[:-1]):
        label = KNOWN_IMAGE_LABELS.get(part.casefold())
        if label is not None:
            return label
    return None


def parse_binary_label(value: Any) -> int:
    """Convert a documented binary label representation to 0 or 1."""
    if pd.isna(value):
        raise ValueError("A supplied image label cannot be missing.")
    normalized = re.sub(r"[\s_-]+", " ", str(value).strip().casefold())
    positive = {"1", "tb", "positive", "active tb", "tb positive"}
    negative = {"0", "non tb", "negative", "healthy", "sick", "tb negative"}
    if normalized in positive:
        return 1
    if normalized in negative:
        return 0
    raise ValueError(
        f"Unsupported label {value!r}; use TB/positive/1 or " "non-TB/negative/0."
    )


def resolve_image_path(value: Any, raw_root: Path, image_root: Path) -> Path | None:
    """Resolve a mapping path and reject paths outside the image collection."""
    if pd.isna(value) or not str(value).strip():
        return None

    supplied = Path(str(value).strip())
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [
            image_root / supplied,
            raw_root / supplied,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(image_root.resolve())
        except ValueError:
            continue
        if resolved.is_file() and resolved.suffix.casefold() in IMAGE_SUFFIXES:
            return resolved
    return None


def is_valid_image(image_path: Path) -> bool:
    """Check image decodability without retaining the image in memory."""
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def read_mapping(
    mapping_path: Path,
    raw_root: Path,
    image_root: Path,
) -> tuple[pd.DataFrame, int]:
    """Load and validate an explicit patient-to-image crosswalk."""
    mapping = normalize_columns(pd.read_csv(mapping_path, dtype=str), str(mapping_path))
    required = {"patient_id", "image_filename"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(
            f"Mapping file must include columns {sorted(required)}; "
            f"missing {sorted(missing)}."
        )

    valid_rows: list[dict[str, Any]] = []
    invalid_rows = 0
    for row in mapping.to_dict(orient="records"):
        raw_patient_id = row.get("patient_id")
        patient_id = (
            ""
            if raw_patient_id is None or pd.isna(raw_patient_id)
            else str(raw_patient_id).strip()
        )
        image_path = resolve_image_path(row.get("image_filename"), raw_root, image_root)
        if not patient_id or image_path is None or not is_valid_image(image_path):
            invalid_rows += 1
            continue

        inferred_label = image_label_from_path(image_path, image_root)
        supplied_label = row.get("label")
        if supplied_label is None or pd.isna(supplied_label):
            supplied_label = ""
        else:
            supplied_label = str(supplied_label).strip()
        if supplied_label:
            label = parse_binary_label(supplied_label)
            if inferred_label is not None and label != inferred_label:
                raise ValueError(
                    f"Mapping label for {image_path.name!r} conflicts with "
                    "its labeled folder."
                )
        else:
            label = inferred_label
        if label is None:
            invalid_rows += 1
            continue

        valid_rows.append(
            {
                "patient_id": patient_id,
                "image_filename": image_path.relative_to(image_root).as_posix(),
                "label": label,
            }
        )

    validated = pd.DataFrame(
        valid_rows, columns=["patient_id", "image_filename", "label"]
    )
    duplicate_rows = int(validated.duplicated(["patient_id", "image_filename"]).sum())
    if duplicate_rows:
        LOGGER.warning("Dropping %d duplicate mapping rows.", duplicate_rows)
        validated = validated.drop_duplicates(["patient_id", "image_filename"])
    return validated.reset_index(drop=True), invalid_rows


def feature_encoding_plan(
    genomic: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Choose deterministic one-hot vocabularies and numeric feature fields."""
    vocabularies: dict[str, list[str]] = {}
    numeric_columns: list[str] = []
    text_columns: list[str] = []

    for column in feature_columns:
        values = genomic[column]
        if column in NUMERIC_COLUMNS:
            numeric_columns.append(column)
            continue

        normalized = values.dropna().astype(str).map(normalize_category_value)
        unique_values = sorted(value for value in normalized.unique() if value)
        is_date = column.endswith("_date")
        if not is_date and len(unique_values) <= MAX_ONE_HOT_CATEGORIES:
            vocabularies[column] = sorted(set(unique_values) | {"unknown"})
        else:
            text_columns.append(column)
    return vocabularies, numeric_columns, text_columns


def encode_features(
    rows: pd.DataFrame,
    feature_columns: list[str],
    vocabularies: dict[str, list[str]],
    numeric_columns: list[str],
    text_columns: list[str],
) -> pd.DataFrame:
    """Encode low-cardinality fields and impute values without inventing labels."""
    encoded_parts: list[pd.DataFrame] = []
    for column, categories in vocabularies.items():
        values = rows[column].astype("string").fillna("unknown")
        values = values.map(normalize_category_value)
        category_values = pd.Categorical(values, categories=categories)
        encoded = pd.get_dummies(
            category_values,
            prefix=column,
            dtype="int8",
        )
        encoded.index = rows.index
        encoded_parts.append(encoded)

    remaining_columns = [
        column for column in feature_columns if column not in vocabularies
    ]
    remaining = rows[remaining_columns].copy()
    for column in numeric_columns:
        values = pd.to_numeric(remaining[column], errors="coerce")
        median = values.median()
        if pd.notna(median):
            missing_count = int(values.isna().sum())
            if missing_count:
                LOGGER.info(
                    "Median-imputing %d missing value(s) in %s.",
                    missing_count,
                    column,
                )
                values = values.fillna(median)
        remaining[column] = values
    for column in text_columns:
        values = remaining[column].astype("string").str.strip()
        remaining[column] = values.fillna("unknown").replace("", "unknown")

    if encoded_parts:
        return pd.concat([*encoded_parts, remaining], axis=1)
    return remaining


def prepare_tb_data(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    mapping_path: Path | None = None,
    genomic_id_column: str = "specimen_identifier",
) -> dict[str, Any]:
    """Write a master CSV containing only explicitly mapped valid image pairs."""
    raw_root = raw_root.resolve()
    output_path = output_path.resolve()
    image_root = raw_root / IMAGE_RELATIVE_PATH
    genomic_path = raw_root / GENOMICS_RELATIVE_PATH
    if not genomic_path.is_file():
        raise FileNotFoundError(f"Genomic CSV not found: {genomic_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    genomic = normalize_columns(pd.read_csv(genomic_path, dtype=str), str(genomic_path))
    genomic_id_column = normalize_column_name(genomic_id_column)
    if genomic_id_column not in genomic.columns:
        raise ValueError(
            f"Genomic identifier column {genomic_id_column!r} not found. "
            f"Available columns: {list(genomic.columns)}"
        )
    absent_identifiers = IDENTIFIER_COLUMNS.difference(genomic.columns)
    if absent_identifiers:
        LOGGER.warning(
            "Some expected identifier fields are absent: %s",
            sorted(absent_identifiers),
        )
    identifier_columns = IDENTIFIER_COLUMNS.intersection(genomic.columns)
    feature_columns = [
        column for column in genomic.columns if column not in identifier_columns
    ]
    vocabularies, numeric_columns, text_columns = feature_encoding_plan(
        genomic, feature_columns
    )

    image_count = sum(
        1
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    invalid_mapping_rows = 0
    matched_mapping_rows = 0
    unmatched_mapping_rows = 0

    if mapping_path is None:
        LOGGER.warning(
            "No explicit image/genomics mapping was supplied. The inspected "
            "sources have no shared image/specimen identifiers; writing an "
            "empty, schema-ready master CSV rather than fabricating pairs."
        )
        mapping = pd.DataFrame(columns=["patient_id", "image_filename", "label"])
    else:
        mapping_path = mapping_path.resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")
        mapping, invalid_mapping_rows = read_mapping(mapping_path, raw_root, image_root)
        mapping_ids = set(mapping["patient_id"])
        selected = genomic.loc[
            genomic[genomic_id_column].astype("string").isin(mapping_ids)
        ].copy()
        duplicate_ids = (
            selected.loc[
                selected[genomic_id_column].duplicated(keep=False),
                genomic_id_column,
            ]
            .dropna()
            .unique()
        )
        if len(duplicate_ids):
            examples = [str(value) for value in duplicate_ids[:5]]
            raise ValueError(
                f"Mapping identifiers are not unique in the genomic table "
                f"({len(duplicate_ids)} duplicated key(s), e.g. {examples}). "
                "Choose an unambiguous --genomic-id-column before joining."
            )

        selected[genomic_id_column] = selected[genomic_id_column].astype(str)
        genomic_rows = selected.rename(columns={genomic_id_column: "patient_id"})
        joined = mapping.merge(
            genomic_rows[["patient_id", *feature_columns]],
            on="patient_id",
            how="inner",
            validate="many_to_one",
        )
        matched_mapping_rows = len(joined)
        unmatched_mapping_rows = len(mapping) - matched_mapping_rows
        if unmatched_mapping_rows:
            LOGGER.warning(
                "Excluding %d mapping row(s) without a genomic record.",
                unmatched_mapping_rows,
            )

        encoded = encode_features(
            joined,
            feature_columns,
            vocabularies,
            numeric_columns,
            text_columns,
        )
        master = pd.concat(
            [
                joined[["patient_id", "image_filename", "label"]].reset_index(
                    drop=True
                ),
                encoded.reset_index(drop=True),
            ],
            axis=1,
        )

    if mapping_path is None:
        empty_features = encode_features(
            genomic.iloc[0:0],
            feature_columns,
            vocabularies,
            numeric_columns,
            text_columns,
        )
        master = pd.concat(
            [
                pd.DataFrame(columns=["patient_id", "image_filename", "label"]),
                empty_features.reset_index(drop=True),
            ],
            axis=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    summary = {
        "genomic_rows": len(genomic),
        "source_feature_columns": len(feature_columns),
        "encoded_feature_columns": len(master.columns) - 3,
        "image_files": image_count,
        "mapping_rows": len(mapping),
        "invalid_mapping_rows": invalid_mapping_rows,
        "unmatched_mapping_rows": unmatched_mapping_rows,
        "paired_samples": len(master),
        "unique_patients": int(master["patient_id"].nunique()),
        "label_counts": (
            master["label"].value_counts().sort_index().to_dict()
            if not master.empty
            else {}
        ),
        "output_path": str(output_path),
    }
    LOGGER.info("TB data preparation summary:")
    for key, value in summary.items():
        LOGGER.info("  %-26s %s", f"{key}:", value)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a cleaned TB image/genomics CSV. An explicit "
            "patient-to-image mapping is required to emit paired rows."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="TB raw data directory (default: data/raw/TB).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV (default: data/processed/tb_cleaned.csv).",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help=(
            "Explicit crosswalk with patient_id and image_filename columns; "
            "label is optional when the image is under health/sick/tb."
        ),
    )
    parser.add_argument(
        "--genomic-id-column",
        default="specimen_identifier",
        help=(
            "Genomic column matched against mapping patient_id "
            "(default: specimen_identifier)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the preparation pipeline and display its summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    prepare_tb_data(
        raw_root=args.raw_root,
        output_path=args.output,
        mapping_path=args.mapping_csv,
        genomic_id_column=args.genomic_id_column,
    )


if __name__ == "__main__":
    main()
