from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.prepare_tb_data import (
    image_label_from_path,
    normalize_column_name,
    parse_binary_label,
    prepare_tb_data,
)


def make_raw_dataset(root: Path) -> tuple[Path, Path]:
    genomic_dir = root / "Genomic"
    image_root = root / "chest x ray" / "imgs"
    genomic_dir.mkdir(parents=True)
    (image_root / "tb").mkdir(parents=True)
    (image_root / "health").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_root / "tb" / "tb1.png")
    Image.new("RGB", (8, 8)).save(image_root / "health" / "h1.png")

    genomic_path = genomic_dir / "TB_Portals_Genomics_March_2026.csv"
    pd.DataFrame(
        [
            {
                "Specimen Identifier": "sample-1",
                "condition_id": "condition-1",
                "sex": "Male",
                "age_of_onset": "26",
                "gene_snp_mutations": "g.123A>G",
                "num_other_variants": "2",
            },
            {
                "Specimen Identifier": "sample-2",
                "condition_id": "condition-2",
                "sex": "Female",
                "age_of_onset": None,
                "gene_snp_mutations": None,
                "num_other_variants": "4",
            },
        ]
    ).to_csv(genomic_path, index=False)
    return root, image_root


def test_normalize_column_name() -> None:
    assert normalize_column_name("  Specimen Identifier ") == "specimen_identifier"
    assert normalize_column_name("Age of Onset (Years)") == "age_of_onset_years"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("TB", 1),
        ("active_tb", 1),
        ("positive", 1),
        ("non-TB", 0),
        ("healthy", 0),
        ("0", 0),
    ],
)
def test_parse_binary_label(value: str, expected: int) -> None:
    assert parse_binary_label(value) == expected


def test_image_label_uses_known_folder_categories(tmp_path: Path) -> None:
    image_root = tmp_path / "imgs"
    assert image_label_from_path(image_root / "tb" / "tb1.png", image_root) == 1
    assert image_label_from_path(image_root / "sick" / "s1.png", image_root) == 0
    assert (
        image_label_from_path(image_root / "test" / "unknown.png", image_root) is None
    )


def test_prepare_tb_data_joins_only_valid_mapped_rows(tmp_path: Path) -> None:
    raw_root, _ = make_raw_dataset(tmp_path / "raw")
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        [
            {"patient_id": "sample-1", "image_filename": "tb/tb1.png"},
            {"patient_id": "sample-2", "image_filename": "health/h1.png"},
            {"patient_id": "missing", "image_filename": "tb/not-found.png"},
        ]
    ).to_csv(mapping_path, index=False)

    output_path = tmp_path / "processed" / "tb_cleaned.csv"
    summary = prepare_tb_data(
        raw_root=raw_root,
        output_path=output_path,
        mapping_path=mapping_path,
    )
    cleaned = pd.read_csv(output_path)

    assert summary["paired_samples"] == 2
    assert summary["invalid_mapping_rows"] == 1
    assert list(cleaned["patient_id"]) == ["sample-1", "sample-2"]
    assert list(cleaned["label"]) == [1, 0]
    assert list(cleaned["image_filename"]) == ["tb/tb1.png", "health/h1.png"]
    assert cleaned["age_of_onset"].tolist() == [26.0, 26.0]
    assert "sex_female" in cleaned.columns
    assert "sex_male" in cleaned.columns
    assert "specimen_identifier" not in cleaned.columns


def test_missing_crosswalk_writes_schema_without_fabricated_pairs(
    tmp_path: Path,
) -> None:
    raw_root, _ = make_raw_dataset(tmp_path / "raw")
    output_path = tmp_path / "processed" / "tb_cleaned.csv"

    summary = prepare_tb_data(raw_root=raw_root, output_path=output_path)
    cleaned = pd.read_csv(output_path)

    assert summary["paired_samples"] == 0
    assert cleaned.empty
    assert list(cleaned.columns[:3]) == [
        "patient_id",
        "image_filename",
        "label",
    ]


def test_mapping_label_must_agree_with_known_image_folder(
    tmp_path: Path,
) -> None:
    raw_root, _ = make_raw_dataset(tmp_path / "raw")
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        [{"patient_id": "sample-1", "image_filename": "tb/tb1.png", "label": 0}]
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="conflicts"):
        prepare_tb_data(
            raw_root=raw_root,
            output_path=tmp_path / "processed" / "tb_cleaned.csv",
            mapping_path=mapping_path,
        )
