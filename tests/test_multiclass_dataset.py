"""Tests for the multi-class Normal / TB / Pneumonia dataset builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.create_multiclass_dataset import (
    MULTICLASS_LABELS,
    assign_rsna_labels,
    split_tb_images,
    validate_dataset,
)


def make_split_frame(paths: list[str], labels: list[int]) -> pd.DataFrame:
    disease = {
        MULTICLASS_LABELS["normal"]: "normal",
        MULTICLASS_LABELS["tb"]: "tuberculosis",
        MULTICLASS_LABELS["pneumonia"]: "pneumonia",
    }
    return pd.DataFrame(
        {
            "image_path": paths,
            "label": labels,
            "disease_type": [disease[label] for label in labels],
        }
    )


def make_valid_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": make_split_frame(["a.png", "b.png", "c.png"], [0, 1, 2]),
        "val": make_split_frame(["d.png", "e.png", "f.png"], [0, 1, 2]),
        "test": make_split_frame(["g.png", "h.png", "i.png"], [2, 1, 0]),
    }


def test_assign_rsna_labels_maps_classes_to_contract() -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["normal.png", "opacity.png"],
            "class": ["Normal", "Lung Opacity"],
        }
    )

    labeled = assign_rsna_labels(frame)

    assert labeled["label"].tolist() == [
        MULTICLASS_LABELS["normal"],
        MULTICLASS_LABELS["pneumonia"],
    ]
    assert labeled["disease_type"].tolist() == ["normal", "pneumonia"]
    assert list(labeled.columns) == ["image_path", "label", "disease_type"]


def test_assign_rsna_labels_rejects_unknown_class() -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["other.png"],
            "class": ["No Lung Opacity / Not Normal"],
        }
    )

    with pytest.raises(ValueError, match="Unexpected RSNA classes"):
        assign_rsna_labels(frame)


def test_split_tb_images_is_disjoint_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "image_path": [f"tb{i:04d}.png" for i in range(800)],
            "label": [MULTICLASS_LABELS["tb"]] * 800,
            "disease_type": ["tuberculosis"] * 800,
        }
    )

    splits = split_tb_images(frame)
    repeated = split_tb_images(frame)

    sizes = {split: len(rows) for split, rows in splits.items()}
    assert sizes == {"train": 560, "val": 120, "test": 120}
    train_paths = set(splits["train"]["image_path"])
    assert not train_paths.intersection(splits["val"]["image_path"])
    assert not train_paths.intersection(splits["test"]["image_path"])
    assert not set(splits["val"]["image_path"]).intersection(
        splits["test"]["image_path"]
    )
    # The same seed must reproduce the same partition.
    assert splits["train"]["image_path"].tolist() == repeated["train"][
        "image_path"
    ].tolist()


def test_validate_dataset_accepts_valid_splits() -> None:
    validate_dataset(make_valid_splits(), check_paths=False)


def test_validate_dataset_rejects_train_val_leakage() -> None:
    splits = make_valid_splits()
    splits["val"] = make_split_frame(["a.png", "b.png", "c.png"], [0, 1, 2])

    with pytest.raises(ValueError, match="appear in both train and val"):
        validate_dataset(splits, check_paths=False)


def test_validate_dataset_rejects_duplicate_rows() -> None:
    splits = make_valid_splits()
    splits["train"] = make_split_frame(["a.png", "a.png", "c.png"], [0, 1, 2])

    with pytest.raises(ValueError, match="duplicate image paths"):
        validate_dataset(splits, check_paths=False)


def test_validate_dataset_rejects_missing_class() -> None:
    splits = make_valid_splits()
    splits["test"] = make_split_frame(["g.png", "h.png", "i.png"], [0, 1, 1])

    with pytest.raises(ValueError, match="do not match"):
        validate_dataset(splits, check_paths=False)


def test_validate_dataset_rejects_label_disease_mismatch() -> None:
    splits = make_valid_splits()
    # Two rows share label 1 while only one keeps the TB disease type.
    frame = make_split_frame(["a.png", "b.png", "x.png", "c.png"], [0, 1, 1, 2])
    frame.loc[2, "disease_type"] = "normal"
    splits["train"] = frame

    with pytest.raises(ValueError, match="maps one label to several diseases"):
        validate_dataset(splits, check_paths=False)


def test_validate_dataset_reports_missing_image_files(tmp_path: Path) -> None:
    created: dict[str, str] = {}
    for name in (
        "a.png",
        "b.png",
        "c.png",
        "d.png",
        "e.png",
        "f.png",
        "g.png",
        "h.png",
        "i.png",
    ):
        path = tmp_path / name
        Image.new("RGB", (8, 8)).save(path)
        created[name] = str(path)
    splits = {
        "train": make_split_frame(
            [created["a.png"], created["b.png"], created["c.png"]], [0, 1, 2]
        ),
        "val": make_split_frame(
            [str(tmp_path / "missing.png"), created["d.png"], created["e.png"]],
            [0, 1, 2],
        ),
        "test": make_split_frame(
            [created["f.png"], created["g.png"], created["h.png"]], [0, 1, 2]
        ),
    }

    with pytest.raises(FileNotFoundError, match="missing files"):
        validate_dataset(splits, check_paths=True)

    # Swapping the missing file for a real one must clear the check.
    splits["val"] = make_split_frame(
        [created["d.png"], created["e.png"], created["i.png"]], [0, 1, 2]
    )
    validate_dataset(splits, check_paths=True)
