from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from scripts.create_federated_splits import partition_by_dirichlet
from scripts.prepare_tb_image_splits import prepare_image_splits
from src.data.dataset import (
    MultimodalTBDataset,
    TBXRayDataset,
    build_eval_transform,
    build_train_transform,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 24), color).save(path)


def test_prepare_image_splits_uses_official_lists_and_keeps_test_unlabeled(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "chest x ray"
    image_root = data_root / "imgs"
    (data_root / "lists").mkdir(parents=True)
    write_image(image_root / "tb" / "tb1.png", (255, 0, 0))
    write_image(image_root / "health" / "h1.png", (0, 255, 0))
    write_image(image_root / "sick" / "s1.png", (0, 0, 255))
    write_image(image_root / "test" / "unknown_1.png", (20, 20, 20))

    lists = {
        "TBX11K_train.txt": ["tb/tb1.png", "health/h1.png"],
        "TBX11K_val.txt": ["sick/s1.png"],
        "all_test.txt": ["test/unknown_1.png"],
    }
    for filename, entries in lists.items():
        (data_root / "lists" / filename).write_text(
            "\n".join(entries), encoding="utf-8"
        )

    frames = prepare_image_splits(
        data_root=data_root,
        output_dir=tmp_path / "processed",
    )

    assert frames["train"]["label"].tolist() == [1, 0]
    assert frames["train"]["label_name"].tolist() == ["tb", "non_tb"]
    assert frames["val"]["label"].tolist() == [0]
    assert frames["test"]["label"].isna().all()
    assert frames["test"]["label_name"].tolist() == ["unknown"]
    assert frames["test"]["split"].tolist() == ["test"]
    assert (tmp_path / "processed" / "tb_train.csv").is_file()
    assert (tmp_path / "processed" / "tb_val.csv").is_file()
    assert (tmp_path / "processed" / "tb_test.csv").is_file()


def test_prepare_image_splits_rejects_extra_images_in_supervised_list(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "chest x ray"
    image_root = data_root / "imgs"
    (data_root / "lists").mkdir(parents=True)
    write_image(image_root / "extra" / "unlabeled.png", (10, 10, 10))
    (data_root / "lists" / "TBX11K_train.txt").write_text(
        "extra/unlabeled.png", encoding="utf-8"
    )
    (data_root / "lists" / "TBX11K_val.txt").write_text("", encoding="utf-8")
    (data_root / "lists" / "all_test.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Unlabeled or unsupported category"):
        prepare_image_splits(data_root, tmp_path / "processed")


def make_labeled_csv(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "imgs"
    records = []
    for label, folder, color in [(0, "health", (0, 255, 0)), (1, "tb", (255, 0, 0))]:
        for index in range(6):
            relative_path = f"{folder}/{folder}_{index}.png"
            write_image(image_root / relative_path, color)
            records.append(
                {
                    "image_path": relative_path,
                    "label": label,
                    "label_name": "non_tb" if label == 0 else "tb",
                    "split": "train",
                }
            )
    csv_path = tmp_path / "tb_train.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path, image_root


def test_tbxray_dataset_returns_tensor_and_integer_label(tmp_path: Path) -> None:
    csv_path, image_root = make_labeled_csv(tmp_path)
    dataset = TBXRayDataset(
        csv_path,
        image_root=image_root,
        transform=build_eval_transform(image_size=32),
    )

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label.ndim == 0
    assert label.dtype == torch.long
    assert dataset.class_distribution == {"non_tb": 6, "tb": 6}


def test_training_and_eval_transforms_are_available(tmp_path: Path) -> None:
    csv_path, image_root = make_labeled_csv(tmp_path)
    train_dataset = TBXRayDataset(
        csv_path, image_root=image_root, transform=build_train_transform(32)
    )
    eval_dataset = TBXRayDataset(
        csv_path, image_root=image_root, transform=build_eval_transform(32)
    )

    assert train_dataset[0][0].shape == eval_dataset[0][0].shape == (3, 32, 32)


def test_multimodal_dataset_preserves_image_only_and_optional_tabular_apis(
    tmp_path: Path,
) -> None:
    csv_path, image_root = make_labeled_csv(tmp_path)
    image_only = MultimodalTBDataset(
        csv_path, image_root=image_root, transform=build_eval_transform(32)
    )
    with_tabular = MultimodalTBDataset(
        csv_path,
        image_root=image_root,
        transform=build_eval_transform(32),
        tabular_features=np.ones((12, 3), dtype=np.float32),
    )

    image, label = image_only[0]
    paired_image, features, paired_label = with_tabular[0]

    assert image.shape == paired_image.shape == (3, 32, 32)
    assert label.item() == paired_label.item()
    assert features.shape == (3,)
    assert features.dtype == torch.float32


def test_multimodal_dataset_rejects_misaligned_tabular_features(
    tmp_path: Path,
) -> None:
    csv_path, image_root = make_labeled_csv(tmp_path)
    with pytest.raises(ValueError, match="row count must match"):
        MultimodalTBDataset(
            csv_path,
            image_root=image_root,
            tabular_features=np.ones((2, 3), dtype=np.float32),
        )


def test_dirichlet_clients_partition_training_rows_deterministically() -> None:
    frame = pd.DataFrame(
        {
            "image_path": [f"image_{index}.png" for index in range(200)],
            "label": [0] * 100 + [1] * 100,
            "label_name": ["non_tb"] * 100 + ["tb"] * 100,
            "split": ["train"] * 200,
        }
    )

    first = partition_by_dirichlet(frame, alpha=0.5, seed=123)
    second = partition_by_dirichlet(frame, alpha=0.5, seed=123)
    first_paths = [path for client in first for path in client["image_path"]]

    assert len(first) == 3
    assert all(not client.empty for client in first)
    assert len(first_paths) == len(set(first_paths)) == len(frame)
    assert set(first_paths) == set(frame["image_path"])
    for first_client, second_client in zip(first, second, strict=True):
        assert (
            first_client["image_path"].tolist() == second_client["image_path"].tolist()
        )


@pytest.mark.parametrize(
    ("alpha", "num_clients"),
    [(0.0, 3), (-1.0, 3), (0.5, 1)],
)
def test_dirichlet_partition_rejects_invalid_parameters(
    alpha: float, num_clients: int
) -> None:
    frame = pd.DataFrame(
        {
            "image_path": ["image.png"],
            "label": [0],
            "label_name": ["non_tb"],
            "split": ["train"],
        }
    )
    with pytest.raises(ValueError):
        partition_by_dirichlet(frame, alpha=alpha, num_clients=num_clients)
