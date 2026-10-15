"""Build supervised image-only CSVs from official TBX11K split lists."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("prepare_tb_image_splits")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "TB" / "chest x ray"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
KNOWN_LABELS = {
    "health": (0, "non_tb"),
    "sick": (0, "non_tb"),
    "tb": (1, "tb"),
}
SPLIT_LISTS = {
    "train": "TBX11K_train.txt",
    "val": "TBX11K_val.txt",
    "test": "all_test.txt",
}
OUTPUT_COLUMNS = ["image_path", "label", "label_name", "split"]


def load_split(
    split_name: str,
    list_filename: str,
    data_root: Path,
) -> pd.DataFrame:
    """Load, validate, and label a single official image-list split."""
    list_path = data_root / "lists" / list_filename
    image_root = data_root / "imgs"
    if not list_path.is_file():
        raise FileNotFoundError(f"Official split list is missing: {list_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory is missing: {image_root}")

    entries = [
        line.strip().replace("\\", "/")
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != len(set(entries)):
        raise ValueError(f"{list_path.name} contains duplicate image paths.")

    records: list[dict[str, object]] = []
    for entry in entries:
        relative_image = Path(entry)
        if relative_image.is_absolute() or ".." in relative_image.parts:
            raise ValueError(f"Unsafe image path in {list_path.name}: {entry!r}")
        image_path = (image_root / relative_image).resolve()
        try:
            image_path.relative_to(image_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Image path escapes image root: {entry!r}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image listed in {list_path.name} is missing: {image_path}"
            )

        category = relative_image.parts[0].casefold()
        if split_name == "test":
            label: int | None = None
            label_name = "unknown"
        else:
            try:
                label, label_name = KNOWN_LABELS[category]
            except KeyError as exc:
                raise ValueError(
                    f"Unlabeled or unsupported category {category!r} in "
                    f"supervised split {list_path.name}."
                ) from exc

        records.append(
            {
                "image_path": image_path.relative_to(image_root).as_posix(),
                "label": label,
                "label_name": label_name,
                "split": split_name,
            }
        )

    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def prepare_image_splits(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Create train/validation/test CSVs and return their contents."""
    data_root = data_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    frames = {
        split: load_split(split, filename, data_root)
        for split, filename in SPLIT_LISTS.items()
    }
    image_sets = {split: set(frame["image_path"]) for split, frame in frames.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = image_sets[left] & image_sets[right]
        if overlap:
            raise ValueError(
                f"Official {left}/{right} splits overlap on "
                f"{len(overlap)} image(s), e.g. {sorted(overlap)[:3]}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        output_path = output_dir / f"tb_{split}.csv"
        frame.to_csv(output_path, index=False)
        LOGGER.info("Wrote %d rows to %s", len(frame), output_path)
        LOGGER.info(
            "%s label counts: %s",
            split,
            dict(Counter(frame["label_name"])),
        )
    return frames


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare TBX11K image-only train, validation, and test CSVs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="TBX11K chest X-ray root containing lists/ and imgs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tb_train.csv, tb_val.csv, and tb_test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Run split preparation and print a concise summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    frames = prepare_image_splits(args.data_root, args.output_dir)
    print("TBX11K image split summary")
    for split, frame in frames.items():
        labeled = frame.dropna(subset=["label"])
        counts = frame["label_name"].value_counts().to_dict()
        print(
            f"  {split:5s}: {len(frame):5,d} images; "
            f"labeled={len(labeled):5,d}; classes={counts}"
        )


if __name__ == "__main__":
    main()
