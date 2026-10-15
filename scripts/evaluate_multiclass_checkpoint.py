"""Evaluate a multi-class Normal / TB / Pneumonia checkpoint on a held-out split.

This is the verification interface for the multi-class phase: it rebuilds the
image-only EfficientNet-B3 classifier from a saved checkpoint, runs inference on
a prepared split CSV, and reports accuracy, macro-F1, per-class precision,
recall and F1, a per-class one-vs-rest AUC, and the confusion matrix. The same
numbers are written to JSON so the reported metrics can be reproduced without
re-running training.

Example
-------
    python scripts/evaluate_multiclass_checkpoint.py \\
        --checkpoint artifacts/checkpoints/multiclass_efficientnet_b3_best.pt

By default the held-out ``multiclass_test.csv`` split is used; pass
``--csv data/processed/multiclass_val.csv`` to reproduce the validation numbers
that training recorded in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    MULTICLASS_LABEL_NAMES,
    TBXRayDataset,
    build_eval_transform,
)
from src.models.image_model import TBImageClassifier  # noqa: E402

LOGGER = logging.getLogger("evaluate_multiclass_checkpoint")

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "multiclass_efficientnet_b3_best.pt"
)
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "multiclass_test.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "outputs" / "multiclass_test_metrics.json"


def evaluate_checkpoint(
    checkpoint_path: Path,
    csv_path: Path,
    image_size: int = 300,
    batch_size: int = 16,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Run a multi-class checkpoint over one prepared split."""
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=False
    )
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    num_classes = int(model_config.get("num_classes", len(MULTICLASS_LABEL_NAMES)))
    class_names = tuple(
        checkpoint.get("class_names", list(MULTICLASS_LABEL_NAMES.values()))
    )
    if len(class_names) != num_classes:
        raise ValueError(
            f"Checkpoint declares {num_classes} classes but stores {len(class_names)} "
            f"names: {class_names}."
        )

    model = TBImageClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()

    dataset = TBXRayDataset(
        annotations=csv_path,
        transform=build_eval_transform(image_size=image_size),
        image_size=image_size,
        label_names={index: name for index, name in enumerate(class_names)},
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        drop_last=False,
    )

    criterion = torch.nn.CrossEntropyLoss()
    use_amp = resolved_device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(resolved_device, non_blocking=True)
            batch_targets = batch_targets.to(resolved_device, dtype=torch.long)
            with autocast(device_type=resolved_device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, batch_targets)
            total_loss += float(loss.detach()) * batch_targets.size(0)
            total_samples += batch_targets.size(0)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
            targets.append(batch_targets.cpu())

    probs = torch.cat(probabilities)
    tgts = torch.cat(targets)
    preds = probs.argmax(dim=1)

    per_class_precision = MulticlassPrecision(num_classes=num_classes, average=None)(
        preds, tgts
    )
    per_class_recall = MulticlassRecall(num_classes=num_classes, average=None)(
        preds, tgts
    )
    per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)(preds, tgts)
    per_class_auc = MulticlassAUROC(num_classes=num_classes, average=None)(probs, tgts)

    per_class = {
        class_names[index]: {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "auc_one_vs_rest": float(per_class_auc[index]),
            "support": int((tgts == index).sum()),
        }
        for index in range(num_classes)
    }

    return {
        "checkpoint": str(checkpoint_path),
        "split_csv": str(csv_path),
        "samples": total_samples,
        "class_names": list(class_names),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loss_unweighted_ce": total_loss / max(total_samples, 1),
        "accuracy": float(
            MulticlassAccuracy(num_classes=num_classes, average="micro")(preds, tgts)
        ),
        "macro_f1": float(
            MulticlassF1Score(num_classes=num_classes, average="macro")(preds, tgts)
        ),
        "macro_auc_one_vs_rest": float(
            MulticlassAUROC(num_classes=num_classes, average="macro")(probs, tgts)
        ),
        "per_class": per_class,
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes)(
            preds, tgts
        ).tolist(),
        "stored_validation_metrics": checkpoint.get("metrics"),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a multi-class checkpoint on a prepared split."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print metrics without writing the JSON artifact.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the requested checkpoint and report the metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        csv_path=csv_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Checkpoint:        {results['checkpoint']}")
    print(f"Checkpoint epoch:  {results['checkpoint_epoch']}")
    print(f"Split:             {results['split_csv']}")
    print(f"Samples:           {results['samples']}")
    print("")
    print(f"accuracy (micro):  {results['accuracy']:.4f}")
    print(f"macro-F1:          {results['macro_f1']:.4f}")
    print(f"macro AUC (OvR):   {results['macro_auc_one_vs_rest']:.4f}")
    print(f"loss (plain CE):   {results['loss_unweighted_ce']:.4f}")
    print("")
    per_class = cast(dict[str, dict[str, float]], results["per_class"])
    class_names = cast(list[str], results["class_names"])
    confusion_matrix = cast(list[list[int]], results["confusion_matrix"])

    print(f"{'class':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'auc':>10}{'n':>7}")
    for class_name, values in per_class.items():
        print(
            f"{class_name:<12}{values['precision']:>11.4f}{values['recall']:>10.4f}"
            f"{values['f1']:>10.4f}{values['auc_one_vs_rest']:>10.4f}"
            f"{values['support']:>7}"
        )
    print("")
    print(f"Confusion matrix rows=true {class_names}:")
    for class_name, row in zip(class_names, confusion_matrix):
        print(f"  {class_name:<12}{row}")

    if not args.no_write:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
