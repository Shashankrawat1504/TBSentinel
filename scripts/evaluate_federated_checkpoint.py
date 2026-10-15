"""Re-evaluate a saved federated global checkpoint on the fixed validation split.

This is the verification interface for the federated phase: it loads a global
checkpoint, rebuilds the image-only EfficientNet-B3 classifier, runs inference on
the official 1,800-image validation split, and recomputes the same metrics that
the Flower global evaluation callback recorded. Reported metrics are therefore
reproducible without re-running the simulation.

Example
-------
    python scripts/evaluate_federated_checkpoint.py \
        --checkpoint artifacts/checkpoints/tb_federated_fedavg_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassF1Score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    TBXRayDataset,
    build_eval_transform,
)
from src.models.image_model import TBImageClassifier  # noqa: E402

LOGGER = logging.getLogger("evaluate_federated_checkpoint")

DEFAULT_VAL_CSV = PROJECT_ROOT / "data" / "processed" / "tb_val.csv"
METRIC_KEYS = (
    "loss",
    "accuracy",
    "macro_f1",
    "auc",
    "sensitivity",
    "specificity",
)


def as_float_map(raw: object) -> dict[str, float]:
    """Return the numeric entries of a mapping as floats, ignoring anything else."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[str(key)] = float(value)
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    val_csv: Path = DEFAULT_VAL_CSV,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    image_size: int = 300,
    batch_size: int = 16,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Run the checkpoint's weights over the global validation split."""
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=False
    )

    model = TBImageClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()

    dataset = TBXRayDataset(
        annotations=val_csv,
        image_root=image_root,
        transform=build_eval_transform(image_size=image_size),
        image_size=image_size,
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

    metrics: dict[str, object] = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(MulticlassAccuracy(num_classes=2, average="micro")(preds, tgts)),
        "macro_f1": float(MulticlassF1Score(num_classes=2, average="macro")(preds, tgts)),
        "auc": float(BinaryAUROC()(probs[:, 1], tgts)),
        "sensitivity": float(
            torch.sum((preds == 1) & (tgts == 1)) / torch.sum(tgts == 1).clamp_min(1)
        ),
        "specificity": float(BinarySpecificity()(preds, tgts)),
        "confusion_matrix": BinaryConfusionMatrix()(preds, tgts).tolist(),
        "samples": total_samples,
        "round": checkpoint.get("round"),
        "stored_metrics": checkpoint.get("metrics"),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Verify a federated global checkpoint on the validation split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Evaluate the requested checkpoint and report stored vs recomputed metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    val_csv = args.val_csv.expanduser().resolve()
    if not val_csv.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        val_csv=val_csv,
        image_root=args.image_root.expanduser().resolve(),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    stored_metrics = as_float_map(results.pop("stored_metrics", None))
    recomputed_metrics = as_float_map(results)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected round: {results['round']}")
    print(f"Validation samples: {results['samples']}")
    print("")
    print(f"{'metric':<14}{'recomputed':>12}{'stored':>12}{'delta':>12}")
    for key in METRIC_KEYS:
        recomputed = recomputed_metrics.get(key)
        stored_value = stored_metrics.get(key)
        if recomputed is None:
            print(f"{key:<14}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
            continue
        if stored_value is None:
            print(f"{key:<14}{recomputed:>12.4f}{'n/a':>12}{'n/a':>12}")
            continue
        delta = recomputed - stored_value
        print(f"{key:<14}{recomputed:>12.4f}{stored_value:>12.4f}{delta:>+12.4f}")
    print("")
    print(f"Confusion matrix (non_tb, tb): {results['confusion_matrix']}")


if __name__ == "__main__":
    main()
