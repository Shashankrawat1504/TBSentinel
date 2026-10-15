"""Hydra entry point for the centralized image-only TB baseline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from src.data.dataset import (
    TBXRayDataset,
    build_eval_transform,
    build_train_transform,
)
from src.models.image_model import TBImageClassifier
from src.training.trainer import EarlyStoppingConfig, FocalLoss, Trainer
from src.utils.logger import finish_wandb, get_logger, init_wandb
from src.utils.seed import make_dataloader, set_seed

LOGGER = logging.getLogger("train_centralized")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve configured relative paths from project root, not Hydra output."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def calculate_class_weights(
    labels: Tensor,
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Return inverse-frequency class weights normalized to mean one."""
    counts = torch.bincount(labels.to(dtype=torch.long), minlength=num_classes)
    if torch.any(counts == 0):
        raise ValueError(
            f"Training data must contain all {num_classes} classes; "
            f"observed counts={counts.tolist()}."
        )
    weights = labels.numel() / (num_classes * counts.float())
    return (weights / weights.mean()).to(device=device)


def create_optimizer(
    model: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """Create the configured optimizer from trainable parameters."""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")
    if name.casefold() == "adamw":
        return AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer {name!r}; supported: adamw.")


def create_scheduler(
    optimizer: Optimizer,
    name: str,
    epochs: int,
    min_learning_rate: float,
) -> LRScheduler:
    """Create the configured learning-rate schedule."""
    if name.casefold() == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler {name!r}; supported: cosine.")


def create_criterion(
    config: DictConfig,
    train_dataset: TBXRayDataset,
    device: torch.device,
) -> nn.Module:
    """Build the configured weighted cross-entropy or focal loss."""
    class_config = config.train.class_imbalance
    class_weights: Tensor | None = None
    if str(class_config.class_weights).casefold() == "auto":
        labels = torch.as_tensor(train_dataset.annotations["label"].to_numpy(copy=True))
        class_weights = calculate_class_weights(
            labels,
            num_classes=int(config.model.num_classes),
            device=device,
        )
        LOGGER.info("Using inverse-frequency class weights: %s", class_weights.tolist())
    elif class_config.class_weights not in (None, False, "none"):
        class_weights = torch.as_tensor(
            class_config.class_weights,
            dtype=torch.float32,
            device=device,
        )

    loss_name = str(class_config.strategy).casefold()
    if loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.train.loss.label_smoothing),
        )
    if loss_name == "focal":
        return FocalLoss(
            gamma=float(class_config.focal_gamma),
            class_weights=class_weights,
            label_smoothing=float(config.train.loss.label_smoothing),
        )
    raise ValueError(
        f"Unsupported loss {loss_name!r}; supported: weighted_cross_entropy, focal."
    )


def build_model(config: DictConfig) -> TBImageClassifier:
    """Instantiate the image classifier and optional configured fusion branch."""
    multimodal_config = config.model.multimodal
    tabular_count = (
        int(multimodal_config.num_tabular_features)
        if bool(multimodal_config.enabled)
        else 0
    )
    if tabular_count:
        raise ValueError(
            "Tabular model inputs are not enabled for the current experiment. "
            "Verified image/genomic pairs are unavailable."
        )
    return TBImageClassifier(
        num_classes=int(config.model.num_classes),
        pretrained=bool(config.model.pretrained),
        freeze_backbone=bool(config.model.freeze_backbone),
        dropout=float(config.model.dropout),
        num_tabular_features=tabular_count,
        tabular_hidden_dims=tuple(
            int(dimension) for dimension in multimodal_config.hidden_dims
        ),
        fusion_dim=int(multimodal_config.fusion_dim),
        fusion_dropout=float(multimodal_config.fusion_dropout),
    )


def build_data_loaders(
    config: DictConfig,
    seed: int,
) -> tuple[
    TBXRayDataset,
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """Load only the official train and validation split CSVs."""
    image_root = resolve_path(config.data.image_root)
    train_csv = resolve_path(config.data.train_csv)
    validation_csv = resolve_path(config.data.val_csv)
    if not train_csv.is_file() or not validation_csv.is_file():
        raise FileNotFoundError(
            "Official train/validation CSVs are missing. Run "
            "scripts/prepare_tb_image_splits.py first. "
            f"train={train_csv}, val={validation_csv}"
        )

    image_size = int(config.model.image_size)
    train_dataset = TBXRayDataset(
        train_csv,
        image_root=image_root,
        transform=build_train_transform(image_size, seed=seed),
    )
    validation_dataset = TBXRayDataset(
        validation_csv,
        image_root=image_root,
        transform=build_eval_transform(image_size),
    )
    if train_dataset.annotations["label"].nunique() != int(config.model.num_classes):
        raise ValueError(
            "Official training split does not contain every configured class."
        )
    if validation_dataset.annotations["label"].nunique() != int(
        config.model.num_classes
    ):
        raise ValueError(
            "Official validation split does not contain every configured class."
        )
    if not train_dataset.annotations["split"].eq("train").all():
        raise ValueError("Training CSV contains rows outside the official train split.")
    if not validation_dataset.annotations["split"].eq("val").all():
        raise ValueError("Validation CSV contains rows outside the official val split.")

    batch_size = int(config.train.batch_size)
    num_workers = int(config.train.num_workers)
    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=bool(config.train.pin_memory),
        seed=seed,
    )
    validation_loader = make_dataloader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=bool(config.train.pin_memory),
        seed=seed + 1,
    )
    LOGGER.info(
        "Using official splits only: train=%d %s; val=%d %s; test is not loaded.",
        len(train_dataset),
        train_dataset.class_distribution,
        len(validation_dataset),
        validation_dataset.class_distribution,
    )
    return train_dataset, train_loader, validation_loader


def run_training(config: DictConfig) -> dict[str, Any]:
    """Run one reproducible centralized baseline experiment."""
    seed = int(config.train.seed)
    set_seed(
        seed=seed,
        deterministic=bool(config.reproducibility.deterministic),
    )
    device_name = str(config.device).casefold()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    LOGGER.info("Training device: %s", device)

    train_dataset, train_loader, validation_loader = build_data_loaders(config, seed)
    model = build_model(config).to(device)
    optimizer = create_optimizer(
        model,
        name=str(config.train.optimizer),
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    scheduler = create_scheduler(
        optimizer,
        name=str(config.train.scheduler.name),
        epochs=int(config.train.epochs),
        min_learning_rate=float(config.train.scheduler.min_learning_rate),
    )
    criterion = create_criterion(config, train_dataset, device).to(device)

    early_config = config.train.early_stopping
    checkpoint_config = config.train.checkpoint
    checkpoint_dir = resolve_path(checkpoint_config.directory)
    serialized_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(serialized_config, dict):
        raise TypeError("Resolved Hydra configuration must be a mapping.")
    class_names = config.data.training.label_names
    if len(class_names) != int(config.model.num_classes):
        raise ValueError(
            "Configured label_names must match model.num_classes: "
            f"{len(class_names)} != {config.model.num_classes}."
        )
    binary_class_names = (str(class_names[0]), str(class_names[1]))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        mixed_precision=bool(config.train.mixed_precision),
        gradient_clip_norm=float(config.train.gradient_clip_norm),
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint_name=str(checkpoint_config.best_filename),
        last_checkpoint_name=str(checkpoint_config.last_filename),
        early_stopping=EarlyStoppingConfig(
            enabled=bool(early_config.enabled),
            patience=int(early_config.patience),
            min_delta=float(early_config.min_delta),
            monitor=str(early_config.monitor),
            mode=str(early_config.mode),
        ),
        checkpoint_config={str(key): value for key, value in serialized_config.items()},
    )

    run = init_wandb(
        config,
        job_type="centralized-image-baseline",
        enabled=bool(config.train.wandb.enabled),
        mode=str(config.train.wandb.mode),
    )
    try:
        result = trainer.fit(
            train_loader,
            validation_loader,
            epochs=int(config.train.epochs),
            class_names=binary_class_names,
            resume_from=config.train.get("resume_from", None),
        )
        best_metrics = result["best_metrics"]
        LOGGER.info(
            "Best validation metrics (epoch %d): accuracy=%.4f "
            "macro_f1=%.4f auc=%.4f sensitivity=%.4f specificity=%.4f",
            result["best_epoch"],
            best_metrics["accuracy"],
            best_metrics["macro_f1"],
            best_metrics["auc"],
            best_metrics["sensitivity"],
            best_metrics["specificity"],
        )
        print("Best centralized TB baseline validation metrics")
        print(f"  best epoch:    {result['best_epoch']}")
        print(f"  accuracy:      {best_metrics['accuracy']:.4f}")
        print(f"  macro-F1:      {best_metrics['macro_f1']:.4f}")
        print(f"  AUC-ROC:       {best_metrics['auc']:.4f}")
        print(f"  sensitivity:   {best_metrics['sensitivity']:.4f}")
        print(f"  specificity:   {best_metrics['specificity']:.4f}")
        print(f"  confusion:     {best_metrics['confusion_matrix']}")
        print(f"  checkpoint:    {result['best_checkpoint']}")
        if run is not None:
            run.summary["best_epoch"] = result["best_epoch"]
            run.summary["best_validation_metrics"] = best_metrics
            run.summary["best_checkpoint"] = result["best_checkpoint"]
        return result
    finally:
        finish_wandb()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(config: DictConfig) -> None:
    """Hydra-compatible centralized training command."""
    global LOGGER
    LOGGER = get_logger(__name__)
    get_logger("src.training.trainer")
    run_training(config)


if __name__ == "__main__":
    main()
