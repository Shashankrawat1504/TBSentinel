"""Training, evaluation, and checkpoint utilities for image classifiers."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassRecall,
)

LOGGER = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calculate focal loss averaged over a batch."""
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        focal_factor = (1.0 - probabilities).pow(self.gamma)
        per_class_loss = -focal_factor * log_probabilities
        if self.label_smoothing:
            per_class_loss = (
                1.0 - self.label_smoothing
            ) * per_class_loss + self.label_smoothing * per_class_loss.mean(
                dim=1, keepdim=True
            )
        target_loss = per_class_loss.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.class_weights is not None:
            target_loss = target_loss * self.class_weights[targets]
        return target_loss.mean()


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping configuration."""

    enabled: bool = True
    patience: int = 4
    min_delta: float = 0.001
    monitor: str = "macro_f1"  # Use macro_f1 for both binary and multi-class
    mode: str = "max"


class Trainer:
    """Run supervised image classification with AMP, metrics, and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        mixed_precision: bool = True,
        gradient_clip_norm: float | None = 1.0,
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        best_checkpoint_name: str = "tb_image_best.pt",
        last_checkpoint_name: str = "tb_image_last.pt",
        early_stopping: EarlyStoppingConfig | None = None,
        checkpoint_config: dict[str, Any] | None = None,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None.")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = mixed_precision and device.type == "cuda"
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.best_checkpoint_path = self.checkpoint_dir / best_checkpoint_name
        self.last_checkpoint_path = self.checkpoint_dir / last_checkpoint_name
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.checkpoint_config = checkpoint_config or {}
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        validation_loader: DataLoader[tuple[Tensor, Tensor]],
        epochs: int,
        class_names: tuple[str, ...] = ("non_tb", "tb"),
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train and validate, selecting the best checkpoint using configured metric.

        ``resume_from`` continues an interrupted run: model, optimizer, scheduler,
        and best-score bookkeeping are restored and training restarts at the next
        epoch. ``history`` then only covers the epochs run by this call.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        num_classes = len(class_names)
        LOGGER.info("Training with %d classes: %s", num_classes, class_names)

        self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_score: float | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        best_metrics: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        start_epoch = 1

        if resume_from is not None:
            (
                start_epoch,
                best_score,
                best_epoch,
                best_metrics,
                epochs_without_improvement,
            ) = self._restore_training_state(resume_from, epochs)

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(train_loader, class_names)
            validation_metrics = self.evaluate(validation_loader, class_names)
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation_metrics.items()},
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_metrics)

            # Log based on number of classes
            if num_classes == 2:
                LOGGER.info(
                    "Epoch %d/%d train_loss=%.4f val_loss=%.4f "
                    "acc=%.4f macro_f1=%.4f auc=%.4f sensitivity=%.4f "
                    "specificity=%.4f",
                    epoch,
                    epochs,
                    train_metrics["loss"],
                    validation_metrics["loss"],
                    validation_metrics["accuracy"],
                    validation_metrics["macro_f1"],
                    validation_metrics["auc"],
                    validation_metrics["sensitivity"],
                    validation_metrics["specificity"],
                )
            else:
                # Multi-class logging
                per_class_f1_str = ", ".join(
                    f"{name}: {validation_metrics['per_class_f1'][name]:.4f}"
                    for name in class_names
                )
                LOGGER.info(
                    "Epoch %d/%d train_loss=%.4f val_loss=%.4f "
                    "acc=%.4f macro_f1=%.4f per_class_f1=(%s)",
                    epoch,
                    epochs,
                    train_metrics["loss"],
                    validation_metrics["loss"],
                    validation_metrics["accuracy"],
                    validation_metrics["macro_f1"],
                    per_class_f1_str,
                )

            score = self._monitored_score(validation_metrics)
            improved = self._is_improvement(score, best_score)
            if improved:
                best_score = score
                best_metrics = validation_metrics
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            training_state = {
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "epochs_without_improvement": epochs_without_improvement,
            }
            if improved:
                self._save_checkpoint(
                    self.best_checkpoint_path,
                    epoch,
                    validation_metrics,
                    class_names,
                    training_state=training_state,
                )
            self._save_checkpoint(
                self.last_checkpoint_path,
                epoch,
                validation_metrics,
                class_names,
                training_state=training_state,
            )
            self._log_to_wandb(epoch_metrics, epoch)

            if (
                self.early_stopping.enabled
                and not improved
                and epochs_without_improvement >= self.early_stopping.patience
            ):
                LOGGER.info(
                    "Early stopping at epoch %d; %s did not improve for %d epochs.",
                    epoch,
                    self.early_stopping.monitor,
                    self.early_stopping.patience,
                )
                break

        if best_metrics is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")
        return {
            "best_epoch": int(
                self.load_best_epoch()
            ),
            "best_metrics": best_metrics,
            "history": history,
            "best_checkpoint": str(self.best_checkpoint_path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }

    def load_best_epoch(self) -> int:
        """Return the epoch recorded in the best checkpoint."""
        checkpoint = torch.load(
            self.best_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        return int(checkpoint["epoch"])

    def _restore_training_state(
        self,
        checkpoint_path: str | Path,
        epochs: int,
    ) -> tuple[int, float | None, int, dict[str, Any] | None, int]:
        """Restore a run and return the epoch to start from plus its bookkeeping."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Resume checkpoint stores no model weights: {path}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        completed_epoch = int(checkpoint.get("epoch", 0))
        if completed_epoch >= epochs:
            raise ValueError(
                f"Resume checkpoint already completed epoch {completed_epoch} "
                f"of {epochs}."
            )
        metrics = checkpoint.get("metrics")
        stored_score = checkpoint.get("best_score")
        if stored_score is None and isinstance(metrics, dict):
            # Checkpoints from before resume support only store the epoch metrics.
            stored_score = metrics.get(self.early_stopping.monitor)
        stored_best_metrics = checkpoint.get("best_metrics")
        if stored_best_metrics is None and isinstance(metrics, dict):
            stored_best_metrics = metrics
        best_epoch = int(checkpoint.get("best_epoch", completed_epoch))
        patience = int(checkpoint.get("epochs_without_improvement", 0))
        LOGGER.info(
            "Resuming %s: finished epoch %d, starting at epoch %d; best %s=%s "
            "at epoch %d, %d epoch(s) without improvement",
            path.name,
            completed_epoch,
            completed_epoch + 1,
            self.early_stopping.monitor,
            f"{float(stored_score):.4f}" if stored_score is not None else "unknown",
            best_epoch,
            patience,
        )
        return (
            completed_epoch + 1,
            float(stored_score) if stored_score is not None else None,
            best_epoch,
            stored_best_metrics if isinstance(stored_best_metrics, dict) else None,
            patience,
        )

    def evaluate(
        self,
        data_loader: DataLoader[tuple[Tensor, Tensor]],
        class_names: tuple[str, ...] = ("non_tb", "tb"),
    ) -> dict[str, Any]:
        """Evaluate loss and classification metrics on labeled rows."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_probabilities: list[Tensor] = []
        all_targets: list[Tensor] = []

        with torch.inference_mode():
            for images, targets in data_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, dtype=torch.long, non_blocking=True)
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                batch_size = targets.size(0)
                total_loss += float(loss.detach()) * batch_size
                total_samples += batch_size
                all_probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
                all_targets.append(targets.cpu())

        if total_samples == 0:
            raise ValueError("Cannot evaluate an empty validation loader.")
        return self._summarize_predictions(
            torch.cat(all_probabilities),
            torch.cat(all_targets),
            total_loss / total_samples,
            class_names,
        )

    @staticmethod
    def _summarize_predictions(
        probabilities: Tensor,
        targets: Tensor,
        mean_loss: float,
        class_names: tuple[str, ...],
    ) -> dict[str, Any]:
        """Calculate classification metrics (binary or multi-class)."""
        predictions = probabilities.argmax(dim=1)
        num_classes = probabilities.shape[1]

        # Common metrics
        accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")(
            predictions, targets
        )
        macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")(
            predictions, targets
        )
        confusion = MulticlassConfusionMatrix(num_classes=num_classes)(
            predictions, targets
        )

        confusion_values = confusion.tolist()

        # Binary-specific metrics
        if num_classes == 2:
            auc = BinaryAUROC()(probabilities[:, 1], targets)
            sensitivity = torch.sum((predictions == 1) & (targets == 1)) / torch.sum(
                targets == 1
            ).clamp_min(1)
            specificity = BinarySpecificity()(predictions, targets)

            return {
                "loss": mean_loss,
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "auc": float(auc),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "confusion_matrix": confusion_values,
                "confusion_matrix_labels": list(class_names),
                "samples": len(targets),
            }
        else:
            # Multi-class specific metrics
            per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)(
                predictions, targets
            )
            per_class_recall = MulticlassRecall(num_classes=num_classes, average=None)(
                predictions, targets
            )

            per_class_f1_dict = {
                class_names[i]: float(per_class_f1[i]) for i in range(num_classes)
            }
            per_class_recall_dict = {
                class_names[i]: float(per_class_recall[i])
                for i in range(num_classes)
            }

            return {
                "loss": mean_loss,
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "per_class_f1": per_class_f1_dict,
                "per_class_recall": per_class_recall_dict,
                "confusion_matrix": confusion_values,
                "confusion_matrix_labels": list(class_names),
                "samples": len(targets),
            }

    def _train_epoch(
        self,
        data_loader: DataLoader[tuple[Tensor, Tensor]],
        class_names: tuple[str, ...],
    ) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_probabilities: list[Tensor] = []
        all_targets: list[Tensor] = []
        for images, targets in data_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, dtype=torch.long, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_norm,
                    )
                self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            all_probabilities.append(
                torch.softmax(logits.detach().float(), dim=1).cpu()
            )
            all_targets.append(targets.detach().cpu())

        if total_samples == 0:
            raise ValueError("Cannot train on an empty training loader.")
        return self._summarize_predictions(
            torch.cat(all_probabilities),
            torch.cat(all_targets),
            total_loss / total_samples,
            class_names,
        )

    def _monitored_score(self, metrics: dict[str, Any]) -> float:
        monitor = self.early_stopping.monitor
        if monitor not in metrics:
            raise ValueError(
                f"Early-stopping metric {monitor!r} is unavailable. "
                f"Available metrics: {sorted(metrics)}"
            )
        score = float(metrics[monitor])
        if not math.isfinite(score):
            raise RuntimeError(
                f"Early-stopping metric {monitor!r} is not finite: {score}"
            )
        return score

    def _is_improvement(self, score: float, best_score: float | None) -> bool:
        if best_score is None:
            return True
        if self.early_stopping.mode == "max":
            return score > best_score + self.early_stopping.min_delta
        if self.early_stopping.mode == "min":
            return score < best_score - self.early_stopping.min_delta
        raise ValueError("Early-stopping mode must be 'min' or 'max'.")

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, Any],
        class_names: tuple[str, ...],
        training_state: dict[str, Any] | None = None,
    ) -> None:
        """Write a resumable model/optimizer/scheduler checkpoint atomically.

        The payload is flushed to disk before the atomic rename so a crash cannot
        leave a truncated checkpoint in place of the previous good one.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "metrics": metrics,
            "class_names": list(class_names),
            "config": self.checkpoint_config,
            **(training_state or {}),
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with open(temporary_path, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)

    @staticmethod
    def _log_to_wandb(metrics: dict[str, Any], step: int) -> None:
        """Log epoch metrics when a W&B run is active."""
        try:
            from src.utils.logger import log_metrics

            serializable = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float, str, bool))
            }
            confusion = metrics.get("val_confusion_matrix")
            labels = metrics.get("val_confusion_matrix_labels")
            if confusion is not None and labels is not None:
                size = len(labels)
                square = len(confusion) == size and all(
                    len(row) == size for row in confusion
                )
                if not square:
                    LOGGER.warning(
                        "Skipping confusion-matrix logging: shape %s does not match "
                        "%d labels.",
                        [len(row) for row in confusion],
                        size,
                    )
                else:
                    serializable.update(
                        {
                            f"val_confusion_{labels[row]}_as_{labels[column]}": int(
                                confusion[row][column]
                            )
                            for row in range(size)
                            for column in range(size)
                        }
                    )
            log_metrics(serializable, step=step)
        except ImportError:
            LOGGER.exception("Weights & Biases logging is unavailable.")
            raise
