"""Federated aggregation strategies and global evaluation hooks for Flower."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.strategy import FedAvg, FedProx, Strategy
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassF1Score,
)

from src.federation.client import set_parameters_to_model
from src.models.image_model import TBImageClassifier

LOGGER = logging.getLogger(__name__)


def aggregate_fit_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    """Aggregate local training metrics across clients weighted by sample count."""
    if not metrics:
        return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_loss = sum(num_samples * float(m.get("train_loss", 0.0)) for num_samples, m in metrics)
    weighted_acc = sum(num_samples * float(m.get("train_acc", 0.0)) for num_samples, m in metrics)

    return {
        "train_loss": weighted_loss / total_samples,
        "train_acc": weighted_acc / total_samples,
    }


def aggregate_evaluate_metrics(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    """Aggregate local evaluation metrics across clients weighted by sample count."""
    if not metrics:
        return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_acc = sum(num_samples * float(m.get("accuracy", 0.0)) for num_samples, m in metrics)
    return {
        "client_eval_accuracy": weighted_acc / total_samples,
    }


def create_global_evaluate_fn(
    model: TBImageClassifier,
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    checkpoint_dir: str | Path = "artifacts/checkpoints",
    best_checkpoint_name: str = "tb_federated_best.pt",
    last_checkpoint_name: str = "tb_federated_last.pt",
    class_names: tuple[str, str] = ("non_tb", "tb"),
    history_tracker: list[dict[str, Any]] | None = None,
    wandb_tracker: Any = None,
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None]:
    """Return an evaluate_fn callback that assesses the aggregated global model on the validation set."""
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / best_checkpoint_name
    last_ckpt_path = ckpt_dir / last_checkpoint_name

    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    best_auc: float = -1.0

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]] | None:
        nonlocal best_auc

        # Update global model with aggregated weights
        set_parameters_to_model(model, parameters)
        model.to(device)
        model.eval()

        total_loss = 0.0
        total_samples = 0
        all_probabilities: list[Tensor] = []
        all_targets: list[Tensor] = []

        with torch.inference_mode():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, dtype=torch.long, non_blocking=True)
                batch_size = targets.size(0)

                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)

                total_loss += float(loss.detach()) * batch_size
                total_samples += batch_size
                all_probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
                all_targets.append(targets.cpu())

        if total_samples == 0:
            LOGGER.warning("Validation set is empty during round %d evaluation.", server_round)
            return None

        probs = torch.cat(all_probabilities)
        tgts = torch.cat(all_targets)
        mean_loss = total_loss / total_samples

        preds = probs.argmax(dim=1)
        acc = float(MulticlassAccuracy(num_classes=2, average="micro")(preds, tgts))
        f1 = float(MulticlassF1Score(num_classes=2, average="macro")(preds, tgts))
        auc = float(BinaryAUROC()(probs[:, 1], tgts))
        sens = float(
            torch.sum((preds == 1) & (tgts == 1)) / torch.sum(tgts == 1).clamp_min(1)
        )
        spec = float(BinarySpecificity()(preds, tgts))
        confusion = BinaryConfusionMatrix()(preds, tgts).tolist()

        metrics_summary: dict[str, Any] = {
            "round": server_round,
            "loss": mean_loss,
            "accuracy": acc,
            "macro_f1": f1,
            "auc": auc,
            "sensitivity": sens,
            "specificity": spec,
            "confusion_matrix": confusion,
            "confusion_matrix_labels": list(class_names),
            "samples": total_samples,
        }

        LOGGER.info(
            "[Global Validation - Round %d] loss=%.4f acc=%.4f macro_f1=%.4f auc=%.4f sens=%.4f spec=%.4f",
            server_round,
            mean_loss,
            acc,
            f1,
            auc,
            sens,
            spec,
        )

        if history_tracker is not None:
            history_tracker.append(metrics_summary)

        if wandb_tracker is not None:
            try:
                wandb_tracker.log(
                    {
                        "round": server_round,
                        "global_val/loss": mean_loss,
                        "global_val/accuracy": acc,
                        "global_val/macro_f1": f1,
                        "global_val/auc": auc,
                        "global_val/sensitivity": sens,
                        "global_val/specificity": spec,
                    },
                    step=server_round,
                )
            except Exception as wandb_err:
                LOGGER.warning("Failed to log to wandb: %s", wandb_err)

        # Save checkpoint info
        checkpoint_payload = {
            "round": server_round,
            "model_state_dict": model.state_dict(),
            "metrics": metrics_summary,
            "class_names": class_names,
        }

        # Save last checkpoint
        torch.save(checkpoint_payload, last_ckpt_path)

        # Check if improved on monitored metric (AUC)
        if auc > best_auc:
            best_auc = auc
            torch.save(checkpoint_payload, best_ckpt_path)
            LOGGER.info(
                "New best global model saved (AUC=%.4f) -> %s",
                best_auc,
                best_ckpt_path,
            )

        scalar_metrics: dict[str, Scalar] = {
            "accuracy": acc,
            "macro_f1": f1,
            "auc": auc,
            "sensitivity": sens,
            "specificity": spec,
        }

        if device.type == "cuda":
            model.to("cpu")
            torch.cuda.empty_cache()

        return mean_loss, scalar_metrics

    return evaluate_fn


def build_strategy(
    strategy_name: str = "fedavg",
    num_clients: int = 3,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int | None = None,
    min_evaluate_clients: int | None = None,
    min_available_clients: int | None = None,
    proximal_mu: float = 0.01,
    local_epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None = None,
    initial_parameters: Parameters | None = None,
) -> Strategy:
    """Build FedAvg or FedProx strategy with configured parameters and fit config callback."""
    norm_name = strategy_name.lower().strip()
    min_fit = min_fit_clients if min_fit_clients is not None else num_clients
    min_eval = min_evaluate_clients if min_evaluate_clients is not None else num_clients
    min_avail = min_available_clients if min_available_clients is not None else num_clients

    def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
        config: dict[str, Scalar] = {
            "server_round": int(server_round),
            "local_epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        }
        if norm_name == "fedprox":
            config["proximal_mu"] = float(proximal_mu)
        return config

    if norm_name == "fedavg":
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=min_avail,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
    elif norm_name == "fedprox":
        return FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit,
            min_evaluate_clients=min_eval,
            min_available_clients=min_avail,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            proximal_mu=proximal_mu,
        )
    else:
        raise ValueError(f"Unsupported strategy {strategy_name!r}. Supported strategies: 'fedavg', 'fedprox'.")
