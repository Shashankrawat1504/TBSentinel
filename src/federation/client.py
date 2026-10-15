"""Flower NumPyClient implementation for federated tuberculosis detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import flwr as fl
import torch
from flwr.common import Context, NDArrays, Scalar
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from src.data.dataset import (
    DEFAULT_IMAGE_ROOT,
    TBXRayDataset,
    build_train_transform,
)
from src.models.image_model import TBImageClassifier

LOGGER = logging.getLogger(__name__)


def get_parameters_from_model(model: nn.Module) -> NDArrays:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters_to_model(model: nn.Module, parameters: NDArrays) -> None:
    """Load NumPy parameters into a PyTorch model's state dictionary."""
    state_dict = model.state_dict()
    for (k, v), param_np in zip(state_dict.items(), parameters):
        state_dict[k] = torch.as_tensor(param_np, dtype=v.dtype, device=v.device)
    model.load_state_dict(state_dict, strict=True)


class TBNumPyClient(fl.client.NumPyClient):
    """Flower client training an EfficientNet-B3 classifier on a local partition.

    Supports both standard FedAvg local updates and FedProx proximal regularization.
    """

    def __init__(
        self,
        cid: str,
        model: TBImageClassifier,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        eval_loader: DataLoader[tuple[Tensor, Tensor]] | None = None,
        device: torch.device | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        local_epochs: int = 1,
        proximal_mu: float = 0.0,
        mixed_precision: bool = True,
        gradient_clip_norm: float | None = 1.0,
    ) -> None:
        self.cid = str(cid)
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.local_epochs = local_epochs
        self.proximal_mu = proximal_mu
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current local model weights as NumPy ndarrays."""
        return get_parameters_from_model(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Update local model weights from received aggregated parameters."""
        set_parameters_to_model(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the model on the local partition using the received weights."""
        self.set_parameters(parameters)

        # Parse round-level configurations
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))
        weight_decay = float(config.get("weight_decay", self.weight_decay))
        proximal_mu = float(config.get("proximal_mu", self.proximal_mu))
        server_round = int(config.get("server_round", 1))

        # Keep a frozen copy of initial weights for FedProx proximal term
        global_params: list[Tensor] | None = None
        if proximal_mu > 0.0:
            global_params = [
                param.detach().clone() for param in self.model.parameters()
            ]

        optimizer: Optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        num_batches = len(self.train_loader)
        for epoch in range(1, local_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (images, targets) in enumerate(self.train_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, dtype=torch.long, non_blocking=True)
                batch_size = targets.size(0)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    task_loss = self.criterion(logits, targets)

                    # Proximal term for FedProx: (mu / 2) * ||w - w^t||^2
                    if proximal_mu > 0.0 and global_params is not None:
                        proximal_loss = torch.zeros(
                            (), dtype=task_loss.dtype, device=task_loss.device
                        )
                        for param, g_param in zip(self.model.parameters(), global_params):
                            proximal_loss += torch.sum((param - g_param) ** 2)
                        loss = task_loss + (proximal_mu / 2.0) * proximal_loss
                    else:
                        loss = task_loss

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm is not None:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip_norm,
                        )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clip_norm,
                        )
                    optimizer.step()

                batch_loss = float(task_loss.detach()) * batch_size
                preds = logits.argmax(dim=1)
                batch_correct = int((preds == targets).sum().item())

                epoch_loss += batch_loss
                epoch_correct += batch_correct
                epoch_samples += batch_size

                if batch_idx % 50 == 0 or batch_idx == num_batches:
                    LOGGER.info(
                        "[Client %s] Round %d Epoch %d/%d - Batch %d/%d (%.1f%%) loss: %.4f acc: %.4f",
                        self.cid,
                        server_round,
                        epoch,
                        local_epochs,
                        batch_idx,
                        num_batches,
                        (batch_idx / num_batches) * 100.0,
                        epoch_loss / max(epoch_samples, 1),
                        epoch_correct / max(epoch_samples, 1),
                    )

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        num_train_samples = len(self.train_loader.dataset)  # type: ignore[arg-type]
        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = correct_predictions / max(total_samples, 1)

        LOGGER.info(
            "[Client %s] Round %d fit completed: %d samples, loss=%.4f, acc=%.4f (proximal_mu=%.4f)",
            self.cid,
            server_round,
            num_train_samples,
            avg_loss,
            avg_acc,
            proximal_mu,
        )

        metrics: dict[str, Scalar] = {
            "train_loss": float(avg_loss),
            "train_acc": float(avg_acc),
            "client_id": self.cid,
        }

        # Clean VRAM after training to avoid fragmentation across clients
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return self.get_parameters(config), num_train_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate local model on local data partition."""
        self.set_parameters(parameters)
        eval_loader = self.eval_loader or self.train_loader

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.inference_mode():
            for images, targets in eval_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, dtype=torch.long, non_blocking=True)
                batch_size = targets.size(0)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                total_loss += float(loss.detach()) * batch_size
                preds = logits.argmax(dim=1)
                correct_predictions += int((preds == targets).sum().item())
                total_samples += batch_size

        num_samples = len(eval_loader.dataset)  # type: ignore[arg-type]
        eval_loss = total_loss / max(total_samples, 1)
        eval_acc = correct_predictions / max(total_samples, 1)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return (
            float(eval_loss),
            num_samples,
            {"accuracy": float(eval_acc), "client_id": self.cid},
        )


def create_client_fn(
    federated_splits_dir: str | Path = "data/federated_splits",
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    image_size: int = 300,
    batch_size: int = 8,
    num_workers: int = 0,
    device: torch.device | None = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    local_epochs: int = 1,
    proximal_mu: float = 0.0,
    mixed_precision: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a client_fn callback compatible with Flower simulation.

    Accepts either Flower 1.x Context or cid string.
    """
    splits_dir = Path(federated_splits_dir).expanduser().resolve()
    resolved_image_root = Path(image_root).expanduser().resolve()
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_kwargs = model_kwargs or {"num_classes": 2, "pretrained": False}

    def client_fn(context_or_cid: Context | str) -> fl.client.Client:
        # Determine client id
        if isinstance(context_or_cid, str):
            cid = str(context_or_cid)
        elif hasattr(context_or_cid, "node_config") and "partition-id" in context_or_cid.node_config:
            cid = str(context_or_cid.node_config["partition-id"])
        else:
            cid = str(getattr(context_or_cid, "cid", "0"))

        client_csv = splits_dir / f"client_{cid}_train.csv"
        if not client_csv.is_file():
            raise FileNotFoundError(f"Missing partition CSV for client {cid}: {client_csv}")

        # Build local training dataset and DataLoader
        train_transform = build_train_transform(image_size=image_size)
        train_dataset = TBXRayDataset(
            annotations=client_csv,
            image_root=resolved_image_root,
            transform=train_transform,
            image_size=image_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=resolved_device.type == "cuda",
            drop_last=False,
        )

        # Initialize model
        model = TBImageClassifier(**resolved_model_kwargs)

        numpy_client = TBNumPyClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            device=resolved_device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            local_epochs=local_epochs,
            proximal_mu=proximal_mu,
            mixed_precision=mixed_precision,
        )

        return numpy_client.to_client()

    return client_fn
