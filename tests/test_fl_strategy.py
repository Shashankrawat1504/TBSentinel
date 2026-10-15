"""Unit tests for Federated strategy creation and global evaluation callback."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from flwr.server.strategy import FedAvg, FedProx
from torch.utils.data import DataLoader, TensorDataset

from src.federation.client import get_parameters_from_model
from src.federation.strategy import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
    build_strategy,
    create_global_evaluate_fn,
)
from src.models.image_model import TBImageClassifier


@pytest.fixture
def mock_classifier() -> TBImageClassifier:
    """Create a lightweight TBImageClassifier for testing."""
    return TBImageClassifier(
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
    )


@pytest.fixture
def mock_val_loader() -> DataLoader:
    """Create a synthetic validation DataLoader."""
    images = torch.rand(6, 3, 64, 64)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=3, shuffle=False)


def test_build_fedavg_strategy() -> None:
    """Test building a FedAvg strategy."""
    strategy = build_strategy(
        strategy_name="fedavg",
        num_clients=3,
        local_epochs=1,
        learning_rate=1e-4,
    )
    assert isinstance(strategy, FedAvg)
    assert not isinstance(strategy, FedProx)


def test_build_fedprox_strategy() -> None:
    """Test building a FedProx strategy with proximal_mu."""
    strategy = build_strategy(
        strategy_name="fedprox",
        proximal_mu=0.05,
        local_epochs=2,
        learning_rate=1e-4,
    )
    assert isinstance(strategy, FedProx)
    assert strategy.proximal_mu == 0.05


def test_build_unsupported_strategy_raises_error() -> None:
    """Test that unsupported strategies raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        build_strategy(strategy_name="unknown_strategy")


def test_metric_aggregations() -> None:
    """Test weighted metric aggregation functions."""
    fit_metrics = [
        (100, {"train_loss": 0.5, "train_acc": 0.8}),
        (200, {"train_loss": 0.2, "train_acc": 0.95}),
    ]
    agg_fit = aggregate_fit_metrics(fit_metrics)
    assert agg_fit["train_loss"] == pytest.approx((100 * 0.5 + 200 * 0.2) / 300)
    assert agg_fit["train_acc"] == pytest.approx((100 * 0.8 + 200 * 0.95) / 300)

    eval_metrics = [
        (50, {"accuracy": 0.7}),
        (150, {"accuracy": 0.9}),
    ]
    agg_eval = aggregate_evaluate_metrics(eval_metrics)
    assert agg_eval["client_eval_accuracy"] == pytest.approx((50 * 0.7 + 150 * 0.9) / 200)


def test_global_evaluate_fn(
    mock_classifier: TBImageClassifier,
    mock_val_loader: DataLoader,
    tmp_path: Path,
) -> None:
    """Test the global evaluation callback tracks metrics and saves checkpoints."""
    history: list[dict] = []
    evaluate_fn = create_global_evaluate_fn(
        model=mock_classifier,
        val_loader=mock_val_loader,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        best_checkpoint_name="test_best.pt",
        last_checkpoint_name="test_last.pt",
        history_tracker=history,
    )

    ndarrays = get_parameters_from_model(mock_classifier)
    result = evaluate_fn(1, ndarrays, {})
    assert result is not None
    loss, metrics = result

    assert loss >= 0.0
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "auc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics

    # Checkpoint was created
    assert (tmp_path / "test_best.pt").is_file()
    assert (tmp_path / "test_last.pt").is_file()
    assert len(history) == 1
    assert history[0]["round"] == 1
