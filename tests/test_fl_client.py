"""Unit tests for Flower TBNumPyClient and client factory."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federation.client import (
    TBNumPyClient,
    create_client_fn,
    get_parameters_from_model,
    set_parameters_to_model,
)
from src.federation.strategy import build_strategy
from src.models.image_model import TBImageClassifier


@pytest.fixture
def mock_image_classifier() -> TBImageClassifier:
    """Create a lightweight TBImageClassifier for unit testing."""
    return TBImageClassifier(
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
    )


@pytest.fixture
def mock_train_loader() -> DataLoader:
    """Create a small synthetic DataLoader for local training."""
    images = torch.rand(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_get_and_set_parameters_roundtrip(mock_image_classifier: TBImageClassifier) -> None:
    """Test extracting parameters to NumPy and restoring them into the model."""
    orig_params = get_parameters_from_model(mock_image_classifier)
    assert len(orig_params) > 0
    assert isinstance(orig_params[0], np.ndarray)

    # Modify floating-point parameters slightly
    modified_params = [
        (p + 0.1 if np.issubdtype(p.dtype, np.floating) else p)
        for p in orig_params
    ]
    set_parameters_to_model(mock_image_classifier, modified_params)

    restored_params = get_parameters_from_model(mock_image_classifier)
    for orig, restored in zip(modified_params, restored_params):
        np.testing.assert_allclose(orig, restored, atol=1e-6)


def test_client_fit_fedavg(
    mock_image_classifier: TBImageClassifier,
    mock_train_loader: DataLoader,
) -> None:
    """Test client fit under standard FedAvg (proximal_mu=0)."""
    client = TBNumPyClient(
        cid="0",
        model=mock_image_classifier,
        train_loader=mock_train_loader,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        local_epochs=1,
        proximal_mu=0.0,
        mixed_precision=False,
    )

    initial_params = client.get_parameters(config={})
    updated_params, num_samples, metrics = client.fit(
        parameters=initial_params,
        config={"local_epochs": 1, "learning_rate": 1e-3, "proximal_mu": 0.0},
    )

    assert num_samples == 4
    assert "train_loss" in metrics
    assert "train_acc" in metrics
    assert metrics["client_id"] == "0"
    assert isinstance(metrics["train_loss"], float)
    assert 0.0 <= float(metrics["train_acc"]) <= 1.0

    # Ensure parameters were updated by gradient descent
    changed = any(
        not np.allclose(init, upd, atol=1e-7)
        for init, upd in zip(initial_params, updated_params)
    )
    assert changed, "Model weights should update during fit."


def test_client_fit_fedprox(
    mock_image_classifier: TBImageClassifier,
    mock_train_loader: DataLoader,
) -> None:
    """Test client fit with FedProx proximal regularization (proximal_mu > 0)."""
    client = TBNumPyClient(
        cid="1",
        model=mock_image_classifier,
        train_loader=mock_train_loader,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        local_epochs=1,
        proximal_mu=0.1,
        mixed_precision=False,
    )

    initial_params = client.get_parameters(config={})
    updated_params, num_samples, metrics = client.fit(
        parameters=initial_params,
        config={"local_epochs": 1, "learning_rate": 1e-3, "proximal_mu": 0.5},
    )

    assert num_samples == 4
    assert float(metrics["train_loss"]) >= 0.0
    assert metrics["client_id"] == "1"
    assert len(updated_params) == len(initial_params)


def test_client_evaluate(
    mock_image_classifier: TBImageClassifier,
    mock_train_loader: DataLoader,
) -> None:
    """Test client evaluate method computes loss and accuracy."""
    client = TBNumPyClient(
        cid="2",
        model=mock_image_classifier,
        train_loader=mock_train_loader,
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    params = client.get_parameters(config={})
    loss, num_samples, metrics = client.evaluate(parameters=params, config={})

    assert num_samples == 4
    assert loss >= 0.0
    assert "accuracy" in metrics
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0


def test_create_client_fn_resolves_context_and_cid(tmp_path: Path) -> None:
    """Test that create_client_fn returns an operational Flower Client."""
    # Create mock client CSV and dummy images
    import pandas as pd
    from PIL import Image

    for name in ["dummy1.png", "dummy2.png"]:
        Image.new("RGB", (64, 64), color="white").save(tmp_path / name)

    client_csv = tmp_path / "client_0_train.csv"
    dummy_df = pd.DataFrame(
        {
            "image_path": ["dummy1.png", "dummy2.png"],
            "label": [0, 1],
            "label_name": ["non_tb", "tb"],
            "split": ["train", "train"],
        }
    )
    dummy_df.to_csv(client_csv, index=False)

    # Mock context
    mock_context = MagicMock()
    mock_context.node_config = {"partition-id": 0}

    # Verify create_client_fn accepts the temporary partition dir
    client_fn = create_client_fn(
        federated_splits_dir=tmp_path,
        image_root=tmp_path,
        image_size=64,
        batch_size=2,
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    # Calling with context
    client = client_fn(mock_context)
    assert client is not None


def test_fedavg_does_not_apply_fedprox_regularization() -> None:
    """FedAvg fit configuration must not carry a FedProx proximal coefficient."""
    strategy = build_strategy(
        strategy_name="fedavg",
        num_clients=3,
        proximal_mu=0.01,
    )

    fit_config = getattr(strategy, "on_fit_config_fn")(1)

    assert "proximal_mu" not in fit_config


def test_fedprox_passes_proximal_regularization_to_clients() -> None:
    """FedProx config carries the configured proximal coefficient."""
    strategy = build_strategy(
        strategy_name="fedprox",
        num_clients=3,
        proximal_mu=0.1,
    )

    fit_config = getattr(strategy, "on_fit_config_fn")(1)

    assert fit_config["proximal_mu"] == 0.1
