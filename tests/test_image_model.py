from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.image_model import TBImageClassifier
from src.training.trainer import EarlyStoppingConfig, FocalLoss, Trainer


@pytest.fixture(scope="module")
def image_batch() -> torch.Tensor:
    return torch.rand(2, 3, 96, 96)


def test_efficientnet_b3_forward_returns_binary_logits(
    image_batch: torch.Tensor,
) -> None:
    model = TBImageClassifier(
        num_classes=2,
        pretrained=False,
        dropout=0.0,
    ).eval()

    with torch.inference_mode():
        logits = model(image_batch)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_freeze_backbone_only_leaves_classifier_trainable() -> None:
    model = TBImageClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.image_head.parameters())


def test_optional_fusion_branch_accepts_aligned_numeric_features(
    image_batch: torch.Tensor,
) -> None:
    model = TBImageClassifier(
        pretrained=False,
        num_tabular_features=4,
        tabular_hidden_dims=(8,),
        fusion_dim=16,
        dropout=0.0,
        fusion_dropout=0.0,
    ).eval()

    with torch.inference_mode():
        logits = model(image_batch, torch.rand(2, 4))
        image_only_logits = model(image_batch)

    assert logits.shape == image_only_logits.shape == (2, 2)


def test_tabular_features_rejected_when_fusion_branch_is_disabled(
    image_batch: torch.Tensor,
) -> None:
    model = TBImageClassifier(pretrained=False)

    with pytest.raises(ValueError, match="branch is disabled"):
        model(image_batch, torch.rand(2, 4))


def test_fusion_branch_validates_feature_dimensions(
    image_batch: torch.Tensor,
) -> None:
    model = TBImageClassifier(
        pretrained=False,
        num_tabular_features=4,
    )

    with pytest.raises(ValueError, match="feature count"):
        model(image_batch, torch.rand(2, 3))


def test_focal_loss_returns_finite_scalar() -> None:
    loss = FocalLoss(gamma=2.0, class_weights=torch.tensor([0.5, 1.5]))
    result = loss(torch.tensor([[2.0, 0.1], [0.3, 1.2]]), torch.tensor([0, 1]))

    assert result.ndim == 0
    assert torch.isfinite(result)


def test_trainer_tracks_metrics_and_saves_best_and_last_checkpoints(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [-1.5, -0.5],
            [0.5, 1.5],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=3, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
        scheduler=None,
        checkpoint_dir=tmp_path,
        early_stopping=EarlyStoppingConfig(
            enabled=False,
            monitor="auc",
            mode="max",
        ),
    )

    result = trainer.fit(loader, loader, epochs=2)

    assert {"accuracy", "macro_f1", "auc", "sensitivity", "specificity"} <= set(
        result["best_metrics"]
    )
    assert len(result["best_metrics"]["confusion_matrix"]) == 2
    assert Path(result["best_checkpoint"]).is_file()
    assert Path(result["last_checkpoint"]).is_file()
