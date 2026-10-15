"""Tests for resuming an interrupted training run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import EarlyStoppingConfig, Trainer

FEATURES = torch.tensor(
    [
        [-2.0, -1.0],
        [-1.0, -2.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [-1.5, -0.5],
        [0.5, 1.5],
    ]
)
LABELS = torch.tensor([0, 0, 1, 1, 0, 1])


def make_loader(labels: torch.Tensor) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(TensorDataset(FEATURES, labels), batch_size=3, shuffle=False)


def make_trainer(checkpoint_dir: Path) -> Trainer:
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    return Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.05),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
        scheduler=None,
        checkpoint_dir=checkpoint_dir,
        early_stopping=EarlyStoppingConfig(
            enabled=False,
            monitor="auc",
            mode="max",
        ),
    )


def test_resume_continues_remaining_epochs_and_protects_best_checkpoint(
    tmp_path: Path,
) -> None:
    loader = make_loader(LABELS)
    first = make_trainer(tmp_path)
    first_result = first.fit(loader, loader, epochs=1)
    best_before = torch.load(
        tmp_path / "tb_image_best.pt", map_location="cpu", weights_only=False
    )

    # Epoch 2 is scored against deliberately inverted labels, so it cannot
    # beat the first epoch and must not overwrite the best checkpoint.
    resumed = make_trainer(tmp_path)
    resumed_result = resumed.fit(
        loader,
        make_loader(1 - LABELS),
        epochs=2,
        resume_from=tmp_path / "tb_image_last.pt",
    )
    best_after = torch.load(
        tmp_path / "tb_image_best.pt", map_location="cpu", weights_only=False
    )
    last_after = torch.load(
        tmp_path / "tb_image_last.pt", map_location="cpu", weights_only=False
    )

    assert len(resumed_result["history"]) == 1
    assert resumed_result["best_epoch"] == 1
    assert first_result["best_epoch"] == 1
    assert best_after["epoch"] == 1
    assert best_after["metrics"]["accuracy"] == best_before["metrics"]["accuracy"]
    assert best_after["best_epoch"] == 1
    assert last_after["epoch"] == 2
    assert last_after["epochs_without_improvement"] == 1


def test_resume_rejects_completed_run(tmp_path: Path) -> None:
    loader = make_loader(LABELS)
    trainer = make_trainer(tmp_path)
    trainer.fit(loader, loader, epochs=1)

    with pytest.raises(ValueError, match="already completed epoch 1 of 1"):
        make_trainer(tmp_path).fit(
            loader,
            loader,
            epochs=1,
            resume_from=tmp_path / "tb_image_last.pt",
        )


def test_resume_rejects_missing_checkpoint(tmp_path: Path) -> None:
    loader = make_loader(LABELS)

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        make_trainer(tmp_path).fit(
            loader,
            loader,
            epochs=2,
            resume_from=tmp_path / "missing.pt",
        )


def test_checkpoint_stores_resume_bookkeeping(tmp_path: Path) -> None:
    loader = make_loader(LABELS)
    make_trainer(tmp_path).fit(loader, loader, epochs=2)

    last = torch.load(
        tmp_path / "tb_image_last.pt", map_location="cpu", weights_only=False
    )
    best = torch.load(
        tmp_path / "tb_image_best.pt", map_location="cpu", weights_only=False
    )

    assert set(best) >= {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_score",
        "best_epoch",
        "best_metrics",
        "epochs_without_improvement",
    }
    assert last["best_score"] == pytest.approx(best["metrics"]["auc"])
    assert last["best_epoch"] == best["epoch"]
    assert last["epochs_without_improvement"] >= 0
