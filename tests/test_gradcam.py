"""Unit tests for TBGradCAMExplainer and XAI visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.models.image_model import TBImageClassifier
from src.xai.gradcam import (
    TBGradCAMExplainer,
    denormalize_image,
    overlay_cam_on_image,
    to_rgb_float,
)

CHECKPOINT_PATH = Path("artifacts/checkpoints/tb_efficientnet_b3_best.pt")


@pytest.fixture(scope="module")
def mock_classifier() -> nn.Module:
    """Fast model fixture for testing explainer logic without weights."""
    model = TBImageClassifier(
        num_classes=2,
        pretrained=False,
        dropout=0.0,
    ).eval()
    return model


@pytest.fixture(scope="module")
def sample_tensor() -> torch.Tensor:
    """Single normalized image tensor (3, 300, 300)."""
    return torch.rand(3, 300, 300)


@pytest.fixture(scope="module")
def sample_batch() -> torch.Tensor:
    """Batch of normalized image tensors (2, 3, 300, 300)."""
    return torch.rand(2, 3, 300, 300)


def test_explainer_resolves_target_layer_automatically(mock_classifier: nn.Module) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        target_layer="auto",
        device="cpu",
    )
    assert explainer.target_layer is mock_classifier.backbone.features[-1]


def test_explainer_resolves_target_layer_by_string(mock_classifier: nn.Module) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        target_layer="backbone.features.8",
        device="cpu",
    )
    assert explainer.target_layer is mock_classifier.backbone.features[8]


def test_explainer_explain_single_tensor(
    mock_classifier: nn.Module,
    sample_tensor: torch.Tensor,
) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        device="cpu",
        image_size=300,
    )
    result = explainer.explain(sample_tensor)

    expected_keys = {
        "predicted_class_id",
        "predicted_class_name",
        "confidence",
        "probabilities",
        "logits",
        "target_class_id",
        "target_class_name",
        "heatmap",
        "overlay",
        "rgb_image",
    }
    assert expected_keys.issubset(result.keys())
    assert result["predicted_class_id"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["heatmap"].shape == (300, 300)
    assert result["heatmap"].min() >= 0.0
    assert result["heatmap"].max() <= 1.0
    assert result["overlay"].shape == (300, 300, 3)
    assert result["overlay"].dtype == np.uint8


def test_explainer_explain_pil_and_numpy(
    mock_classifier: nn.Module,
) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        device="cpu",
        image_size=300,
    )

    # PIL Image input
    pil_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    res_pil = explainer.explain(pil_img)
    assert res_pil["overlay"].shape == (300, 300, 3)

    # Numpy array input
    np_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res_np = explainer.explain(np_img)
    assert res_np["overlay"].shape == (300, 300, 3)


def test_explainer_explain_batch(
    mock_classifier: nn.Module,
    sample_batch: torch.Tensor,
) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        device="cpu",
        image_size=300,
    )
    batch_results = explainer.explain_batch(sample_batch)

    assert len(batch_results) == 2
    for res in batch_results:
        assert res["heatmap"].shape == (300, 300)
        assert res["overlay"].shape == (300, 300, 3)
        assert 0.0 <= res["confidence"] <= 1.0


def test_overlay_cam_on_image() -> None:
    rgb = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[20:80, 20:80] = 1.0

    overlay = overlay_cam_on_image(rgb, cam, alpha=0.5, colormap="jet")
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_denormalize_image(sample_tensor: torch.Tensor) -> None:
    denorm = denormalize_image(sample_tensor)
    assert denorm.shape == (300, 300, 3)
    assert denorm.dtype == np.float32
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0


def test_create_figure_and_save(
    mock_classifier: nn.Module,
    sample_tensor: torch.Tensor,
    tmp_path: Path,
) -> None:
    explainer = TBGradCAMExplainer(
        model=mock_classifier,
        device="cpu",
        image_size=300,
    )
    explanation = explainer.explain(sample_tensor)

    # Test figure creation
    fig = explainer.create_figure(explanation, ground_truth=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test save
    out_prefix = tmp_path / "test_case"
    saved = explainer.save_explanation(
        explanation=explanation,
        output_path=out_prefix,
        ground_truth=1,
        save_figure=True,
        save_overlay_only=True,
    )
    assert Path(saved["figure_path"]).is_file()
    assert Path(saved["overlay_path"]).is_file()


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="Best checkpoint not found")
def test_explainer_with_best_checkpoint() -> None:
    explainer = TBGradCAMExplainer(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )
    assert explainer.model is not None
    dummy = torch.rand(1, 3, 300, 300)
    res = explainer.explain(dummy)
    assert res["predicted_class_name"] in ("non_tb", "tb")
