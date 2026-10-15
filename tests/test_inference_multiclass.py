"""Tests for the multi-class inference contract and clinical interpretation.

These tests exercise the API against a stub service so they stay fast and do not
depend on a GPU or a checkpoint: what is under test here is the response
contract, the class-agnostic plumbing and the configured clinical text.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.inference import api as api_module
from src.inference.clinical import (
    build_differential,
    build_explanation,
    build_interpretation,
    confidence_band,
    format_headline,
)
from src.inference.config import load_inference_config
from src.xai.focus import describe_focus

CLASS_NAMES = ("normal", "tb", "pneumonia")


class StubTransform:
    """Minimal albumentations-compatible transform for the stub service."""

    def __call__(self, image: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {"image": torch.zeros(3, 8, 8)}


class StubService:
    """Stand-in for InferenceService returning a fixed three-class prediction."""

    class_names = CLASS_NAMES
    checkpoint_path = Path("stub_multiclass_checkpoint.pt")
    device = torch.device("cpu")
    eval_transform = StubTransform()

    def predict(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "predicted_class_id": 2,
            "predicted_class_name": "pneumonia",
            "confidence": 0.91,
            "probabilities": {"normal": 0.04, "tb": 0.05, "pneumonia": 0.91},
            "logits": [0.1, 0.2, 2.4],
        }

    def explain(
        self, image_tensor: torch.Tensor, target_class: int | None = None
    ) -> dict[str, Any]:
        target_id = 2 if target_class is None else int(target_class)
        heatmap = np.zeros((8, 8), dtype=np.float32)
        heatmap[2:6, 2:6] = 0.9
        return {
            **self.predict(image_tensor),
            "target_class_id": target_id,
            "target_class_name": CLASS_NAMES[target_id],
            "heatmap": heatmap,
            "heatmap_color": np.zeros((8, 8, 3), dtype=np.uint8),
            "overlay": np.zeros((8, 8, 3), dtype=np.uint8),
            "rgb_image": np.zeros((8, 8, 3), dtype=np.float32),
            "colormap": "turbo",
            "alpha": 0.55,
            "focus": describe_focus(heatmap),
        }


@pytest.fixture(scope="module")
def png_upload() -> io.BytesIO:
    """A decodable PNG upload accepted by the API."""
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with the stub service and the shipped clinical text installed."""
    clinical = load_inference_config()["clinical"]
    monkeypatch.setattr(api_module, "inference_service", StubService())
    monkeypatch.setattr(api_module, "clinical_config", clinical)
    return TestClient(api_module.app)


# ---------------------------------------------------------------------------
# Shipped configuration
# ---------------------------------------------------------------------------


def test_shipped_config_serves_the_three_class_checkpoint() -> None:
    config = load_inference_config()

    assert config["class_names"] == CLASS_NAMES
    assert config["checkpoint_path"].name == "multiclass_efficientnet_b3_best.pt"
    assert config["model_type"] == "multiclass"
    assert config["image_size"] == 300


def test_every_configured_class_has_clinical_text() -> None:
    config = load_inference_config()
    clinical = config["clinical"]

    for class_name in config["class_names"]:
        interpretation = build_interpretation(
            class_name, 0.8, {name: 0.1 for name in config["class_names"]}, clinical
        )
        assert interpretation is not None, f"missing clinical text for {class_name}"
        assert interpretation["label"]
        assert interpretation["headline"]
        assert len(interpretation["notes"]) >= 2
        assert interpretation["tone"] in {"success", "warning", "error", "info"}


def test_build_interpretation_formats_probability_placeholders() -> None:
    clinical = load_inference_config()["clinical"]
    probabilities = {"normal": 0.11, "tb": 0.72, "pneumonia": 0.17}

    interpretation = build_interpretation("tb", 0.72, probabilities, clinical)

    assert interpretation is not None
    assert "72.0%" in interpretation["headline"]
    assert interpretation["label"] == "Tuberculosis"
    assert interpretation["tone"] == "error"


def test_build_interpretation_returns_none_for_unknown_class() -> None:
    clinical = load_inference_config()["clinical"]

    assert build_interpretation("covid", 0.5, {"covid": 0.5}, clinical) is None
    assert build_interpretation("tb", 0.5, {}, None) is None


def test_format_headline_keeps_unformattable_template() -> None:
    template = "Confidence {confidence:.0%} with {unknown_placeholder}"

    rendered = format_headline(template, "tb", 0.5, {"tb": 0.5})

    assert rendered == template


# ---------------------------------------------------------------------------
# Structured explanation (confidence band, differential, flags)
# ---------------------------------------------------------------------------


def test_confidence_band_reports_the_three_levels() -> None:
    assert confidence_band(0.93)["level"] == "High"
    assert confidence_band(0.72)["level"] == "Moderate"
    assert confidence_band(0.41)["level"] == "Low"
    assert confidence_band(0.41)["tone"] == "error"


def test_confidence_band_reports_the_runner_up_margin() -> None:
    clinical = load_inference_config()["clinical"]
    band = confidence_band(
        0.75,
        {"normal": 0.15, "tb": 0.75, "pneumonia": 0.10},
        class_name="tb",
        clinical_config=clinical,
    )

    assert band["runner_up"] == "normal"
    assert band["runner_up_label"] == "Normal"
    assert band["margin"] == pytest.approx(0.60, abs=1e-9)
    assert band["near_tie"] is False


def test_confidence_band_caps_a_near_tie_at_moderate() -> None:
    """Argmax over a 0.90/0.86 split is not a high-confidence answer."""
    band = confidence_band(
        0.90,
        {"normal": 0.05, "tb": 0.86, "pneumonia": 0.09},
        class_name="pneumonia",
    )

    assert band["near_tie"] is True
    assert band["level"] == "Moderate"
    assert "Capped to moderate" in band["explanation"]


def test_differential_ranks_and_labels_the_remaining_classes() -> None:
    clinical = load_inference_config()["clinical"]
    rows = build_differential(
        "pneumonia", {"normal": 0.05, "tb": 0.40, "pneumonia": 0.55}, clinical
    )

    assert [row["class_name"] for row in rows] == ["tb", "normal"]
    assert rows[0]["label"] == "Tuberculosis"
    assert "competing possibility" in rows[0]["note"]
    assert "Unlikely" in rows[1]["note"]


def test_build_explanation_carries_band_focus_differential_and_disclaimer() -> None:
    clinical = load_inference_config()["clinical"]
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[1:4, 1:4] = 0.9
    focus = describe_focus(heatmap)

    explanation = build_explanation(
        "pneumonia",
        0.91,
        {"normal": 0.04, "tb": 0.05, "pneumonia": 0.91},
        clinical,
        focus=focus,
    )

    assert explanation is not None
    assert explanation["class_name"] == "pneumonia"
    assert explanation["confidence"]["level"] == "High"
    assert explanation["focus"] == focus
    assert explanation["differential"][0]["class_name"] == "tb"
    assert explanation["disclaimer"]
    # Nothing to warn about: decisive prediction on a localised map.
    assert explanation["flags"] == []


def test_build_explanation_flags_an_indeterminate_screen() -> None:
    clinical = load_inference_config()["clinical"]

    explanation = build_explanation(
        "tb", 0.44, {"normal": 0.20, "tb": 0.44, "pneumonia": 0.36}, clinical
    )

    assert explanation is not None
    assert explanation["confidence"]["level"] == "Low"
    assert explanation["confidence"]["near_tie"] is True
    assert any("close to tied" in flag for flag in explanation["flags"])
    # Focus was not computed on this path and must say so, not pretend.
    assert explanation["focus"]["available"] is False
    assert explanation["focus"]["reason"] == "not_computed"


def test_build_explanation_returns_none_for_an_unconfigured_class() -> None:
    clinical = load_inference_config()["clinical"]

    assert build_explanation("covid", 0.5, {"covid": 0.5}, clinical) is None


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_health_reports_task_and_class_names(client: TestClient) -> None:
    payload = client.get("/health").json()

    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["num_classes"] == len(CLASS_NAMES)
    assert payload["task"] == "multiclass_normal_tb_pneumonia"


def test_undecodable_upload_is_a_client_error_not_a_server_error(
    client: TestClient,
) -> None:
    """A mislabelled file must be a 400, never a 500 that looks like a bug."""
    # Valid PNG signature followed by garbage: content type says image/png, the
    # bytes are not a decodable image.
    truncated = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    client = TestClient(api_module.app, raise_server_exceptions=False)

    for endpoint in ("/predict", "/explain"):
        response = client.post(
            endpoint, files={"file": ("broken.png", io.BytesIO(truncated), "image/png")}
        )
        assert response.status_code == 400, (endpoint, response.text)
        assert "decoded" in response.json()["detail"]


def test_predict_returns_three_classes_and_interpretation(
    client: TestClient, png_upload: io.BytesIO
) -> None:
    response = client.post(
        "/predict", files={"file": ("xray.png", png_upload, "image/png")}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["predicted_class_name"] == "pneumonia"
    assert set(payload["probabilities"]) == set(CLASS_NAMES)
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert payload["interpretation"]["label"] == "Pneumonia"
    assert payload["interpretation"]["tone"] == "warning"
    # The structured explanation travels with the prediction: banded confidence,
    # a ranked differential and the standing disclaimer.
    assert payload["interpretation"]["confidence"]["level"] == "High"
    assert payload["interpretation"]["differential"][0]["class_name"] == "tb"
    assert payload["interpretation"]["disclaimer"]
    assert payload["interpretation"]["focus"]["available"] is False


def test_explain_returns_gradcam_images_and_interpretation(
    client: TestClient, png_upload: io.BytesIO
) -> None:
    response = client.post(
        "/explain", files={"file": ("xray.png", png_upload, "image/png")}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload["probabilities"]) == set(CLASS_NAMES)
    for key in ("heatmap_base64", "overlay_base64", "rgb_image_base64"):
        assert payload[key]
    # The client must be told which class the heatmap explains.
    assert payload["target_class_name"] == "pneumonia"
    assert payload["target_class_id"] == payload["predicted_class_id"]
    assert payload["interpretation"]["class_name"] == "pneumonia"


def test_explain_serves_a_coloured_heatmap_and_the_focus_block(
    client: TestClient, png_upload: io.BytesIO
) -> None:
    """The heatmap artifact must be a coloured RGB PNG plus a readable focus."""
    response = client.post(
        "/explain", files={"file": ("xray.png", png_upload, "image/png")}
    )
    payload = response.json()

    heatmap_png = base64.b64decode(payload["heatmap_base64"])
    assert heatmap_png[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(heatmap_png)) as image:
        assert image.mode == "RGB", "the downloaded heatmap must be colourised"

    focus = payload["focus"]
    assert focus["available"] is True
    assert focus["region_label"]
    assert "Activation is concentrated" in focus["summary"]
    assert focus["convention"]
    # The same block also travels inside the clinical explanation.
    assert payload["interpretation"]["focus"] == focus


def test_explain_can_target_any_reported_class(
    client: TestClient, png_upload: io.BytesIO
) -> None:
    """Requesting another class must render that class's map, not the prediction's."""
    response = client.post(
        "/explain?target_class=tb",
        files={"file": ("xray.png", png_upload, "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["predicted_class_name"] == "pneumonia"
    assert payload["target_class_name"] == "tb"
    assert payload["target_class_id"] == 1
    # The explanation still describes the prediction, not the inspected class.
    assert payload["interpretation"]["class_name"] == "pneumonia"


def test_explain_rejects_an_unknown_target_class(
    client: TestClient, png_upload: io.BytesIO
) -> None:
    response = client.post(
        "/explain?target_class=covid",
        files={"file": ("xray.png", png_upload, "image/png")},
    )

    assert response.status_code == 400
    assert "covid" in response.json()["detail"]
    assert "normal" in response.json()["detail"]


def test_invalid_upload_is_rejected_before_the_model_is_consulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad upload must be a 400 even when no model is loaded."""
    monkeypatch.setattr(api_module, "inference_service", None)
    client = TestClient(api_module.app)

    response = client.post(
        "/predict", files={"file": ("notes.txt", io.BytesIO(b"nope"), "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image"


def test_valid_upload_reports_service_unavailable_when_unloaded(
    monkeypatch: pytest.MonkeyPatch, png_upload: io.BytesIO
) -> None:
    monkeypatch.setattr(api_module, "inference_service", None)
    client = TestClient(api_module.app)

    response = client.post(
        "/predict", files={"file": ("xray.png", png_upload, "image/png")}
    )

    assert response.status_code == 503
