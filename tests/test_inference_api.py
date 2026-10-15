"""Tests for the FastAPI inference API."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.inference.api import app
from src.inference.config import load_inference_config

client = TestClient(app)

# The served contract follows configs/inference/inference.yaml: three classes for
# the multi-class checkpoint, two for the binary TB models.
INFERENCE_CONFIG = load_inference_config()
CLASS_NAMES = INFERENCE_CONFIG["class_names"]


@pytest.fixture(scope="module")
def sample_image() -> io.BytesIO:
    """Create a sample test image."""
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data


def test_predict_endpoint(sample_image: io.BytesIO):
    """Test the /predict endpoint with a sample image."""
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/predict", files=files)

    # 503 is acceptable when no checkpoint is available; a 500 means a real
    # server-side failure and must not be tolerated.
    assert response.status_code in [200, 503], response.text
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class_id" in data
        assert "predicted_class_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0


def test_explain_endpoint(sample_image: io.BytesIO):
    """Test the /explain endpoint with a sample image."""
    files = {"file": ("test.png", sample_image, "image/png")}
    response = client.post("/explain", files=files)

    assert response.status_code in [200, 503], response.text
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class_id" in data
        assert "predicted_class_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "heatmap_base64" in data
        assert "overlay_base64" in data
        assert "rgb_image_base64" in data


def test_predict_invalid_file_type():
    """Test the /predict endpoint with invalid file type."""
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_explain_invalid_file_type():
    """Test the /explain endpoint with invalid file type."""
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/explain", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


# The tests above use a bare TestClient, which does not run the app's lifespan
# startup, so the service is uninitialised and those assertions never exercise
# real inference. The following tests start the app properly (when a checkpoint
# is present) and therefore fail on any server-side regression - for example an
# un-awaited coroutine in the request handlers.

CHECKPOINT = Path(INFERENCE_CONFIG["checkpoint_path"])
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason="configured checkpoint not available"
)


@requires_checkpoint
def test_predict_runs_real_inference(sample_image: io.BytesIO):
    """POSTing a real image must return a fully populated prediction."""
    sample_image.seek(0)
    with TestClient(app) as live_client:
        files = {"file": ("test.png", sample_image, "image/png")}
        response = live_client.post("/predict", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["predicted_class_name"] in CLASS_NAMES, data
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"]) == set(CLASS_NAMES), data
    # The clinical summary must come back worded for the predicted class.
    assert data["interpretation"] is not None
    assert data["interpretation"]["class_name"] == data["predicted_class_name"]
    assert data["interpretation"]["headline"]


@requires_checkpoint
def test_explain_runs_real_gradcam(sample_image: io.BytesIO):
    """POSTing a real image must return decodable Grad-CAM artifacts."""
    import base64

    sample_image.seek(0)
    with TestClient(app) as live_client:
        files = {"file": ("test.png", sample_image, "image/png")}
        response = live_client.post("/explain", files=files)

    assert response.status_code == 200, response.text
    data = response.json()
    for key in ("heatmap_base64", "overlay_base64", "rgb_image_base64"):
        assert base64.b64decode(data[key])[:8] == b"\x89PNG\r\n\x1a\n", key
    assert set(data["probabilities"]) == set(CLASS_NAMES), data
