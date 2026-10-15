"""FastAPI backend for the TBSentinel Clinical Decision Support System.

The served task is driven by the checkpoint: ``multiclass_efficientnet_b3_best.pt``
answers Normal / TB / Pneumonia, and the binary TB models still work if the
inference configuration points at them. Responses are class-agnostic —
``predicted_class_name`` plus the full probability map — so the dashboard adapts
to whatever the model serves.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.inference.clinical import build_explanation
from src.inference.config import api_settings, load_inference_config
from src.inference.service import InferenceService
from src.utils.logger import get_logger

LOGGER = get_logger("inference.api")

# Global service instance and its clinical interpretation text
inference_service: InferenceService | None = None
clinical_config: dict[str, Any] = {}


# Pydantic models for request/response
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    checkpoint_path: str | None = None
    device: str
    task: str | None = None
    class_names: list[str] = Field(default_factory=list)
    num_classes: int = 0


class PredictionResponse(BaseModel):
    """Prediction response.

    ``predicted_class_name`` and ``probabilities`` carry the full class
    distribution; ``interpretation`` holds the configured clinical summary.
    """

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    probabilities: dict[str, float]
    logits: list[float]
    interpretation: dict[str, Any] | None = None


class ExplanationResponse(BaseModel):
    """Explanation response with Grad-CAM overlay.

    ``target_class_name`` is the class the heatmap explains. It normally equals
    ``predicted_class_name``, and is reported explicitly so a client never has to
    assume which of three classes the map refers to. ``heatmap_base64`` is the
    coloured map (same colormap as the overlay), and ``focus`` is the structured
    regional reading of it that also travels inside ``interpretation``.
    """

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    probabilities: dict[str, float]
    target_class_id: int | None = None
    target_class_name: str | None = None
    heatmap_base64: str
    overlay_base64: str
    rgb_image_base64: str
    focus: dict[str, Any] | None = None
    interpretation: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the inference service and clinical text when the app starts.

    The lifespan context replaced the deprecated ``on_event("startup")`` hook,
    which newer Starlette releases removed from ``Router`` entirely.
    """
    await startup_event()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="TBSentinel API",
    description=(
        "Clinical Decision Support System for chest X-ray triage: "
        "Normal / Tuberculosis / Pneumonia with Grad-CAM explanations"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def get_inference_config() -> dict[str, Any]:
    """Load inference settings from ``configs/inference/inference.yaml``."""
    return load_inference_config()


def describe_task(class_names: tuple[str, ...] | list[str]) -> str:
    """Human-readable task label derived from the served class list."""
    names = [str(name).casefold() for name in class_names]
    if len(names) == 2:
        return "binary_tb"
    if {"normal", "tb", "pneumonia"} <= set(names):
        return "multiclass_normal_tb_pneumonia"
    return f"multiclass_{len(names)}_class"


def validate_image_upload(file: UploadFile) -> None:
    """Reject non-image uploads before any model state is consulted."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")


def decode_upload(contents: bytes) -> Image.Image:
    """Decode image bytes, rejecting undecodable uploads as a client error.

    A mislabelled or truncated file (a ``image/png`` content type that is not a
    PNG) is the caller's mistake, so it becomes a 400 rather than a 500 that
    would look like a broken model.
    """
    try:
        image: Image.Image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as exc:  # PIL raises several unrelated exception types
        LOGGER.warning("Rejected undecodable upload: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file could not be decoded as an image. Send a valid "
                "PNG, JPEG or BMP chest radiograph."
            ),
        ) from exc

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


async def startup_event() -> None:
    """Initialize the inference service and clinical text on startup."""
    global inference_service, clinical_config

    config = get_inference_config()
    clinical_config = dict(config.get("clinical") or {})
    LOGGER.info(
        "Initializing inference service: model_type=%s checkpoint=%s device=%s "
        "classes=%s",
        config.get("model_type"),
        config.get("checkpoint_path"),
        config.get("device"),
        config.get("class_names"),
    )

    try:
        inference_service = InferenceService(
            checkpoint_path=config["checkpoint_path"],
            device=config["device"],
            image_size=config["image_size"],
            confidence_threshold=config["confidence_threshold"],
            class_names=config["class_names"],
            interpretation_config=clinical_config,
            gradcam_config=dict(config.get("gradcam") or {}),
        )
        LOGGER.info(
            "Inference service initialized successfully: task=%s",
            describe_task(inference_service.class_names),
        )
    except Exception as e:
        LOGGER.error("Failed to initialize inference service: %s", e)
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint to verify API status."""
    if inference_service is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            checkpoint_path=None,
            device="unknown",
        )

    class_names = [str(name) for name in inference_service.class_names]
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        checkpoint_path=str(inference_service.checkpoint_path),
        device=str(inference_service.device),
        task=describe_task(inference_service.class_names),
        class_names=class_names,
        num_classes=len(class_names),
    )


async def preprocess_image(upload_file: UploadFile) -> torch.Tensor:
    """Preprocess uploaded image file to tensor.

    Args:
        upload_file: Uploaded image file.

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W).
    """
    if inference_service is None:
        # Callers validate the service first; this keeps the helper self-contained.
        raise HTTPException(status_code=503, detail="Inference service not initialized")

    # Read image file
    contents = await upload_file.read()

    # Decode (raises 400 for a mislabelled or truncated file) and force RGB.
    image = decode_upload(contents)

    # Convert to numpy array
    image_array = np.array(image)

    # Apply evaluation transform
    transformed = inference_service.eval_transform(image=image_array)
    image_tensor = transformed["image"].unsqueeze(0)

    return image_tensor


def array_to_base64(array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded string.

    Args:
        array: Numpy array (H, W) or (H, W, 3).

    Returns:
        Base64-encoded string.
    """
    if array.ndim == 2:
        # Grayscale
        image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    else:
        # RGB
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def encode_heatmap_png(explanation: dict[str, Any]) -> str:
    """Base64 PNG of the *coloured* Grad-CAM map.

    Falls back to the raw map for explanations produced before the coloured map
    existed, so an older backend still serves a decodable artifact.
    """
    colored = explanation.get("heatmap_color")
    if colored is not None:
        return array_to_base64(colored)
    return array_to_base64(explanation["heatmap"])


def resolve_target_class(name: str, class_names: tuple[str, ...] | list[str]) -> int:
    """Resolve a class name to its index, or raise a precise 400.

    The class selector in the dashboard is built from ``/health``, so an unknown
    name here means a hand-written request; answering with the valid options is
    friendlier than a 500 from a deeper layer.
    """
    names = [str(item) for item in class_names]
    wanted = name.strip().casefold()
    for index, candidate in enumerate(names):
        if candidate.casefold() == wanted:
            return index
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown target class {name!r}. Available classes: "
            f"{', '.join(names)}."
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded chest X-ray.

    Args:
        file: Uploaded image file (JPG, PNG, etc.).

    Returns:
        Predicted class, its confidence, the full class probability map and the
        configured clinical interpretation text.
    """
    validate_image_upload(file)

    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")

    try:
        # Preprocess image (async: reads the upload stream)
        image_tensor = await preprocess_image(file)

        # Run prediction
        prediction = inference_service.predict(image_tensor)
        # No Grad-CAM on this endpoint, so the focus block reports that it was
        # not computed instead of pretending the model looked nowhere.
        interpretation = build_explanation(
            str(prediction["predicted_class_name"]),
            float(prediction["confidence"]),
            dict(prediction["probabilities"]),
            clinical_config,
        )

        return PredictionResponse(**prediction, interpretation=interpretation)

    except HTTPException:
        # Already a precise client/server error (e.g. 400 for a bad upload).
        raise
    except Exception as e:
        LOGGER.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    file: UploadFile = File(...),
    target_class: str | None = Query(
        default=None,
        description=(
            "Class to explain (e.g. 'tb'). Defaults to the predicted class; any "
            "class reported by /health can be requested."
        ),
    ),
) -> ExplanationResponse:
    """Generate the Grad-CAM explanation for an uploaded chest X-ray.

    Args:
        file: Uploaded image file (JPG, PNG, etc.).
        target_class: Optional class name to explain instead of the predicted one,
            which is how the dashboard shows what each class is attending to.

    Returns:
        Prediction plus coloured heatmap, overlay and preprocessed images
        (base64 PNG), the structured focus and the clinical explanation.
    """
    validate_image_upload(file)

    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")

    target_index = (
        resolve_target_class(target_class, inference_service.class_names)
        if target_class is not None
        else None
    )

    try:
        # Preprocess image (async: reads the upload stream)
        image_tensor = await preprocess_image(file)

        # Generate explanation (default target: the predicted class)
        explanation = inference_service.explain(
            image_tensor, target_class=target_index
        )

        # Convert numpy arrays to base64
        heatmap_base64 = encode_heatmap_png(explanation)
        overlay_base64 = array_to_base64(explanation["overlay"])
        rgb_image_base64 = array_to_base64(explanation["rgb_image"])
        # The explanation always describes the *prediction*; the focus block
        # describes the map that was actually rendered, which may be another
        # class when the caller asked for one.
        interpretation = build_explanation(
            str(explanation["predicted_class_name"]),
            float(explanation["confidence"]),
            dict(explanation["probabilities"]),
            clinical_config,
            focus=explanation.get("focus"),
        )

        return ExplanationResponse(
            predicted_class_id=explanation["predicted_class_id"],
            predicted_class_name=explanation["predicted_class_name"],
            confidence=explanation["confidence"],
            probabilities=explanation["probabilities"],
            target_class_id=explanation.get("target_class_id"),
            target_class_name=explanation.get("target_class_name"),
            heatmap_base64=heatmap_base64,
            overlay_base64=overlay_base64,
            rgb_image_base64=rgb_image_base64,
            focus=explanation.get("focus"),
            interpretation=interpretation,
        )

    except HTTPException:
        # Already a precise client/server error (e.g. 400 for a bad upload).
        raise
    except Exception as e:
        LOGGER.error("Explanation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    LOGGER.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn

    # Same bind rules as ``python -m src.inference.main``: loopback unless the
    # operator opts in via TBSENTINEL_API_HOST / TBSENTINEL_API_PORT or the
    # api: block of configs/inference/inference.yaml.
    settings = api_settings()
    uvicorn.run(
        app,
        host=settings["host"],
        port=settings["port"],
        log_level=settings["log_level"],
    )
