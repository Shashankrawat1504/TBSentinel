"""HTTP client for the TBSentinel FastAPI inference service.

The dashboard is a presentation layer only — it never loads the model itself.
Everything below talks to the backend over ``/health``, ``/predict`` and
``/explain`` so the backend contract stays the single source of truth.
"""

from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

DEFAULT_API_URL = os.environ.get("TBSENTINEL_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60
# Short display names for the classes the API reports in /health.
CLASS_DISPLAY_NAMES = {
    "normal": "Normal",
    "tb": "TB",
    "pneumonia": "Pneumonia",
    "non_tb": "Non-TB",
}
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp"]
MIME_BY_SUFFIX = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "bmp": "image/bmp",
}


class ApiError(RuntimeError):
    """Raised when the inference backend cannot serve a request."""


def check_api_health(api_url: str) -> dict[str, Any] | None:
    """Return the ``/health`` payload, or ``None`` when the API is unreachable.

    Cached by the caller (``st.cache_data``) — this function stays pure so it
    can be reused from tests and scripts.
    """
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _post_image(
    api_url: str,
    endpoint: str,
    image_bytes: bytes,
    filename: str,
    mime_type: str,
) -> dict[str, Any]:
    """POST an image to one inference endpoint and return the decoded JSON."""
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.post(
            url,
            files={"file": (filename, image_bytes, mime_type)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        detail = ""
        status = exc.response.status_code if exc.response is not None else "?"
        if exc.response is not None:
            try:
                detail = str(exc.response.json().get("detail", ""))
            except ValueError:
                detail = exc.response.text[:200]
        raise ApiError(f"{url} returned HTTP {status}. {detail}".strip()) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"{url} timed out after {REQUEST_TIMEOUT_SECONDS}s.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Could not reach {url}. Is the FastAPI server running?") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(f"{url} returned a non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise ApiError(f"{url} returned an unexpected response shape.")
    return payload


def run_analysis(
    api_url: str,
    image_bytes: bytes,
    filename: str,
    mime_type: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Call ``/predict`` and ``/explain`` concurrently.

    Returns ``(prediction, explanation, explanation_error)``. Prediction errors
    propagate as :class:`ApiError`; explanation failures are reported separately
    so a working prediction is still shown.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        predict_future = pool.submit(
            _post_image, api_url, "predict", image_bytes, filename, mime_type
        )
        explain_future = pool.submit(
            _post_image, api_url, "explain", image_bytes, filename, mime_type
        )

        prediction = predict_future.result()
        explanation: dict[str, Any] | None = None
        explain_error: str | None = None
        try:
            explanation = explain_future.result()
        except ApiError as exc:
            explain_error = str(exc)
    return prediction, explanation, explain_error


def run_explain(
    api_url: str,
    image_bytes: bytes,
    filename: str,
    mime_type: str,
    target_class: str | None = None,
) -> dict[str, Any]:
    """Call ``/explain`` for one class and return the explanation payload.

    ``target_class`` is a class name as reported by ``/health`` (e.g. ``tb``);
    ``None`` asks the backend for the map of whichever class it predicted. Used by
    the explainability card's class selector, so a reviewer can see what each
    class is attending to on the same radiograph.
    """
    endpoint = "explain"
    if target_class:
        endpoint = f"explain?target_class={quote(str(target_class))}"
    return _post_image(api_url, endpoint, image_bytes, filename, mime_type)


def prediction_explanation(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The explanation that belongs to the prediction itself.

    The Grad-CAM card can be pointed at another class, and a ``/explain`` payload
    for that class carries a focus describing *that* map. The first explanation
    (which always explains the predicted class) is therefore kept separately as
    ``prediction_explanation``, so the clinical card, the stored record and the
    exported report never splice a different class's focus into the prediction's
    report. Falls back to the live explanation for records that predate the field.
    """
    if not record:
        return None
    snapshot = record.get("prediction_explanation") or {}
    if snapshot.get("interpretation"):
        return snapshot

    # No snapshot (older session or a record written by an earlier version): the
    # live explanation is usable only when its map is still the prediction's.
    explanation = record.get("explanation") or {}
    if explanation.get("interpretation"):
        prediction = record.get("prediction") or {}
        predicted = str(prediction.get("predicted_class_name") or "").casefold()
        target = str(explanation.get("target_class_name") or "").casefold()
        if not target or not predicted or target == predicted:
            return explanation
    return None


def prediction_interpretation(record: dict[str, Any] | None) -> dict[str, Any]:
    """Structured clinical explanation for the prediction (``{}`` when absent)."""
    payload = prediction_explanation(record) or {}
    interpretation = payload.get("interpretation") or {}
    if interpretation:
        return dict(interpretation)
    prediction = (record or {}).get("prediction") or {}
    return dict(prediction.get("interpretation") or {})


def decode_base64_png(payload: str) -> bytes:
    """Decode a base64-encoded PNG string into raw bytes."""
    data = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    return base64.b64decode(data)


def model_strategy_label(checkpoint_path: str | None) -> str:
    """Infer the served model family from the checkpoint filename."""
    if not checkpoint_path:
        return "Unknown"
    name = Path(checkpoint_path).name.lower()
    if "fedprox" in name:
        return "Federated (FedProx)"
    if "fedavg" in name:
        return "Federated (FedAvg)"
    if "federated" in name:
        return "Federated"
    return "Centralized"


def health_summary(health: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise a ``/health`` payload into UI-friendly flags.

    ``class_names``/``class_list``/``task`` come from the served model, so the
    dashboard renders whatever the backend really answers (two classes for the
    binary checkpoints, three for the multi-class one).
    """
    if not health:
        return {
            "api_online": False,
            "model_loaded": False,
            "device": "—",
            "checkpoint_name": "—",
            "strategy": "Unknown",
            "task": None,
            "task_label": "—",
            "class_names": [],
            "class_list": "—",
        }
    checkpoint = str(health.get("checkpoint_path") or "—")
    class_names = [str(name) for name in (health.get("class_names") or [])]
    task = health.get("task")
    return {
        "api_online": True,
        "model_loaded": bool(health.get("model_loaded")),
        "device": str(health.get("device", "unknown")).upper(),
        "checkpoint_name": Path(checkpoint).name if checkpoint != "—" else "—",
        "strategy": model_strategy_label(
            health.get("checkpoint_path") if health else None
        ),
        "task": task,
        "task_label": (
            f"{len(class_names)}-class" if class_names else "—"
        ),
        "class_names": class_names,
        "class_list": " / ".join(
            CLASS_DISPLAY_NAMES.get(name, name.title()) for name in class_names
        )
        or "—",
    }
