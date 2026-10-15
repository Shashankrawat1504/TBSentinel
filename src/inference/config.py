"""Runtime configuration for the inference service.

``configs/inference/inference.yaml`` is the single source of truth for the
served checkpoint, preprocessing, class order, Grad-CAM settings and the
clinical interpretation copy. This module is the only place that reads it, so
the FastAPI layer, the clinical interpreter and the tests cannot drift apart.

Loading never raises: an unreadable or incomplete file falls back to the
multi-class defaults and logs a warning, because the API must still report a
useful error instead of crashing at import time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("inference.config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference" / "inference.yaml"
CHECKPOINTS_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "multiclass_efficientnet_b3_best.pt"
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("normal", "tb", "pneumonia")
DEFAULT_MODEL_TYPE = "multiclass"
DEFAULT_IMAGE_SIZE = 300
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# The API is unauthenticated, so loopback is the only safe default: the service
# must not be reachable from the local network unless an operator opts in.
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_API_LOG_LEVEL = "info"
API_HOST_ENV = "TBSENTINEL_API_HOST"
API_PORT_ENV = "TBSENTINEL_API_PORT"
WILDCARD_HOSTS = {"0.0.0.0", "::", "[::]"}


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path, expanding Hydra-style ``${paths.*}`` tokens."""
    text = str(value)
    text = text.replace("${paths.checkpoints_dir}", str(CHECKPOINTS_DIR))
    text = text.replace("${paths.root_dir}", str(PROJECT_ROOT))
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def default_config() -> dict[str, Any]:
    """Return the built-in multi-class defaults."""
    return {
        "model_type": DEFAULT_MODEL_TYPE,
        "checkpoint_path": DEFAULT_CHECKPOINT,
        "device": "auto",
        "image_size": DEFAULT_IMAGE_SIZE,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "class_names": DEFAULT_CLASS_NAMES,
        "gradcam": {
            "target_layer": "auto",
            "colormap": "turbo",
            "alpha": 0.55,
            "eigen_smooth": False,
            "blur_kernel": 9,
            "clip_percentile": 99.0,
            "focus_threshold": 0.5,
        },
        "clinical": {},
        "source": "built-in defaults",
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any] | None:
    """Return ``path`` as a plain mapping, or ``None`` when unreadable.

    Hydra-style ``${paths.*}`` tokens are deliberately left unresolved here;
    callers that need real paths run them through :func:`resolve_path`.
    """
    if not path.is_file():
        return None
    try:
        from omegaconf import OmegaConf

        # Resolve nothing here: the file uses Hydra-style ${paths.*} tokens that
        # are resolved by resolve_path(). Using OmegaConf.get() directly would
        # try to interpolate them against a config that has no ``paths`` node
        # and raise, which previously made the API silently ignore the whole
        # file and serve the built-in defaults.
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # noqa: BLE001 - config problems must not crash the API
        LOGGER.warning("Could not read %s (%s).", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def load_inference_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load inference settings, falling back to defaults when unavailable.

    Args:
        config_path: Optional override used by tests; defaults to
            ``configs/inference/inference.yaml``.

    Returns:
        Mapping with ``model_type``, ``checkpoint_path`` (resolved ``Path``),
        ``device``, ``image_size``, ``confidence_threshold``, ``class_names``
        (tuple, model output order), ``gradcam``, ``clinical`` and ``source``.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml_mapping(path)
    if raw is None:
        LOGGER.warning("Inference config not found at %s; using defaults.", path)
        return default_config()

    try:
        raw_names = raw.get("class_names") or list(DEFAULT_CLASS_NAMES)
        class_names = tuple(str(name) for name in raw_names)
        if not class_names:
            raise ValueError("class_names must not be empty")
        clinical = raw.get("clinical") or {}
        gradcam = raw.get("gradcam") or {}
        # ``clinical`` mixes per-class copy with session-wide text (the
        # disclaimer), so the string entries are lifted out and kept alongside
        # the per-class blocks.
        clinical_blocks: dict[str, Any] = {
            str(name): {str(key): value for key, value in dict(block).items()}
            for name, block in dict(clinical).items()
            if isinstance(block, dict)
        }
        disclaimer = dict(clinical).get("disclaimer")
        if isinstance(disclaimer, str) and disclaimer.strip():
            clinical_blocks["disclaimer"] = disclaimer.strip()
        return {
            "model_type": str(raw.get("model_type", DEFAULT_MODEL_TYPE)),
            "checkpoint_path": resolve_path(
                raw.get("checkpoint_path") or DEFAULT_CHECKPOINT
            ),
            "device": str(raw.get("device", "auto")),
            "image_size": int(raw.get("image_size", DEFAULT_IMAGE_SIZE)),
            "confidence_threshold": float(
                raw.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
            ),
            "class_names": class_names,
            "gradcam": {str(key): value for key, value in dict(gradcam).items()},
            "clinical": clinical_blocks,
            "source": str(path),
        }
    except Exception as exc:  # noqa: BLE001 - config problems must not crash the API
        LOGGER.warning("Could not read %s (%s); using defaults.", path, exc)
        return default_config()


def api_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return the bind settings for the inference API.

    The service has no authentication layer, so it binds to loopback unless the
    operator explicitly asks for wider exposure. Precedence, highest first:

    1. the ``TBSENTINEL_API_HOST`` / ``TBSENTINEL_API_PORT`` environment
       variables (the quickest way to opt in for a LAN demo),
    2. the ``api:`` block of ``configs/inference/inference.yaml``,
    3. the loopback defaults.

    Args:
        config_path: Optional override used by tests; defaults to
            ``configs/inference/inference.yaml``.

    Returns:
        Mapping with ``host``, ``port``, ``reload`` and ``log_level``.
    """
    settings: dict[str, Any] = {
        "host": DEFAULT_API_HOST,
        "port": DEFAULT_API_PORT,
        "reload": False,
        "log_level": DEFAULT_API_LOG_LEVEL,
    }

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    block = dict((_load_yaml_mapping(path) or {}).get("api") or {})
    if block.get("host"):
        settings["host"] = str(block["host"])
    if block.get("port"):
        settings["port"] = int(block["port"])
    if block.get("reload") is not None:
        settings["reload"] = bool(block["reload"])
    if block.get("log_level"):
        settings["log_level"] = str(block["log_level"])

    env_host = os.environ.get(API_HOST_ENV)
    if env_host:
        settings["host"] = env_host.strip()
    env_port = os.environ.get(API_PORT_ENV)
    if env_port:
        settings["port"] = int(env_port)

    if settings["host"] in WILDCARD_HOSTS:
        LOGGER.warning(
            "The inference API is binding to %s with no authentication: every "
            "machine that can reach this host can call /predict and /explain. "
            "Use loopback (%s) unless this is a deliberate LAN demo.",
            settings["host"],
            DEFAULT_API_HOST,
        )
    return settings
