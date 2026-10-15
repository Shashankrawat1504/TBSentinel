"""Where a Grad-CAM map actually looks: regions, coverage and share of activation.

The heatmap on its own is a picture; a clinician reading an automated screen needs
a sentence. This module turns a normalised activation map into a small, stable
description of the *focus* of the map — how much of the radiograph is active, the
peak activation, where the activation sits (side and vertical zone) and whether it
is bilateral, lateralised or central.

Conventions and honest limits
-----------------------------
* **Radiological convention.** A frontal chest radiograph is read as if facing the
  patient, so the left side of the *image* is the patient's **right** side. The
  side labels here follow that convention, and :data:`CONVENTION_NOTE` carries the
  reminder into every report so a reader is never left guessing.
* **Image-space approximation.** Zones come from thirds of the 300x300 activation
  map, aligned to the radiograph. The map is produced by the last convolutional
  block, so it is coarse — a hotspot indicates a region, not a lesion. The caveat
  travels with the description (:data:`COARSE_MAP_CAVEAT`) rather than being
  silently dropped.
* **Flat maps are reported as flat.** When no pixel exceeds the threshold the
  result says so instead of inventing a focus region.

Everything here is pure NumPy and deterministic, so it is unit-testable without a
model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_THRESHOLD = 0.5

# A map counts as bilateral when each side holds at least this share of the hot
# pixels; below it the activation is treated as lateralised to one side.
BILATERAL_SHARE = 0.20
# Centre strip (45–55 % of image width) share that makes a map read as central.
CENTRAL_SHARE = 0.35
# Side label boundaries on the normalised x axis.
_CENTRE_LOW = 0.45
_CENTRE_HIGH = 0.55

CONVENTION_NOTE = (
    "Side labels follow the radiological convention for a frontal radiograph: "
    "the left of the image is the patient's right side."
)

COARSE_MAP_CAVEAT = (
    "The heatmap is generated at the resolution of the model's last convolutional "
    "block and then upsampled, so it localises a region — not a lesion. Read it as "
    "'where the model looked', never as a segmentation of disease."
)


def _zone_for(y: float) -> str:
    """Vertical third of the image for a normalised y coordinate (0 = top)."""
    if y < 1.0 / 3.0:
        return "upper"
    if y < 2.0 / 3.0:
        return "mid"
    return "lower"


def _normalise(heatmap: np.ndarray) -> np.ndarray:
    """Return the map as float32 in [0, 1], or zeros when it carries no signal."""
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D activation map, got shape {array.shape}.")
    peak = float(np.nanmax(array)) if array.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / peak, 0.0, 1.0).astype(np.float32)


def describe_focus(
    heatmap: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Describe the focus of one normalised Grad-CAM map.

    Args:
        heatmap: 2D activation map, normalised to [0, 1] (``explain()`` output).
        threshold: Activation level above which a pixel counts as "hot".

    Returns:
        Mapping with ``available``, ``summary``, ``peak``, ``coverage``,
        ``active_pixels``, ``centroid`` (normalised x/y), ``bounding_box``,
        ``side``, ``zone``, ``region_label``, ``side_shares``, ``convention`` and
        ``caveat``. ``available`` is ``False`` for a flat map, and the descriptive
        fields then carry the "no dominant focus" text.
    """
    array = _normalise(heatmap)
    height, width = array.shape
    threshold = float(min(max(threshold, 0.0), 1.0))

    hot = array >= threshold
    active_pixels = int(np.count_nonzero(hot))
    peak = float(array.max()) if array.size else 0.0
    coverage = float(active_pixels / array.size) if array.size else 0.0

    if active_pixels == 0:
        return {
            "available": False,
            "summary": (
                "The activation map is flat — no region exceeded "
                f"{threshold:.0%} activation — so there is no dominant focus to "
                "report for this class. Treat the score as unsupported by a "
                "localised imaging feature and over-read the radiograph."
            ),
            "threshold": threshold,
            "peak": peak,
            "coverage": coverage,
            "active_pixels": 0,
            "centroid": None,
            "bounding_box": None,
            "side": None,
            "zone": None,
            "region_label": None,
            "side_shares": {"patient_right": 0.0, "patient_left": 0.0},
            "convention": CONVENTION_NOTE,
            "caveat": COARSE_MAP_CAVEAT,
        }

    rows, cols = np.nonzero(hot)
    weights = array[rows, cols]
    total_weight = float(weights.sum()) or 1.0
    centroid_x = float((cols * weights).sum() / total_weight / (width - 1 or 1))
    centroid_y = float((rows * weights).sum() / total_weight / (height - 1 or 1))

    # Image-space sides: left half of the image is the patient's right side.
    left_half = int(np.count_nonzero(cols < width / 2))
    right_half = active_pixels - left_half
    patient_right_share = left_half / active_pixels
    patient_left_share = right_half / active_pixels

    centre_band = int(
        np.count_nonzero((cols >= _CENTRE_LOW * width) & (cols <= _CENTRE_HIGH * width))
    )
    centre_share = centre_band / active_pixels

    if (
        patient_right_share >= BILATERAL_SHARE
        and patient_left_share >= BILATERAL_SHARE
    ):
        side = "bilateral"
    elif centre_share >= CENTRAL_SHARE:
        side = "central"
    elif centroid_x < _CENTRE_HIGH:
        side = "patient_right"
    else:
        side = "patient_left"

    zone = _zone_for(centroid_y)

    if side == "bilateral":
        region_label = f"both lungs, predominantly the {zone} zones"
    elif side == "central":
        region_label = f"the central {zone} region of the image"
    else:
        readable = "right" if side == "patient_right" else "left"
        region_label = f"the patient's {readable} {zone} zone"

    sentences = [
        f"Activation is concentrated in {region_label}: peak {peak:.0%} "
        f"activation, {coverage:.1%} of the image above the {threshold:.0%} "
        "threshold."
    ]
    if side == "bilateral":
        sentences.append("Activation is split across both lungs.")
    elif side == "central":
        sentences.append(
            "Activation is central, which on a frontal radiograph overlaps the "
            "mediastinum and both para-hilar regions."
        )

    return {
        "available": True,
        "summary": " ".join(sentences),
        "threshold": threshold,
        "peak": peak,
        "coverage": coverage,
        "active_pixels": active_pixels,
        "centroid": {"x": round(centroid_x, 4), "y": round(centroid_y, 4)},
        "bounding_box": {
            "x0": round(float(cols.min() / (width - 1 or 1)), 4),
            "y0": round(float(rows.min() / (height - 1 or 1)), 4),
            "x1": round(float(cols.max() / (width - 1 or 1)), 4),
            "y1": round(float(rows.max() / (height - 1 or 1)), 4),
        },
        "side": side,
        "zone": zone,
        "region_label": region_label,
        "side_shares": {
            "patient_right": round(patient_right_share, 4),
            "patient_left": round(patient_left_share, 4),
        },
        "convention": CONVENTION_NOTE,
        "caveat": COARSE_MAP_CAVEAT,
    }
