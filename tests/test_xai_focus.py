"""Tests for the Grad-CAM rendering helpers and the focus-region reader.

These cover the explainability quality layer without needing a checkpoint: the
activation maps are synthetic, so every assertion is about the transformation the
model output goes through before a clinician sees it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.xai.focus import describe_focus
from src.xai.gradcam import (
    DEFAULT_COLORMAP,
    colorize_heatmap,
    refine_heatmap,
    resolve_colormap,
)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def test_refine_heatmap_keeps_the_marked_region_hot_and_drops_the_outlier() -> None:
    """A single extreme pixel must not drown a genuinely marked region."""
    raw = np.zeros((64, 64), dtype=np.float32)
    raw[10:20, 10:20] = 0.6
    raw[30, 30] = 5.0  # a lone extreme pixel, as raw CAMs often have

    refined = refine_heatmap(raw, blur_kernel=9, clip_percentile=99.0)

    assert refined.shape == raw.shape
    assert refined.dtype == np.float32
    assert refined.min() >= 0.0
    assert refined.max() == pytest.approx(1.0, abs=1e-5)
    # The marked region stays above the standard 0.5 hot threshold ...
    assert float(refined[12:18, 12:18].min()) >= 0.5
    # ... the untouched background stays cold ...
    assert float(refined[40:60, 40:60].max()) < 0.2
    # ... and the lone pixel is smoothed away instead of owning the map.
    assert float(refined[29:32, 29:32].max()) < 0.2


def test_refine_heatmap_smooths_pixel_to_pixel_noise() -> None:
    """Smoothing must remove the blockiness of the last-convolution map."""
    rng = np.random.default_rng(0)
    raw = rng.random((64, 64)).astype(np.float32)

    refined = refine_heatmap(raw, blur_kernel=9, clip_percentile=99.0)

    roughness = lambda array: float(np.abs(np.diff(array)).mean())
    assert roughness(refined) < roughness(raw) * 0.5


def test_refine_heatmap_can_disable_the_percentile_stretch() -> None:
    """Documents why the shipped default has no percentile stretch."""
    # A small hot spot on a wide, dim plateau: the 90th percentile sits at the
    # plateau, so the stretch lifts the whole plateau to full intensity.
    raw = np.zeros((64, 64), dtype=np.float32)
    raw[0:2, :] = 1.0
    raw[2:64, :] = 0.4

    faithful = refine_heatmap(raw, blur_kernel=1, clip_percentile=100.0)
    stretched = refine_heatmap(raw, blur_kernel=1, clip_percentile=90.0)

    # Without the stretch only the hot spot clears the threshold ...
    assert float((faithful >= 0.5).mean()) == pytest.approx(2 / 64, abs=0.02)
    # ... while the stretch inflates coverage, which is why it is off by default.
    assert float((stretched >= 0.5).mean()) > 0.9


def test_refine_heatmap_keeps_a_flat_map_flat() -> None:
    refined = refine_heatmap(np.zeros((16, 16), dtype=np.float32), blur_kernel=9)

    assert refined.shape == (16, 16)
    assert float(refined.max()) == 0.0


def test_refine_heatmap_rejects_a_non_2d_map() -> None:
    with pytest.raises(ValueError):
        refine_heatmap(np.zeros((8, 8, 3), dtype=np.float32))


def test_colorize_heatmap_returns_a_readable_rgb_image() -> None:
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[0:8, :] = 1.0  # hot band
    heatmap[24:32, :] = 0.05  # cold band

    coloured = colorize_heatmap(heatmap, DEFAULT_COLORMAP)

    assert coloured.shape == (32, 32, 3)
    assert coloured.dtype == np.uint8
    hot = coloured[0:8].reshape(-1, 3).mean(axis=0)
    cold = coloured[24:32].reshape(-1, 3).mean(axis=0)
    assert not np.allclose(hot, cold), "hot and cold regions must differ in colour"


def test_resolve_colormap_round_trips_names_and_codes() -> None:
    code, name = resolve_colormap("turbo")
    assert name == "turbo"
    assert code == getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)

    assert resolve_colormap("nonsense") == (cv2.COLORMAP_JET, "jet")
    assert resolve_colormap(cv2.COLORMAP_JET)[1] == "jet"


# ---------------------------------------------------------------------------
# Focus regions
# ---------------------------------------------------------------------------


def test_focus_of_a_left_upper_blob_is_the_patients_right_upper_zone() -> None:
    """Image left is the patient's right: the convention must be applied."""
    heatmap = np.zeros((300, 300), dtype=np.float32)
    heatmap[40:100, 30:90] = 0.9

    focus = describe_focus(heatmap)

    assert focus["available"] is True
    assert focus["side"] == "patient_right"
    assert focus["zone"] == "upper"
    assert focus["region_label"] == "the patient's right upper zone"
    assert focus["centroid"]["x"] < 0.5
    assert focus["convention"]
    # The summary is read by clinicians, so it must be well formed: exactly one
    # closing period and no run-on sentence.
    assert focus["summary"].endswith("threshold.")
    assert ".." not in focus["summary"]
    assert " Activation" not in focus["summary"]


def test_focus_of_a_right_lower_blob_is_the_patients_left_lower_zone() -> None:
    heatmap = np.zeros((300, 300), dtype=np.float32)
    heatmap[210:270, 220:280] = 0.8

    focus = describe_focus(heatmap)

    assert focus["side"] == "patient_left"
    assert focus["zone"] == "lower"
    assert focus["centroid"]["y"] > 0.6
    assert focus["bounding_box"]["x1"] > 0.7


def test_focus_detects_bilateral_activation() -> None:
    heatmap = np.zeros((300, 300), dtype=np.float32)
    heatmap[100:180, 20:120] = 0.9
    heatmap[100:180, 180:280] = 0.9

    focus = describe_focus(heatmap)

    assert focus["side"] == "bilateral"
    assert "both lungs" in focus["region_label"]
    assert focus["side_shares"]["patient_left"] > 0.2
    assert focus["side_shares"]["patient_right"] > 0.2


def test_focus_of_a_central_band_reads_as_central() -> None:
    # Hugging the midline but asymmetric: a band that straddles the midline
    # evenly is bilateral, which is the more useful reading and is covered above.
    heatmap = np.zeros((300, 300), dtype=np.float32)
    heatmap[100:200, 136:150] = 0.95

    focus = describe_focus(heatmap)

    assert focus["side"] == "central"
    assert "central" in focus["region_label"]
    assert focus["coverage"] > 0.0


def test_focus_of_a_flat_map_is_reported_as_absent() -> None:
    focus = describe_focus(np.zeros((300, 300), dtype=np.float32))

    assert focus["available"] is False
    assert "flat" in focus["summary"]
    assert focus["region_label"] is None
    assert focus["active_pixels"] == 0


def test_focus_coverage_matches_the_threshold() -> None:
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[0:50, :] = 1.0  # exactly half the image is hot

    focus = describe_focus(heatmap, threshold=0.5)

    assert focus["coverage"] == pytest.approx(0.5, abs=1e-6)
    assert focus["active_pixels"] == 5000
    assert focus["peak"] == pytest.approx(1.0)

    # A stricter threshold excludes the dimmer halo, so coverage must fall; the
    # map itself is re-normalised, so 'absent' is reserved for a flat map.
    haloed = np.zeros((100, 100), dtype=np.float32)
    haloed[0:50, :] = 1.0
    haloed[50:100, :] = 0.4

    strict = describe_focus(haloed, threshold=0.5)
    loose = describe_focus(haloed, threshold=0.3)

    assert strict["coverage"] == pytest.approx(0.5, abs=1e-6)
    assert loose["coverage"] == pytest.approx(1.0, abs=1e-6)
