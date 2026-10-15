"""Class-level clinical interpretation for predictions.

The wording lives in ``configs/inference/inference.yaml`` (``clinical:`` block)
so medical copy can be reviewed or localised without touching code, and the API
and the dashboard cannot drift apart. Everything here degrades gracefully: when
no text is configured for a class the caller gets ``None`` and renders no
interpretation instead of failing.

Two levels are produced:

* :func:`build_interpretation` — the configured copy for one class (label, tone,
  headline, plausibility notes).
* :func:`build_explanation` — the same copy plus the structured evidence a
  clinician needs to judge the screen: a confidence band with its runner-up
  margin, the Grad-CAM focus, the differential over the remaining classes, any
  generated flags, and the standing disclaimer.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("inference.clinical")

# Confidence bands. Deliberately simple, documented thresholds: a screen is only
# called high confidence when the model is both confident and separated from the
# runner-up class.
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MODERATE = 0.60
# Below this top-1 minus top-2 gap the classes are treated as near-tied, which
# caps the band at Moderate: argmax alone is not a clinical separation.
NEAR_TIE_MARGIN = 0.10
# A competing class at or above this probability is reported as a live
# differential rather than a footnote.
COMPETING_PROBABILITY = 0.30
UNLIKELY_PROBABILITY = 0.15

DEFAULT_DISCLAIMER = (
    "TBSentinel is a research prototype. This output is a model screen, not a "
    "diagnosis: it must be interpreted by a qualified clinician alongside the "
    "patient's history, examination and confirmatory tests."
)


def format_headline(
    template: str,
    class_name: str,
    confidence: float,
    probabilities: dict[str, float],
) -> str:
    """Fill a configured headline template, leaving unknown placeholders intact.

    Available placeholders: ``{confidence}``, ``{class_name}`` and
    ``{prob_<class_name>}`` for every class in ``probabilities``.
    """
    values: dict[str, Any] = {
        "confidence": confidence,
        "class_name": class_name,
    }
    for name, value in probabilities.items():
        values[f"prob_{name}"] = value
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        LOGGER.warning(
            "Headline template for %r could not be formatted (%s).", class_name, exc
        )
        return template


def build_interpretation(
    class_name: str,
    confidence: float,
    probabilities: dict[str, float],
    clinical_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the structured clinical summary for one prediction.

    Args:
        class_name: Predicted class as returned by the model.
        confidence: Probability of the predicted class.
        probabilities: Full class probability map.
        clinical_config: ``clinical`` block from the inference configuration.

    Returns:
        Mapping with ``class_name``, ``label``, ``tone``, ``headline`` and
        ``notes``, or ``None`` when the class has no configured text.
    """
    if not clinical_config:
        return None
    block = clinical_config.get(class_name)
    if not isinstance(block, dict) or not block:
        LOGGER.warning("No clinical interpretation configured for %r.", class_name)
        return None

    template = str(block.get("headline", ""))
    raw_notes = block.get("notes", []) or []
    return {
        "class_name": class_name,
        "label": str(block.get("label", class_name)),
        "tone": str(block.get("tone", "info")),
        "headline": (
            format_headline(template, class_name, confidence, probabilities)
            if template
            else ""
        ),
        "notes": [str(note) for note in raw_notes],
    }


def class_label(class_name: str, clinical_config: dict[str, Any] | None = None) -> str:
    """Display label for a class, using configured copy when available."""
    block = (clinical_config or {}).get(class_name)
    if isinstance(block, dict) and block.get("label"):
        return str(block["label"])
    return class_name.replace("_", " ").title()


def confidence_band(
    confidence: float,
    probabilities: dict[str, float] | None = None,
    class_name: str | None = None,
    clinical_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a confidence score and report the runner-up margin.

    The band is derived from the winning probability, and then capped at
    *Moderate* when the runner-up class sits within :data:`NEAR_TIE_MARGIN`: a
    0.44 / 0.42 split is not a high-confidence answer even though argmax had to
    pick one class.

    Args:
        confidence: Probability of the predicted class.
        probabilities: Full class probability map (optional but recommended).
        class_name: Predicted class, so it can be excluded from the runner-up.
        clinical_config: Optional clinical block used to label the runner-up.

    Returns:
        Mapping with ``value``, ``level``, ``tone``, ``explanation``, ``margin``,
        ``runner_up``, ``runner_up_label``, ``runner_up_probability`` and
        ``near_tie``.
    """
    value = float(confidence)
    others = {
        str(name): float(prob)
        for name, prob in (probabilities or {}).items()
        if str(name) != str(class_name)
    }
    runner_up = max(others, key=others.get) if others else None
    runner_up_probability = others[runner_up] if runner_up else None
    margin = value - runner_up_probability if runner_up_probability is not None else None
    near_tie = margin is not None and margin < NEAR_TIE_MARGIN

    if value >= CONFIDENCE_HIGH:
        level, tone = "High", "success"
        explanation = (
            f"High confidence ({value:.1%}): the model is both decisive and "
            "separated from the competing classes."
        )
    elif value >= CONFIDENCE_MODERATE:
        level, tone = "Moderate", "warning"
        explanation = (
            f"Moderate confidence ({value:.1%}): the leading class is clear but "
            "not decisive — read the Grad-CAM region before trusting the call."
        )
    else:
        level, tone = "Low", "error"
        explanation = (
            f"Low confidence ({value:.1%}): the classes are close to tied, so "
            "treat this screen as indeterminate and over-read it."
        )

    if near_tie and level == "High":
        level, tone = "Moderate", "warning"
        explanation += (
            " Capped to moderate because the runner-up class is within "
            f"{margin:.1%} of the prediction."
        )

    return {
        "value": value,
        "level": level,
        "tone": tone,
        "explanation": explanation,
        "margin": margin,
        "runner_up": runner_up,
        "runner_up_label": (
            class_label(runner_up, clinical_config) if runner_up else None
        ),
        "runner_up_probability": runner_up_probability,
        "near_tie": bool(near_tie),
    }


def build_differential(
    class_name: str,
    probabilities: dict[str, float],
    clinical_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Describe the remaining classes as a ranked differential.

    Returns one row per non-predicted class, most probable first, with the row
    labelled as a live alternative, a less likely possibility or an unlikely
    class so the reader knows what still has to be excluded.
    """
    rows: list[dict[str, Any]] = []
    for name, probability in sorted(
        ((str(k), float(v)) for k, v in probabilities.items() if str(k) != str(class_name)),
        key=lambda item: item[1],
        reverse=True,
    ):
        if probability >= COMPETING_PROBABILITY:
            note = (
                f"Remains a competing possibility at {probability:.1%}; do not "
                "exclude it on this screen alone."
            )
        elif probability >= UNLIKELY_PROBABILITY:
            note = (
                f"Less likely ({probability:.1%}) but not excluded by this "
                "screen; correlate clinically."
            )
        else:
            note = f"Unlikely on this image ({probability:.1%})."
        rows.append(
            {
                "class_name": name,
                "label": class_label(name, clinical_config),
                "probability": probability,
                "note": note,
            }
        )
    return rows


def _focus_section(focus: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise the focus block so clients always read the same keys."""
    if not focus:
        return {
            "available": False,
            "summary": (
                "Not computed for this response. The Grad-CAM view (or the "
                "/explain endpoint) reports which regions drove this call."
            ),
            "reason": "not_computed",
        }
    return dict(focus)


def build_explanation(
    class_name: str,
    confidence: float,
    probabilities: dict[str, float],
    clinical_config: dict[str, Any] | None,
    focus: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble the full structured explanation returned with a prediction.

    Args:
        class_name: Predicted class as returned by the model.
        confidence: Probability of the predicted class.
        probabilities: Full class probability map.
        clinical_config: ``clinical`` block from the inference configuration.
        focus: Optional :func:`src.xai.focus.describe_focus` output for the
            Grad-CAM map that explains this prediction.

    Returns:
        The :func:`build_interpretation` payload extended with ``confidence``
        (banded), ``focus``, ``differential``, ``flags`` and ``disclaimer`` — or
        ``None`` when the class has no configured text, in which case the caller
        renders no interpretation.
    """
    interpretation = build_interpretation(
        class_name, confidence, probabilities, clinical_config
    )
    if interpretation is None:
        return None

    band = confidence_band(
        confidence, probabilities, class_name=class_name, clinical_config=clinical_config
    )
    differential = build_differential(class_name, probabilities, clinical_config)
    focus_section = _focus_section(focus)

    flags: list[str] = []
    if band["level"] == "Low":
        flags.append(
            "Confidence is low: the leading classes are close to tied and the "
            "screen should be treated as indeterminate."
        )
    elif band["near_tie"]:
        flags.append(
            "Near-tie between the leading classes — the winning class has a "
            f"margin of only {band['margin']:.1%}."
        )
    if focus_section.get("available") is False and focus_section.get("reason") != "not_computed":
        flags.append(
            "The activation map is flat: the model did not concentrate on any "
            "single region for this class."
        )

    disclaimer = DEFAULT_DISCLAIMER
    if clinical_config:
        configured = clinical_config.get("disclaimer")
        if isinstance(configured, str) and configured.strip():
            disclaimer = configured.strip()

    return {
        **interpretation,
        "confidence": band,
        "focus": focus_section,
        "differential": differential,
        "flags": flags,
        "disclaimer": disclaimer,
    }
