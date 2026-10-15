"""Model performance metrics for the Metrics page.

Two artefacts are read, both written by the training pipeline and neither
invented in the UI:

* ``artifacts/outputs/multiclass_test_metrics.json`` — held-out metrics for the
  three-class model (Normal / TB / Pneumonia) that the API serves by default.
* ``artifacts/outputs/federated_*_results.json`` — the binary TB benchmark that
  compares the centralized baseline with FedAvg and FedProx. Its documented
  values live in ``docs/federated_learning.md`` and are mirrored here as a
  fallback so the page still renders when the artefacts are absent (e.g. a fresh
  clone). Those values are the best-validation metrics on the official
  1,800-image TBX11K binary validation split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "artifacts" / "outputs"

# Documented baseline (docs/federated_learning.md, artifacts/outputs/federated_comparison.md)
_CENTRALIZED_FALLBACK: dict[str, Any] = {
    "name": "Centralized baseline",
    "key": "centralized",
    "accuracy": 0.9878,
    "macro_f1": 0.9679,
    "auc": 0.9986,
    "sensitivity": 0.9050,
    "specificity": 0.9981,
    "confusion_matrix": [[1597, 3], [19, 181]],
    "rounds": None,
    "source": "documented baseline",
}

_FEDAVG_FALLBACK: dict[str, Any] = {
    "name": "Federated (FedAvg)",
    "key": "fedavg",
    "accuracy": 0.9811,
    "macro_f1": 0.9485,
    "auc": 0.9936,
    "sensitivity": 0.8350,
    "specificity": 0.9994,
    "confusion_matrix": [[1599, 1], [33, 167]],
    "rounds": 5,
    "source": "documented fallback",
}

_FEDPROX_FALLBACK: dict[str, Any] = {
    "name": "Federated (FedProx)",
    "key": "fedprox",
    "accuracy": 0.9761,
    "macro_f1": 0.9331,
    "auc": 0.9764,
    "sensitivity": 0.7850,
    "specificity": 1.0000,
    "confusion_matrix": [[1600, 0], [43, 157]],
    "rounds": 5,
    "source": "documented fallback",
}

METRIC_KEYS = ["accuracy", "macro_f1", "auc", "sensitivity", "specificity"]

# The three-class model that the API serves by default. Its held-out metrics are
# written by the multi-class evaluation script; nothing is hardcoded here so the
# page can never advertise numbers the checkpoint did not produce.
MULTICLASS_FILE = "multiclass_test_metrics.json"

METRIC_HELP = {
    "accuracy": (
        "Accuracy",
        "Share of all validation images classified correctly. High accuracy can "
        "mask poor TB detection when the dataset is imbalanced.",
    ),
    "macro_f1": (
        "Macro-F1",
        "Unweighted mean of the TB and Non-TB F1 scores. It penalises the model "
        "for failing on the minority TB class, so it is more honest than accuracy.",
    ),
    "auc": (
        "AUC-ROC",
        "Probability that a randomly chosen TB case is ranked above a random "
        "Non-TB case. Measures ranking quality independent of the threshold.",
    ),
    "sensitivity": (
        "Sensitivity (Recall)",
        "Of all truly TB-positive images, the share the model flags. In screening, "
        "missing a case (false negative) is the costliest error.",
    ),
    "specificity": (
        "Specificity",
        "Of all truly Non-TB images, the share the model correctly clears. Low "
        "specificity floods clinics with unnecessary follow-ups.",
    ),
}


def _load_federated(filename: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Load one federated results JSON, falling back to documented values."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return dict(fallback)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(fallback)

    best = payload.get("best_metrics") or {}
    if not best:
        return dict(fallback)
    return {
        "name": fallback["name"],
        "key": fallback["key"],
        "accuracy": float(best.get("accuracy", fallback["accuracy"])),
        "macro_f1": float(best.get("macro_f1", fallback["macro_f1"])),
        "auc": float(best.get("auc", fallback["auc"])),
        "sensitivity": float(best.get("sensitivity", fallback["sensitivity"])),
        "specificity": float(best.get("specificity", fallback["specificity"])),
        "confusion_matrix": best.get("confusion_matrix", fallback["confusion_matrix"]),
        "rounds": payload.get("best_round", fallback["rounds"]),
        "source": f"artifacts/outputs/{filename}",
    }


def load_comparison() -> list[dict[str, Any]]:
    """Return the three model rows: centralized baseline, FedAvg and FedProx."""
    return [
        dict(_CENTRALIZED_FALLBACK),
        _load_federated("federated_fedavg_results.json", _FEDAVG_FALLBACK),
        _load_federated("federated_fedprox_results.json", _FEDPROX_FALLBACK),
    ]


def delta_vs_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``delta_<metric>`` keys to federated rows relative to centralized."""
    baseline = rows[0]
    for row in rows[1:]:
        for key in METRIC_KEYS:
            row[f"delta_{key}"] = float(row[key]) - float(baseline[key])
    return rows


# ---------------------------------------------------------------------------
# Served three-class model (Normal / TB / Pneumonia)
# ---------------------------------------------------------------------------

# Per-class metric vocabulary used by the multi-class section. The one-vs-rest
# AUC is how a multi-class AUC is reported, so it is labelled explicitly.
MULTICLASS_METRIC_HELP: dict[str, tuple[str, str]] = {
    "precision": (
        "Precision",
        "Of the images the model labelled with this class, the share that truly "
        "belong to it. Low precision means avoidable over-calls.",
    ),
    "recall": (
        "Recall (Sensitivity)",
        "Of all images truly in this class, the share the model found. For TB this "
        "is the safety-critical number: a missed case is a false negative.",
    ),
    "f1": (
        "F1 score",
        "Harmonic mean of precision and recall for this class — low when either "
        "side is weak, so it exposes a class the model only half handles.",
    ),
    "auc": (
        "AUC-ROC (one-vs-rest)",
        "For each class against all others: the chance a true member of the class "
        "is ranked above a random non-member. Threshold-independent.",
    ),
}


def load_multiclass() -> dict[str, Any] | None:
    """Load the served model's held-out metrics, or ``None`` when unavailable.

    Reads ``artifacts/outputs/multiclass_test_metrics.json`` exactly as written by
    the multi-class evaluation script. Nothing is fabricated when the artefact is
    missing — the page states that the numbers are unavailable instead.
    """
    path = OUTPUTS_DIR / MULTICLASS_FILE
    if not path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    per_class_raw = payload.get("per_class") or {}
    if not per_class_raw:
        return None

    per_class: dict[str, dict[str, float | int]] = {}
    for name, stats in per_class_raw.items():
        stats = stats or {}
        per_class[str(name)] = {
            "precision": float(stats.get("precision") or 0.0),
            "recall": float(stats.get("recall") or 0.0),
            "f1": float(stats.get("f1") or 0.0),
            "auc": float(stats.get("auc_one_vs_rest") or 0.0),
            "support": int(stats.get("support") or 0),
        }

    class_names = [str(name) for name in (payload.get("class_names") or per_class)]
    return {
        "class_names": class_names,
        "samples": int(payload.get("samples") or 0),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "accuracy": float(payload.get("accuracy") or 0.0),
        "macro_f1": float(payload.get("macro_f1") or 0.0),
        "macro_auc": float(payload.get("macro_auc_one_vs_rest") or 0.0),
        "loss": float(payload.get("loss_unweighted_ce") or 0.0),
        "per_class": per_class,
        "confusion_matrix": payload.get("confusion_matrix") or [],
        "validation": payload.get("stored_validation_metrics") or {},
        "source": f"artifacts/outputs/{MULTICLASS_FILE}",
    }
