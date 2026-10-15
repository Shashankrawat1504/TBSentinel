"""Local persistence for saved patient assessments.

Uses SQLite (stdlib only) so records survive restarts and can be browsed,
searched, exported and deleted from the Patient History page. The database
lives under ``data/patient_records/`` inside the repository — it is a local
demo store, **not** a production EMR integration.

Schema (``assessments``)
------------------------
One row per saved assessment. Images are stored inline as BLOBs:

* ``image_jpeg``  — the uploaded radiograph, downscaled + re-encoded to keep
                    the database small while still being reviewable.
* ``overlay_png`` / ``heatmap_png`` — Grad-CAM artefacts returned by the API.

The probabilities are stored per class. ``prob_tb`` / ``prob_non_tb`` come from
the binary phase; ``prob_normal`` / ``prob_pneumonia`` were added for the
three-class model and are created by an in-place migration so databases written
before the upgrade keep working.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

# Repository root: frontend/tbui/storage.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = PROJECT_ROOT / "data" / "patient_records"
DB_PATH = RECORDS_DIR / "assessments.db"

_MAX_IMAGE_EDGE = 1280
_JPEG_QUALITY = 86

# Reentrant: public helpers call init_db() while already holding the lock.
_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id       TEXT UNIQUE NOT NULL,
    patient_id      TEXT NOT NULL,
    patient_name    TEXT,
    age             INTEGER,
    gender          TEXT,
    notes           TEXT,
    filename        TEXT,
    image_jpeg      BLOB,
    overlay_png     BLOB,
    heatmap_png     BLOB,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    prob_tb         REAL,
    prob_non_tb     REAL,
    prob_normal     REAL,
    prob_pneumonia  REAL,
    interpretation  TEXT,
    strategy        TEXT,
    latency_seconds REAL,
    explain_error   TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON assessments (patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON assessments (created_at DESC);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


# Columns added after the first released schema, with their migration DDL.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("prob_normal", "ALTER TABLE assessments ADD COLUMN prob_normal REAL"),
    ("prob_pneumonia", "ALTER TABLE assessments ADD COLUMN prob_pneumonia REAL"),
    ("interpretation", "ALTER TABLE assessments ADD COLUMN interpretation TEXT"),
)

# Result filter label -> stored ``predicted_class`` values.
PREDICTION_FILTERS: tuple[str, ...] = (
    "All",
    "Normal",
    "TB",
    "Pneumonia",
    "Non-TB",
)
_FILTER_VALUES: dict[str, tuple[str, ...]] = {
    "Normal": ("normal",),
    "TB": ("tb",),
    "Pneumonia": ("pneumonia",),
    "Non-TB": ("non_tb",),
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema (idempotent)."""
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(assessments)")}
    for column, statement in _ADDED_COLUMNS:
        if column not in existing:
            conn.execute(statement)


def init_db() -> None:
    """Create the database and table if needed, then apply migrations."""
    with _lock, _connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_patient_id() -> str:
    """Generate a unique patient id of the form ``PT-YYYYMMDD-####``."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"PT-{today}-"
    init_db()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT patient_id FROM assessments WHERE patient_id LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
    used = {int(r["patient_id"].rsplit("-", 1)[-1]) for r in rows
            if r["patient_id"].rsplit("-", 1)[-1].isdigit()}
    number = 1
    while number in used:
        number += 1
    return f"{prefix}{number:04d}"


def _next_record_id(conn: sqlite3.Connection) -> str:
    """Return the next sequential ``TSR-######`` id (safe after deletions)."""
    row = conn.execute(
        "SELECT record_id FROM assessments ORDER BY id DESC LIMIT 1"
    ).fetchone()
    last = 0
    if row:
        suffix = str(row["record_id"]).rsplit("-", 1)[-1]
        if suffix.isdigit():
            last = int(suffix)
    return f"TSR-{last + 1:06d}"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _probability(probabilities: dict[str, Any], class_name: str) -> float | None:
    """Return a class probability, or ``None`` when the model has no such class.

    ``None`` (SQL NULL) is stored instead of 0.0 so a three-class record does not
    display a meaningless "Non-TB 0 %" bar, and a legacy binary record does not
    display an invented Normal/Pneumonia bar.
    """
    value = probabilities.get(class_name)
    return None if value is None else float(value)


def compress_image(image_bytes: bytes) -> bytes:
    """Downscale an upload and re-encode it as JPEG for compact storage."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.thumbnail((_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def insert_assessment(record: dict[str, Any]) -> str:
    """Persist one assessment and return its human-readable ``record_id``.

    ``record`` expects the keys used by the dashboard session state plus the
    patient demographics: ``prediction``, ``explanation`` (optional),
    ``image_bytes``, ``filename``, ``strategy``, ``timestamp``,
    ``latency_seconds``, ``explain_error`` (optional) and the patient fields
    ``patient_id``, ``patient_name``, ``age``, ``gender``, ``notes``.
    """
    # Local import keeps this module import-light (and free of a cycle).
    from tbui.api import decode_base64_png, prediction_interpretation

    prediction = record["prediction"]
    explanation = record.get("explanation") or {}
    class_name = str(prediction.get("predicted_class_name", "")).lower()
    probabilities = prediction.get("probabilities", {}) or {}
    # The stored explanation is the prediction's own: the structured /explain copy
    # (confidence band, focus, differential, disclaimer) when it explains the
    # predicted class, otherwise the score-level /predict copy.
    interpretation = prediction_interpretation(record)

    overlay = explanation.get("overlay_base64")
    heatmap = explanation.get("heatmap_base64")
    created_at = record.get("timestamp") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    init_db()
    with _lock, _connect() as conn:
        record_id = _next_record_id(conn)
        conn.execute(
            """
            INSERT INTO assessments (
                record_id, patient_id, patient_name, age, gender, notes,
                filename, image_jpeg, overlay_png, heatmap_png,
                predicted_class, confidence, prob_tb, prob_non_tb,
                prob_normal, prob_pneumonia, interpretation,
                strategy, latency_seconds, explain_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                record.get("patient_id") or "UNSPECIFIED",
                record.get("patient_name") or None,
                record.get("age"),
                record.get("gender") or None,
                record.get("notes") or None,
                record.get("filename") or None,
                compress_image(record["image_bytes"]),
                decode_base64_png(str(overlay)) if overlay else None,
                decode_base64_png(str(heatmap)) if heatmap else None,
                class_name,
                float(prediction.get("confidence", 0.0)),
                _probability(probabilities, "tb"),
                _probability(probabilities, "non_tb"),
                _probability(probabilities, "normal"),
                _probability(probabilities, "pneumonia"),
                json.dumps(interpretation) if interpretation else None,
                record.get("strategy") or "Unknown",
                float(record.get("latency_seconds") or 0.0),
                record.get("explain_error"),
                created_at,
            ),
        )
    return record_id


def list_records(
    search: str = "",
    prediction_filter: str = "All",
) -> list[dict[str, Any]]:
    """Return assessment metadata (without image blobs), newest first.

    ``search`` matches patient id, patient name or record id (case-insensitive);
    ``prediction_filter`` is one of :data:`PREDICTION_FILTERS`.
    """
    init_db()
    query = (
        "SELECT id, record_id, patient_id, patient_name, age, gender, notes,"
        "       filename, predicted_class, confidence, prob_tb, prob_non_tb,"
        "       prob_normal, prob_pneumonia, interpretation, strategy,"
        "       latency_seconds, explain_error, created_at"
        " FROM assessments"
    )
    clauses: list[str] = []
    params: list[Any] = []
    term = (search or "").strip()
    if term:
        clauses.append("(patient_id LIKE ? OR patient_name LIKE ? OR record_id LIKE ?)")
        like = f"%{term}%"
        params.extend([like, like, like])
    selected_classes = _FILTER_VALUES.get(prediction_filter)
    if selected_classes:
        placeholders = ", ".join("?" for _ in selected_classes)
        clauses.append(f"predicted_class IN ({placeholders})")
        params.extend(selected_classes)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"

    with _lock, _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_record(record_id: str) -> dict[str, Any] | None:
    """Return one full record (including image blobs) or ``None``."""
    init_db()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE record_id = ?", (record_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_record(record_id: str) -> bool:
    """Delete one record; returns True when a row was removed."""
    init_db()
    with _lock, _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM assessments WHERE record_id = ?", (record_id,)
        )
        return cursor.rowcount > 0


def get_stats() -> dict[str, Any]:
    """Aggregate counters used by the dashboard KPI cards."""
    init_db()
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                                     AS total,
                SUM(CASE WHEN predicted_class = 'normal'
                         THEN 1 ELSE 0 END)                                  AS normal,
                SUM(CASE WHEN predicted_class = 'tb' THEN 1 ELSE 0 END)      AS tb,
                SUM(CASE WHEN predicted_class = 'pneumonia'
                         THEN 1 ELSE 0 END)                                  AS pneumonia,
                SUM(CASE WHEN predicted_class = 'non_tb'
                         THEN 1 ELSE 0 END)                                  AS non_tb,
                COUNT(DISTINCT patient_id)                                   AS patients,
                AVG(confidence)                                              AS avg_confidence,
                MAX(created_at)                                              AS last_at
            FROM assessments
            """
        ).fetchone()
    total = int(row["total"] or 0)
    tb = int(row["tb"] or 0)
    normal = int(row["normal"] or 0)
    pneumonia = int(row["pneumonia"] or 0)
    return {
        "total": total,
        "normal": normal,
        "tb": tb,
        "pneumonia": pneumonia,
        "non_tb": int(row["non_tb"] or 0),
        "patients": int(row["patients"] or 0),
        "avg_confidence": float(row["avg_confidence"] or 0.0),
        "tb_rate": (tb / total) if total else 0.0,
        "normal_rate": (normal / total) if total else 0.0,
        "pneumonia_rate": (pneumonia / total) if total else 0.0,
        "last_at": row["last_at"],
    }


def recent_records(limit: int = 5) -> list[dict[str, Any]]:
    """Newest ``limit`` records (metadata only) for the dashboard preview."""
    return list_records()[:limit]
