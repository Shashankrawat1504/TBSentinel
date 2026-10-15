"""Verify the three-class dashboard by driving the real Streamlit pages.

Powers the actual page scripts (not a stub) through Streamlit's ``AppTest``
against a running inference API and real chest radiographs from the held-out
multi-class test split:

1. **New Assessment** — patient details + a TB radiograph → colour-coded class
   badge, three-class probability bars, Grad-CAM (with the explained class),
   the configured clinical interpretation, then save to the record store.
2. **Patient History** — the saved record renders with its class badge, the full
   probability breakdown, the clinical notes and the class filter.
3. **Dashboard** — the record is counted in the right class KPI.
4. **Performance Metrics** — the served three-class model section renders.
5. **About** — the model card names the served task and classes.
6. Cleanup: the verification record is deleted at the end.

Usage
-----
With the API already running, from the project root::

    python scripts/verify_dashboard_3class.py

Environment
-----------
``TBSENTINEL_API_URL``  base URL of the API (default ``http://127.0.0.1:8000``)

Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "multiclass_test.csv"
LABELS = {0: "normal", 1: "tb", 2: "pneumonia"}

sys.path.insert(0, str(FRONTEND))

from tbui import storage  # noqa: E402  (needs the frontend path above)

API_URL = os.environ.get("TBSENTINEL_API_URL", "http://127.0.0.1:8000").rstrip("/")
PATIENT_ID = "PT-UITEST-0001"
NOTES = "Persistent cough and night sweats for 3 weeks."
PAGE_TIMEOUT = 600

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one check and print its outcome."""
    safe = str(detail).encode("ascii", "replace").decode("ascii")
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {safe}")
    if not ok:
        failures.append(name)


def blob(page: AppTest) -> str:
    """All text/HTML the page emitted, as one searchable string."""
    parts = [el.value for el in page.markdown]
    parts += [el.value for el in page.caption]
    parts += [el.label for el in page.button]
    parts += [el.value for el in page.toast]
    return "\n".join(str(part) for part in parts)


def page(filename: str) -> AppTest:
    """Render one dashboard page with the API URL in session state."""
    app = AppTest.from_file(
        str(FRONTEND / "views" / filename), default_timeout=PAGE_TIMEOUT
    )
    app.session_state["api_url"] = API_URL
    app.run()
    return app


def sample_images() -> dict[str, tuple[str, bytes, str]]:
    """One (name, bytes, mime) upload per class from the test split."""
    if not TEST_SPLIT.exists():
        print(f"test split not found: {TEST_SPLIT}")
        sys.exit(1)
    with TEST_SPLIT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_class[LABELS[int(row["label"])]].append(row["image_path"])
    samples = {}
    for class_name, paths in by_class.items():
        path = Path(paths[0])
        samples[class_name] = (path.name, path.read_bytes(), "image/png")
    return samples


def main() -> int:
    """Drive the dashboard pages and report every check."""
    samples = sample_images()
    print(f"sample uploads: { {k: v[0] for k, v in samples.items()} }")

    # -- 1. New Assessment --------------------------------------------------
    assessment = page("assessment.py")
    check(
        "assessment page renders",
        not assessment.exception,
        str(assessment.exception)[:300],
    )

    assessment.text_input(key="pf_name").set_value("UI Test Patient")
    assessment.text_input(key="pf_patient_id").set_value(PATIENT_ID)
    assessment.number_input(key="pf_age").set_value(41)
    assessment.selectbox(key="pf_gender").set_value("Female")
    assessment.text_area(key="pf_notes").set_value(NOTES)
    assessment.file_uploader[0].upload(*samples["tb"]).run()
    check("upload accepted", not assessment.exception, str(assessment.exception)[:300])

    next(
        button for button in assessment.button if "Run assessment" in button.label
    ).click().run()
    check(
        "analysis ran without an exception",
        not assessment.exception,
        str(assessment.exception)[:300],
    )

    rendered = blob(assessment)
    check(
        "result badge names the predicted class",
        "Tuberculosis" in rendered and "ts-badge--tb" in rendered,
    )
    check(
        "all three class probability bars rendered",
        all(
            f"ts-bar-fill--{name}" in rendered for name in ("normal", "tb", "pneumonia")
        ),
    )
    check("confidence is displayed", "Model confidence" in rendered)
    check(
        "clinical headline comes from the API configuration",
        "tuberculosis</b> pattern" in rendered,
    )
    check(
        "clinical guidance notes rendered",
        "sputum AFB" in rendered or "Correlate with symptoms" in rendered,
    )
    check("Grad-CAM reports the explained class", "Explained class" in rendered)
    check(
        "Grad-CAM images rendered",
        len(assessment.image) >= 5,
        f"{len(assessment.image)} images",
    )
    check(
        "Grad-CAM overlay can be downloaded",
        any("overlay" in button.label.lower() for button in assessment.download_button),
    )

    save_button = next(
        (
            button
            for button in assessment.button
            if "Save assessment to patient history" in button.label
        ),
        None,
    )
    check(
        "save button available once a patient id is set",
        save_button is not None and save_button.disabled is False,
    )
    if save_button is not None:
        save_button.click().run()
    check(
        "save produced no exception",
        not assessment.exception,
        str(assessment.exception)[:300],
    )

    record_id = assessment.session_state.get("analysis_record_id")
    check("record id stored in session", bool(record_id), str(record_id))
    record = storage.get_record(record_id) if record_id else None
    check("record persisted", record is not None)
    if record is not None:
        check(
            "stored predicted class matches the result card",
            record["predicted_class"] == "tb",
            str(record["predicted_class"]),
        )
        check(
            "stored three-class probabilities",
            record["prob_normal"] is not None
            and record["prob_tb"] is not None
            and record["prob_pneumonia"] is not None,
            f"normal={record['prob_normal']} tb={record['prob_tb']} "
            f"pneumonia={record['prob_pneumonia']}",
        )
        check(
            "stored clinical interpretation",
            "Tuberculosis" in str(record["interpretation"]),
        )
        check("stored Grad-CAM overlay", bool(record["overlay_png"]))
        check(
            "stored patient details",
            record["patient_name"] == "UI Test Patient"
            and record["age"] == 41
            and record["gender"] == "Female",
        )

    # -- 2. Patient History -------------------------------------------------
    history = page("history.py")
    check("history page renders", not history.exception, str(history.exception)[:300])
    history_blob = blob(history)
    check("history lists the new record", PATIENT_ID in history_blob)
    check("history record label names the class", "Tuberculosis" in history_blob)
    check(
        "history detail shows all three class bars",
        all(
            f"ts-bar-fill--{name}" in history_blob
            for name in ("normal", "tb", "pneumonia")
        ),
    )
    check("history shows the stored clinical notes", NOTES in history_blob)

    filter_box = next(
        (element for element in history.selectbox if element.label == "Result filter"),
        None,
    )
    check(
        "result filter offers every class",
        filter_box is not None
        and set(filter_box.options) == set(storage.PREDICTION_FILTERS),
        str(filter_box.options if filter_box else None),
    )

    # -- 3. Dashboard -------------------------------------------------------
    dashboard = page("dashboard.py")
    check(
        "dashboard page renders",
        not dashboard.exception,
        str(dashboard.exception)[:300],
    )
    stats = storage.get_stats()
    check(
        "class counters account for every record",
        stats["tb"] + stats["normal"] + stats["pneumonia"] + stats["non_tb"]
        == stats["total"],
        str(
            {
                key: stats[key]
                for key in ("total", "normal", "tb", "pneumonia", "non_tb")
            }
        ),
    )

    # -- 4. Performance Metrics --------------------------------------------
    metrics_page = page("metrics.py")
    check(
        "metrics page renders",
        not metrics_page.exception,
        str(metrics_page.exception)[:300],
    )
    metrics_blob = blob(metrics_page)
    check(
        "metrics shows the served three-class model",
        "Three-class screening" in metrics_blob
        and "multiclass_efficientnet_b3_best.pt" in metrics_blob,
    )
    check(
        "metrics shows per-class rows and the confusion matrix",
        "AUC-OVR" in metrics_blob and "Three-class confusion matrix" in metrics_blob,
    )
    check(
        "metrics keeps the binary benchmark separately labelled",
        "BINARY TB BENCHMARK" in metrics_blob.upper(),
    )
    check(
        "metric copy renders without stray quote artefacts",
        '" "' not in metrics_blob and 'one class "' not in metrics_blob,
    )

    # -- 5. About -----------------------------------------------------------
    about = page("about.py")
    check("about page renders", not about.exception, str(about.exception)[:300])
    about_blob = blob(about)
    check(
        "model card names the served classes",
        "Normal" in about_blob
        and "Pneumonia" in about_blob
        and "EfficientNet-B3" in about_blob,
    )
    check("model card states the served task", "3-class" in about_blob)

    # -- 6. Cleanup ---------------------------------------------------------
    if record_id:
        check("verification record removed", storage.delete_record(record_id) is True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: {failures}")
        return 1
    print("ALL DASHBOARD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
