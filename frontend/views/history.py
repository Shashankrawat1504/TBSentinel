"""Patient History — searchable record browser with detail view and export."""

from __future__ import annotations

import csv
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from tbui import api, components, storage, theme

api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
health = components.cached_health(api_url)

components.render_topbar(
    "Patient History",
    "Browse, search and export saved patient assessments",
    health,
)

# ---------------------------------------------------------------------------
# Toolbar: search + filter + export
# ---------------------------------------------------------------------------

try:
    all_records = storage.list_records()
    stats = storage.get_stats()
except Exception as exc:  # pragma: no cover - storage errors
    theme.alert(
        "error",
        "Record store unavailable",
        f"<code>{html.escape(str(exc))}</code> — the SQLite database under "
        "<code>data/patient_records/</code> could not be read.",
    )
    st.stop()

toolbar = st.columns([2, 1, 1, 1])
with toolbar[0]:
    search = st.text_input(
        "🔍 Search", placeholder="Patient ID, name or record ID…",
        help="Case-insensitive partial match.",
    )
with toolbar[1]:
    pred_filter = st.selectbox("Result filter", list(storage.PREDICTION_FILTERS))
with toolbar[2]:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    refresh = st.button("↻ Refresh", width="stretch")
    if refresh:
        st.rerun()
with toolbar[3]:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    export_all = st.button("⬇ Export all (CSV)", width="stretch",
                           disabled=not all_records)

records = storage.list_records(search=search, prediction_filter=pred_filter)

if export_all:
    buffer = io.StringIO()
    fields = [
        "record_id", "patient_id", "patient_name", "age", "gender",
        "predicted_class", "confidence", "prob_normal", "prob_tb",
        "prob_pneumonia", "prob_non_tb", "strategy", "created_at", "notes",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in all_records:
        writer.writerow(record)
    st.download_button(
        "⬇ Save CSV export",
        data=buffer.getvalue(),
        file_name="tbsentinel_assessments.csv",
        mime="text/csv",
    )

summary_line = (
    f"**{stats['total']}** records · **{stats['normal']}** normal · "
    f"**{stats['tb']}** TB · **{stats['pneumonia']}** pneumonia"
)
if stats.get("non_tb"):
    summary_line += f" · **{stats['non_tb']}** legacy binary non-TB"
summary_line += f" · **{stats['patients']}** patients"
st.caption(summary_line)

# ---------------------------------------------------------------------------
# Records table
# ---------------------------------------------------------------------------

if not records:
    theme.render_html(
        """
        <div style="border:1.5px dashed #CBD9E4;border-radius:14px;
                    padding:36px 16px;text-align:center;background:#fff;">
            <div style="font-size:30px;">🗂️</div>
            <div style="font-weight:700;color:#0F172A;margin-top:8px;font-size:15px;">
                No matching records</div>
            <div style="color:#5B6B7F;font-size:13px;">
                Save an assessment from the New Assessment page to build the
                patient record store, or clear the search filter.</div>
        </div>
        """
    )
    st.stop()


def _stored_interpretation(record: dict) -> dict | None:
    """Parse the clinical summary stored with a record, if any."""
    raw = record.get("interpretation")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# Map visible row index -> record_id for the detail viewer.
options = {
    f"{r['record_id']} · {r['patient_id']} · {r['patient_name'] or 'Unnamed'}"
    f" · {theme.class_label(r['predicted_class'])}"
    f" · {r['created_at']}": r["record_id"]
    for r in records
}

selected_label = st.selectbox(
    f"Open a record to view details ({len(records)} shown)",
    list(options.keys()),
    label_visibility="collapsed",
)
selected_id = options[selected_label]
record = storage.get_record(selected_id)

if record is None:  # pragma: no cover - race with deletion
    theme.alert("warning", "Record not found",
                "The selected record no longer exists.")
    st.stop()

# ---------------------------------------------------------------------------
# Record detail
# ---------------------------------------------------------------------------

class_kind = theme.class_kind(record["predicted_class"])

with st.container(border=True, key="card_detail"):
    head = st.columns([3, 1])
    with head[0]:
        theme.render_html(
            f"""
            <p class="ts-eyebrow">Record {html.escape(record['record_id'])}</p>
            <p class="ts-card-title" style="font-size:18px;">
                🩺 Assessment detail</p>
            {theme.badge(record['predicted_class'])}
            """
        )
    with head[1]:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Delete record", width="stretch"):
            st.session_state["confirm_delete"] = selected_id

    # Deletion confirmation (two-step to avoid accidents)
    if st.session_state.get("confirm_delete") == selected_id:
        theme.alert(
            "warning",
            "Delete this assessment?",
            "This permanently removes the record, its radiograph and Grad-CAM "
            "maps from the local store. This cannot be undone.",
        )
        confirm_cols = st.columns([1, 1])
        with confirm_cols[0]:
            if st.button("Yes, delete permanently", type="primary",
                         width="stretch"):
                try:
                    storage.delete_record(selected_id)
                except Exception as exc:  # pragma: no cover
                    theme.alert("error", "Delete failed",
                                html.escape(str(exc)))
                else:
                    st.session_state.pop("confirm_delete", None)
                    st.toast(f"{selected_id} deleted", icon="🗑️")
                    st.rerun()
        with confirm_cols[1]:
            if st.button("Cancel", width="stretch"):
                st.session_state.pop("confirm_delete", None)
                st.rerun()

    detail_left, detail_right = st.columns([1, 1], gap="large")

    # -- Left: images + Grad-CAM -------------------------------------------
    with detail_left:
        tabs = st.tabs(["Original", "Grad-CAM overlay", "Heatmap"])
        with tabs[0]:
            if record.get("image_jpeg"):
                st.image(Image.open(io.BytesIO(record["image_jpeg"])),
                         width="stretch")
            else:
                st.caption("No image stored for this record.")
        with tabs[1]:
            if record.get("overlay_png"):
                st.image(Image.open(io.BytesIO(record["overlay_png"])),
                         width="stretch")
            else:
                st.caption("No Grad-CAM overlay stored for this record.")
        with tabs[2]:
            if record.get("heatmap_png"):
                st.image(Image.open(io.BytesIO(record["heatmap_png"])),
                         width="stretch")
            else:
                st.caption("No heatmap stored for this record.")

    # -- Right: patient + result details ------------------------------------
    with detail_right:
        confidence = float(record["confidence"])
        theme.render_html(
            f"""
            <div class="ts-result ts-result--{class_kind}">
                <div class="ts-conf-value">{confidence:.1%}</div>
                <div class="ts-conf-label">Model confidence</div>
                <div class="ts-bar">
                    <div class="ts-bar-fill ts-bar-fill--{class_kind}"
                         style="width:{confidence * 100:.1f}%"></div>
                </div>
            </div>
            """
        )
        theme.render_html(
            f"""
            <div style="margin-top:14px;">
                {theme.kv_row("Patient ID", record['patient_id'])}
                {theme.kv_row("Name", record['patient_name'] or '—')}
                {theme.kv_row("Age", str(record['age']) if record['age'] else '—')}
                {theme.kv_row("Gender", record['gender'] or '—')}
                {theme.kv_row("Source file", record['filename'] or '—')}
                {theme.kv_row("Serving model", record['strategy'] or '—')}
                {theme.kv_row("Latency", f"{float(record['latency_seconds'] or 0):.2f}s")}
                {theme.kv_row("Assessed at", record['created_at'])}
            </div>
            """
        )
        # Only the classes the serving model actually returned are drawn; a
        # legacy two-class record has no normal/pneumonia probability stored.
        probability_rows = "".join(
            theme.prob_row(
                theme.class_label(class_name),
                float(record[column] or 0.0),
                theme.class_kind(class_name),
            )
            for class_name, column in (
                ("normal", "prob_normal"),
                ("tb", "prob_tb"),
                ("pneumonia", "prob_pneumonia"),
                ("non_tb", "prob_non_tb"),
            )
            if record.get(column) is not None
        )
        theme.render_html(
            f"""
            <div style="margin-top:14px;">
                <p class="ts-eyebrow">Class probabilities</p>
                {probability_rows}
            </div>
            """
        )

    # The explanation stored with the record, rendered exactly as it was when the
    # assessment was made (confidence band, focus, differential, disclaimer).
    stored_interpretation = _stored_interpretation(record)
    if stored_interpretation:
        theme.render_html('<hr class="ts-divider"/>')
        theme.render_html(
            '<p class="ts-eyebrow">Stored clinical explanation</p>'
            '<p class="ts-card-sub">Reproduced from the record — the same text '
            "that was shown when the assessment was saved.</p>"
        )
        components.render_interpretation_block(
            stored_interpretation,
            str(record["predicted_class"]),
            float(record["confidence"]),
            {
                name: float(record[column])
                for name, column in (
                    ("normal", "prob_normal"),
                    ("tb", "prob_tb"),
                    ("pneumonia", "prob_pneumonia"),
                    ("non_tb", "prob_non_tb"),
                )
                if record.get(column) is not None
            },
        )
    else:
        theme.render_html('<hr class="ts-divider"/>')
        theme.alert(
            "info",
            "No stored explanation for this record",
            "This assessment was saved before structured explanations were "
            "stored, so only the class probabilities above are available. "
            "Re-run the assessment to capture the full explanation.",
        )

    if record.get("notes"):
        theme.render_html('<hr class="ts-divider"/>')
        theme.render_html(
            f"""
            <div style="background:#FAFCFE;border:1px solid var(--ts-border);
                        border-radius:12px;padding:14px 16px;">
                <div style="font-size:12px;font-weight:700;letter-spacing:.8px;
                            text-transform:uppercase;color:var(--ts-muted);
                            margin-bottom:6px;">Clinical notes</div>
                <div style="font-size:13.5px;line-height:1.7;color:#334155;">
                    {html.escape(record['notes'])}</div>
            </div>
            """
        )

    if record.get("explain_error"):
        theme.alert("warning", "Grad-CAM was unavailable",
                    html.escape(str(record["explain_error"])))

    # -- Exports ------------------------------------------------------------
    theme.render_html('<hr class="ts-divider"/>')
    export_cols = st.columns(3)
    report = {
        "project": "TBSentinel",
        "record_id": record["record_id"],
        "generated_at": record["created_at"],
        "patient": {
            "patient_id": record["patient_id"],
            "name": record["patient_name"],
            "age": record["age"],
            "gender": record["gender"],
            "notes": record["notes"],
        },
        "file": record["filename"],
        "model": record["strategy"],
        "predicted_class": record["predicted_class"],
        "predicted_class_label": theme.class_label(record["predicted_class"]),
        "confidence": record["confidence"],
        "probabilities": {
            name: record[column]
            for name, column in (
                ("normal", "prob_normal"),
                ("tb", "prob_tb"),
                ("pneumonia", "prob_pneumonia"),
                ("non_tb", "prob_non_tb"),
            )
            if record.get(column) is not None
        },
        "interpretation": _stored_interpretation(record),
        "latency_seconds": record["latency_seconds"],
        "disclaimer": "Research prototype. Not for clinical diagnosis.",
    }
    with export_cols[0]:
        st.download_button(
            "⬇ Summary (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"{record['record_id']}_summary.json",
            mime="application/json",
            width="stretch",
        )
    with export_cols[1]:
        if record.get("overlay_png"):
            st.download_button(
                "⬇ Grad-CAM overlay",
                data=record["overlay_png"],
                file_name=f"{record['record_id']}_gradcam_overlay.png",
                mime="image/png",
                width="stretch",
            )
    with export_cols[2]:
        if record.get("image_jpeg"):
            st.download_button(
                "⬇ Radiograph",
                data=record["image_jpeg"],
                file_name=f"{record['record_id']}_radiograph.jpg",
                mime="image/jpeg",
                width="stretch",
            )

theme.render_html(
    '<p class="ts-card-sub" style="text-align:center;margin-top:14px;">'
    "Records are stored locally in <code>data/patient_records/assessments.db</code> "
    "and are not shared with any external service.</p>"
)
