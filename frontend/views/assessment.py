"""New Assessment — patient details, upload, prediction, Grad-CAM and save."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from tbui import api, components, storage, theme

PATIENT_KEYS = {
    "name": "pf_name",
    "age": "pf_age",
    "gender": "pf_gender",
    "patient_id": "pf_patient_id",
    "notes": "pf_notes",
}

api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
health = components.cached_health(api_url)

components.render_topbar(
    "New Assessment",
    "",
    health,
)

# ---------------------------------------------------------------------------
# Patient information form
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_patient"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Patient</p>
        <p class="ts-card-title">Patient information</p>
        <p class="ts-card-sub">Optional for the model (image-only inference), but
        stored with the record so the assessment joins the patient's history.</p>
        """
    )
    row1 = st.columns([3, 1, 1])
    with row1[0]:
        st.text_input("Patient name", key=PATIENT_KEYS["name"],
                      placeholder="Name")
    with row1[1]:
        st.number_input("Age", min_value=0, max_value=120, value=0, step=1,
                        key=PATIENT_KEYS["age"],
                        help="0 = not provided")
    with row1[2]:
        st.selectbox("Gender", ["Female", "Male", "Other"],
                     key=PATIENT_KEYS["gender"])

    row2 = st.columns([2, 1, 3])
    # The Generate button must be instantiated *before* the Patient ID text
    # input: newer Streamlit forbids writing st.session_state[key] after the
    # widget that owns that key has been created in the same run. Column order
    # in the code does not affect the visual layout.
    with row2[1]:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("🪄 Generate",
                     help="Create a unique patient ID such as PT-20261011-0003."):
            st.session_state[PATIENT_KEYS["patient_id"]] = (
                storage.generate_patient_id()
            )
            st.rerun()
    with row2[0]:
        st.text_input(
            "Patient ID", key=PATIENT_KEYS["patient_id"],
            placeholder="PT-0001",
            help="Used to group this assessment in the patient's history.",
        )
    with row2[2]:
        st.text_area("Clinical notes (optional)", key=PATIENT_KEYS["notes"],
                     placeholder="Symptoms, exposure history, prior imaging…",
                     height=100)

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload + run
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1], gap="large")
with left:
    analyze_clicked = components.render_upload_card()
with right:
    if analyze_clicked:
        components.execute_analysis(api_url, health)
    components.render_result_card(st.session_state.get("analysis"))

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

# Full width: the three Grad-CAM panels (original / heatmap / overlay) need the
# room to be read side by side, and the interpretation below is long-form text.
components.render_explainability_card(st.session_state.get("analysis"))

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

components.render_clinical_card(st.session_state.get("analysis"))

# ---------------------------------------------------------------------------
# Save to patient history + export
# ---------------------------------------------------------------------------

analysis = st.session_state.get("analysis")

with st.container(border=True, key="card_save"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Archive</p>
        <p class="ts-card-title">💾 Save assessment</p>
        <p class="ts-card-sub">Stores the patient details, prediction and Grad-CAM
        maps in the local record store (SQLite) so the case can be reviewed,
        searched and exported later.</p>
        """
    )

    if analysis is None:
        theme.render_html(
            '<p class="ts-card-sub">Run an assessment first — the save controls '
            "will appear here once a result is available.</p>"
        )
    else:
        patient = {
            "patient_id": st.session_state[PATIENT_KEYS["patient_id"]].strip(),
            "patient_name": st.session_state[PATIENT_KEYS["name"]].strip(),
            "age": st.session_state[PATIENT_KEYS["age"]] or None,
            "gender": st.session_state[PATIENT_KEYS["gender"]],
            "notes": st.session_state[PATIENT_KEYS["notes"]].strip(),
        }
        if patient["gender"] == "—":
            patient["gender"] = None

        already_saved = bool(st.session_state.get("analysis_saved"))
        saved_record_id = st.session_state.get("analysis_record_id")

        if not patient["patient_id"]:
            theme.alert(
                "info",
                "Patient ID required to save",
                "Enter a patient ID (or press <b>🪄 Auto-generate</b>) so this "
                "assessment can be linked to a patient record.",
            )

        save_col, export_col = st.columns([1, 1])
        with save_col:
            if already_saved:
                st.button(
                    f"✅ Saved as {saved_record_id}",
                    disabled=True, width="stretch",
                )
            else:
                if st.button(
                    "💾 Save assessment to patient history",
                    type="primary", width="stretch",
                    disabled=not patient["patient_id"],
                ):
                    try:
                        record_id = storage.insert_assessment(
                            {**analysis, **patient}
                        )
                    except Exception as exc:  # pragma: no cover - storage errors
                        theme.alert(
                            "error",
                            "Could not save the assessment",
                            f"<code>{html.escape(str(exc))}</code> — the record "
                            "store may be locked or read-only. The assessment is "
                            "still available in this session and can be exported.",
                        )
                    else:
                        st.session_state["analysis_saved"] = True
                        st.session_state["analysis_record_id"] = record_id
                        st.toast(f"Assessment {record_id} saved", icon="💾")
                        st.rerun()

        with export_col:
            report = components.build_summary_report(analysis, patient,
                                                     saved_record_id)
            st.download_button(
                "⬇ Download summary (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"{saved_record_id or 'assessment'}_summary.json",
                mime="application/json",
                width="stretch",
            )

        if already_saved:
            theme.alert(
                "success",
                f"Assessment {saved_record_id} stored",
                "Open <b>Patient History</b> to search, reopen or export this "
                "record at any time.",
            )
