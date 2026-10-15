"""Dashboard — clinical overview, system status and recent assessments."""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from tbui import api, components, storage, theme

# Page paths for st.switch_page — resolved relative to the entrypoint
# (frontend/app.py), hence the "views/" prefix.
PAGE_FILES = {
    "New Assessment": "views/assessment.py",
    "Patient History": "views/history.py",
}

api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
health = components.cached_health(api_url)
summary = api.health_summary(health)

components.render_topbar(
    "Dashboard",
    "Clinical overview, system status and recent assessments",
    health,
)

# ---------------------------------------------------------------------------
# Welcome hero
# ---------------------------------------------------------------------------

theme.render_html(
    f"""
    <div class="ts-hero">
        <div class="ts-hero-left">
            <div class="ts-logo">{theme.LOGO_SVG}</div>
            <div>
                <p class="ts-brand">TBSentinel</p>
                <p class="ts-tagline">AI-powered chest X-ray screening with explainable results</p>
            </div>
        </div>
        <div class="ts-hero-right">
            <span class="ts-chip">EfficientNet-B3</span>
            <span class="ts-chip">Grad-CAM Explainable</span>
        </div>
    </div>
    """
)

st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# System status strip
# ---------------------------------------------------------------------------

if not summary["api_online"]:
    theme.alert(
        "error",
        "Inference backend offline",
        "The dashboard cannot reach the FastAPI service. Start it with "
        "<code>python -m src.inference.main</code>, then press "
        "<b>Re-check</b> in the sidebar. Assessment pages will stay disabled "
        "until the model is available.",
    )
elif not summary["model_loaded"]:
    theme.alert(
        "warning",
        "API reachable but model not loaded",
        "The service is up but no checkpoint is loaded — check the checkpoint "
        "path in <code>configs/inference/inference.yaml</code>.",
    )
else:
    theme.alert(
        "success",
        "All systems operational",
        f"The inference service is online and serving the "
        f"<b>{html.escape(summary['strategy'])}</b> checkpoint on "
        f"<b>{html.escape(summary['device'])}</b>. Ready for new assessments.",
    )

status_a, status_b, status_c, status_d = st.columns(4)
with status_a:
    theme.render_html(
        theme.stat_card(
            "Inference API",
            "Online" if summary["api_online"] else "Offline",
            "🖧",
            foot=api_url,
            tone="" if summary["api_online"] else "danger",
        )
    )
with status_b:
    theme.render_html(
        theme.stat_card(
            "Model",
            "Loaded" if summary["model_loaded"] else "Unavailable",
            "🧠",
            foot=summary["strategy"],
            tone="" if summary["model_loaded"] else "danger",
        )
    )
with status_c:
    theme.render_html(
        theme.stat_card("Device", summary["device"], "💻",
                        foot="Compute device reported by the API")
    )
with status_d:
    theme.render_html(
        theme.stat_card("Checkpoint", summary["checkpoint_name"], "📦",
                        foot="Best validation checkpoint on disk")
    )

st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick statistics (from the local record store)
# ---------------------------------------------------------------------------

try:
    stats = storage.get_stats()
except Exception:  # pragma: no cover - storage is best-effort on the dashboard
    stats = {"total": 0, "normal": 0, "tb": 0, "pneumonia": 0, "non_tb": 0,
             "patients": 0, "avg_confidence": 0.0, "tb_rate": 0.0,
             "normal_rate": 0.0, "pneumonia_rate": 0.0, "last_at": None}

stats_label, stats_action = st.columns([3, 1])
with stats_label:
    theme.render_html(
        '<p class="ts-eyebrow">Activity</p>'
        '<p class="ts-card-title" style="font-size:18px;">📊 Assessment statistics</p>'
        '<p class="ts-card-sub">Aggregated over every assessment saved to the '
        'local patient record store.</p>'
    )
with stats_action:
    if st.button("＋ Start new assessment", type="primary", width="stretch"):
        st.switch_page(PAGE_FILES["New Assessment"])

k1, k2, k3, k4 = st.columns(4)
with k1:
    theme.render_html(
        theme.stat_card("Total assessments", str(stats["total"]), "🗂️",
                        foot=f"{stats['patients']} distinct patients")
    )
with k2:
    theme.render_html(
        theme.stat_card("Normal", str(stats["normal"]), "✅",
                        foot=f"{stats['normal_rate']:.1%} of assessments",
                        tone="success")
    )
with k3:
    theme.render_html(
        theme.stat_card("TB detected", str(stats["tb"]), "⚠️",
                        foot=f"{stats['tb_rate']:.1%} positivity rate",
                        tone="danger")
    )
with k4:
    theme.render_html(
        theme.stat_card("Pneumonia", str(stats["pneumonia"]), "🫁",
                        foot=f"{stats['pneumonia_rate']:.1%} of assessments",
                        tone="warn")
    )

st.caption(
    f"Mean model confidence **{stats['avg_confidence']:.1%}** · "
    f"last assessment **{stats['last_at'] or '—'}**"
    + (
        f" · {stats['non_tb']} legacy binary non-TB record(s)"
        if stats.get("non_tb")
        else ""
    )
)

st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Recent assessments preview
# ---------------------------------------------------------------------------

try:
    recent = storage.recent_records(limit=5)
except Exception:  # pragma: no cover
    recent = []

with st.container(border=True, key="card_recent"):
    head_a, head_b = st.columns([3, 1])
    with head_a:
        theme.render_html(
            '<p class="ts-eyebrow">Activity</p>'
            '<p class="ts-card-title">🕘 Recent assessments</p>'
            '<p class="ts-card-sub">The five most recent saved assessments. '
            'Open Patient History to search, review or export any record.</p>'
        )
    with head_b:
        if recent and st.button("View all records", width="stretch"):
            st.switch_page(PAGE_FILES["Patient History"])

    if not recent:
        theme.render_html(
            """
            <div style="border:1.5px dashed #CBD9E4;border-radius:12px;
                        padding:30px 16px;text-align:center;background:#FAFCFE;">
                <div style="font-size:26px;">📭</div>
                <div style="font-weight:600;color:#0F172A;margin-top:6px;">
                    No assessments saved yet</div>
                <div style="color:#5B6B7F;font-size:13px;">
                    Run your first assessment to see it appear here.</div>
            </div>
            """
        )
    else:
        rows = "".join(
            f"""
            <tr>
                <td><b>{html.escape(r['record_id'])}</b></td>
                <td>{html.escape(r['patient_id'])}</td>
                <td>{html.escape(r['patient_name'] or '—')}</td>
                <td>{theme.badge(r['predicted_class'], compact=True)}</td>
                <td class="ts-num">{float(r['confidence']):.1%}</td>
                <td>{html.escape(r['created_at'])}</td>
            </tr>
            """
            for r in recent
        )
        theme.render_html(
            f"""
            <table class="ts-table">
                <thead>
                    <tr>
                        <th>Record</th><th>Patient ID</th><th>Name</th>
                        <th>Result</th><th class="ts-num">Confidence</th>
                        <th>Assessed at</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """
        )

st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How to use
# ---------------------------------------------------------------------------

with st.expander("ℹ️ How to use this system", expanded=False):
    st.markdown(
        """
**TBSentinel** screens frontal chest radiographs for tuberculosis using an
EfficientNet-B3 classifier trained (centrally and federated) on the TBX11K
benchmark, and explains every decision with Grad-CAM.

1. **Check system status** — the sidebar must show *Model loaded and ready*.
2. **New Assessment** — enter the patient's details, upload a chest X-ray and
   press *Run Assessment*. Prediction and the Grad-CAM attention map are
   generated in a single step.
3. **Review the result** — red indicates a TB-positive screen, green a
   TB-negative one. Always read the Grad-CAM overlay together with the
   prediction: warm regions are where the model looked.
4. **Save the assessment** — records are stored locally (SQLite) and can be
   searched, reopened and exported from **Patient History**.
5. **Check Performance Metrics** before trusting outputs — it shows how the
   model behaves on the held-out validation split, including sensitivity.

> **Reminder:** this is a research prototype, not a certified medical device.
> Every output must be confirmed by a qualified radiologist.
    """
    )
