"""Reusable Streamlit widgets for the TBSentinel clinical dashboard.

Every page composes the same building blocks: topbar, sidebar, KPI cards,
diagnostic result card, Grad-CAM explainability card and the clinical
interpretation card. Keeping them here guarantees a consistent look and means
prediction + explanation behaviour only ever has to be written once.
"""

from __future__ import annotations

import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from tbui import api, theme

# ---------------------------------------------------------------------------
# Chrome: topbar, sidebar, footer
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def cached_health(api_url: str) -> dict[str, Any] | None:
    """Cached ``/health`` probe (10 s TTL) shared by every page."""
    return api.check_api_health(api_url)


def invalidate_health_cache() -> None:
    """Drop the cached health payload (used by the Re-check button)."""
    cached_health.clear()


def render_topbar(page_title: str, page_sub: str, health: dict[str, Any] | None) -> None:
    """Slim branded header bar rendered at the top of every page."""
    summary = api.health_summary(health)
    if not summary["api_online"]:
        status = theme.pill("down", "API offline")
    elif not summary["model_loaded"]:
        status = theme.pill("down", "Model not loaded")
    else:
        status = theme.pill("ok", "System ready")
    theme.render_html(
        f"""
        <div class="ts-topbar">
            <div>
                <p class="ts-topbar-title">{html.escape(page_title)}</p>
                <p class="ts-topbar-sub">{html.escape(page_sub)}</p>
            </div>
            <div class="ts-topbar-right">
                <span class="ts-chip" style="background:var(--ts-primary-soft);
                      color:var(--ts-primary-dark);border-color:#B9DEE9;">
                    Chest X-ray AI
                </span>
                <span class="ts-chip" style="background:var(--ts-primary-soft);
                      color:var(--ts-primary-dark);border-color:#B9DEE9;">
                    Grad-CAM Explainable
                </span>
                {status}
            </div>
        </div>
        """
    )


def render_sidebar(health: dict[str, Any] | None) -> str:
    """Render the sidebar status/connection panel; returns the API URL.

    The brand block and navigation are rendered by the app frame (``app.py``)
    above this panel, so only system status lives here.
    """
    summary = api.health_summary(health)
    with st.sidebar:
        st.markdown('<p class="ts-nav-label">System status</p>',
                    unsafe_allow_html=True)
        if summary["api_online"] and summary["model_loaded"]:
            theme.render_html(
                f"""
                {theme.pill("ok", "Model loaded and ready")}
                <div style="margin-top:12px;">
                    {theme.kv_row("Model", summary["strategy"])}
                    {theme.kv_row("Device", summary["device"])}
                    {theme.kv_row("Checkpoint", summary["checkpoint_name"])}
                    {theme.kv_row("Classes", summary["class_list"])}
                </div>
                """
            )
        elif summary["api_online"]:
            theme.render_html(
                f'{theme.pill("down", "API reachable, model not loaded")}'
            )
        else:
            theme.render_html(f'{theme.pill("down", "Inference API offline")}')

        st.markdown("---")
        st.markdown('<p class="ts-nav-label">Connection</p>',
                    unsafe_allow_html=True)
        api_url = st.text_input(
            "Inference API URL",
            value=st.session_state.get("api_url", api.DEFAULT_API_URL),
            help="Base URL of the FastAPI inference service.",
        )
        st.session_state["api_url"] = api_url

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Re-check", width="stretch"):
                invalidate_health_cache()
                st.rerun()
        with col_b:
            st.caption("Auto-refreshes every 10 s.")

        if not summary["api_online"]:
            theme.alert(
                "warning",
                "Backend not detected",
                "Start the inference service first:<br/>"
                "<code>python -m src.inference.main</code>",
            )

        st.markdown("---")
        with st.expander("How to use this system", expanded=False):
            st.markdown(
                """
1. Confirm the sidebar shows **Model loaded and ready**.
2. Open **New Assessment**, enter patient details and upload a
   frontal chest X-ray (JPG / PNG / BMP).
3. Press **Run Assessment** — prediction and Grad-CAM run in parallel.
4. Read the colour-coded result, then the attention map: warm
   areas are where the model looked.
5. **Save** the assessment to keep it in the patient record, and
   export the summary for your notes.

The result badge is colour-coded: **green** = normal, **red** =
tuberculosis, **amber** = pneumonia / lung opacity. Every result is a
research output — **not a diagnosis**.
                """
            )

        st.markdown("---")
        theme.alert(
            "warning",
            "Medical disclaimer",
            "Research prototype for demonstration only. Not a certified medical "
            "device and not validated for clinical use. Always confirm findings "
            "with a qualified radiologist.",
        )
        theme.render_html(
            """
            <div style="text-align:center;color:#94A3B8;font-size:11.5px;
                        padding-top:4px;">TBSentinel v1.0.0</div>
            """
        )
    return api_url


def render_footer() -> None:
    """Page footer with the standing research disclaimer."""
    theme.render_html(
        """
        <div class="ts-footer">
            <strong>Research Prototype – Not for Clinical Diagnosis</strong><br/>
            TBSentinel v1.0.0 · Federated Multimodal CDSS · EfficientNet-B3 + Grad-CAM ·
            Built with Streamlit, FastAPI and Flower
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Assessment input
# ---------------------------------------------------------------------------


def render_upload_card() -> bool:
    """Render the upload + preview card; return True when Analyze is pressed."""
    clicked = False
    with st.container(border=True, key="card_upload"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Step 1 · Input</p>
            <p class="ts-card-title">🩻 Chest X-ray upload</p>
            <p class="ts-card-sub">Accepted formats: JPG, JPEG, PNG, BMP. Use a
            frontal chest radiograph for best results.</p>
            """
        )
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image…",
            type=api.SUPPORTED_TYPES,
            help="The image is sent to the local FastAPI inference service.",
            key="uploader_input"
        )

        if uploaded_file is None:
            st.session_state["pending_input"] = None
            st.session_state["analysis"] = None
            theme.render_html(
                """
                <div style="border:2px dashed #CBD5E1;border-radius:16px;
                            padding:48px 24px;text-align:center;background:#F8FAFC;
                            margin-top:20px;transition:all .2s ease;">
                    <div style="font-size:56px;margin-bottom:16px;">🫁</div>
                    <div style="font-weight:700;color:#1E293B;margin-top:12px;font-size:16px;">
                        No image selected</div>
                    <div style="color:#64748B;font-size:14px;margin-top:6px;line-height:1.5;">
                        Drag a chest X-ray into the box above to begin.</div>
                </div>
                """
            )
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.button("Run assessment", type="primary", width="stretch",
                      disabled=True, key="btn_upload_disabled")
            return False

        uploaded_file.seek(0)
        image_bytes = uploaded_file.getvalue()
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            theme.alert(
                "error",
                "Unreadable image",
                "The uploaded file could not be decoded. Re-export it as PNG or "
                "JPEG and try again.",
            )
            return False

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.image(image, caption="Original radiograph", width="stretch")
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        theme.kv_list(
            [
                ("File", uploaded_file.name),
                ("Resolution", f"{image.width} x {image.height} px"),
                ("Mode", image.mode),
            ]
        )
        signature = f"{uploaded_file.name}:{len(image_bytes)}"
        stored = st.session_state.get("analysis")
        if stored and stored.get("signature") != signature:
            st.session_state["analysis"] = None
        st.session_state["pending_input"] = {
            "bytes": image_bytes,
            "name": uploaded_file.name,
            "type": uploaded_file.type or "image/png",
            "signature": signature,
        }
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        clicked = st.button("Run assessment", type="primary", width="stretch", key="btn_upload")
    return clicked


def execute_analysis(api_url: str, health: dict[str, Any] | None) -> None:
    """Validate the pending upload and run inference, reporting inside a status."""
    pending = st.session_state.get("pending_input")
    if not pending:
        theme.alert("warning", "No image", "Please upload a chest X-ray first.")
        return

    try:
        Image.open(io.BytesIO(pending["bytes"])).verify()
    except Exception:
        theme.alert(
            "error",
            "Unreadable image",
            "The uploaded file could not be decoded. Re-export it as PNG or JPEG "
            "and try again.",
        )
        return

    summary = api.health_summary(health)
    if not (summary["api_online"] and summary["model_loaded"]):
        theme.alert(
            "error",
            "Inference service unavailable",
            "The backend must be running before analysis. Start it with "
            "<code>python -m src.inference.main</code> and re-check the "
            "connection in the sidebar.",
        )
        return

    started = datetime.now(timezone.utc)
    failure: str | None = None
    prediction: dict[str, Any] = {}
    explanation: dict[str, Any] | None = None
    explain_error: str | None = None

    with st.status("Running assessment…", expanded=True) as status:
        st.write("Sending image to the inference service…")
        try:
            prediction, explanation, explain_error = api.run_analysis(
                api_url,
                pending["bytes"],
                pending["name"],
                pending["type"],
            )
        except api.ApiError as exc:
            failure = str(exc)
            status.update(label="Assessment failed", state="error")
        else:
            st.write(f"Scoring probabilities across {summary['class_list']}…")
            if explanation is not None:
                st.write("Generating Grad-CAM attention map…")
                status.update(label="Assessment complete", state="complete")
            else:
                status.update(
                    label="Prediction ready (explanation unavailable)",
                    state="complete",
                )

    if failure is not None:
        theme.alert("error", "Prediction failed", html.escape(failure))
        return

    st.session_state["analysis"] = {
        "prediction": prediction,
        "explanation": explanation,
        # Snapshot of the explanation for the *predicted* class: recomputing the
        # map for another class later must not rewrite the prediction's report.
        "prediction_explanation": explanation,
        "image_bytes": pending["bytes"],
        "filename": pending["name"],
        "mime_type": pending.get("type") or "image/png",
        "signature": pending.get("signature"),
        "strategy": summary["strategy"],
        "timestamp": started.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "latency_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    if explain_error is not None:
        st.session_state["analysis"]["explain_error"] = explain_error
    # A fresh result has not been saved yet.
    st.session_state["analysis_saved"] = False
    st.rerun()


# ---------------------------------------------------------------------------
# Result / explainability / clinical cards
# ---------------------------------------------------------------------------


def explain_option_label(class_name: str) -> str:
    """Display label for one entry of the Grad-CAM class selector."""
    if class_name == "__predicted__":
        return "Predicted class"
    return theme.class_label(class_name)


def run_explain_for_class(
    analysis: dict[str, Any],
    target_class: str | None,
) -> None:
    """Recompute the Grad-CAM map for one class and update the live analysis.

    Only the explanation is replaced: the prediction, the patient and the stored
    record keep the original call. Any failure is reported and the previous map is
    left in place rather than blanking the card.
    """
    api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
    label = (
        "the predicted class"
        if target_class is None
        else theme.class_label(target_class)
    )
    with st.spinner(f"Recomputing Grad-CAM for {label}…"):
        try:
            explanation = api.run_explain(
                api_url,
                analysis["image_bytes"],
                str(analysis.get("filename") or "xray.png"),
                str(analysis.get("mime_type") or "image/png"),
                target_class=target_class,
            )
        except api.ApiError as exc:
            theme.alert(
                "error",
                "Could not recompute the Grad-CAM map",
                html.escape(str(exc)),
            )
            return

    st.session_state["analysis"]["explanation"] = explanation
    st.session_state["analysis"].pop("explain_error", None)
    st.rerun()


def render_interpretation_block(
    interpretation: dict[str, Any] | None,
    class_name: str,
    confidence: float,
    probabilities: dict[str, float] | None = None,
) -> None:
    """Render a structured clinical explanation consistently.

    Used by the New Assessment result and by the Patient History detail view, so a
    stored record reads exactly like a fresh one. Records written before the
    structured fields existed only carry ``headline``/``notes``; every section
    below tolerates a missing key instead of failing.
    """
    interpretation = interpretation or {}
    probabilities = probabilities or {}
    kind = theme.class_kind(class_name)

    headline = str(interpretation.get("headline") or "").strip()
    notes = [str(note) for note in (interpretation.get("notes") or [])]
    tone = str(interpretation.get("tone") or theme.CLASS_TONES.get(kind, "info"))
    if tone not in {"error", "warning", "info", "success"}:
        tone = "info"
    if not headline:
        distribution = ", ".join(
            f"{theme.class_label(name)} {float(probabilities.get(name, 0.0)):.1%}"
            for name in theme.ordered_classes(list(probabilities))
        )
        headline = (
            f"Model screen: <b>{theme.class_label(class_name)}</b> with "
            f"{confidence:.1%} confidence ({distribution})."
        )
    # SIMPLIFIED: Shorter, punchier notes
    if not notes:
        notes = [
            "✓ Verify Grad-CAM highlights match actual radiographic findings",
            "✓ Correlate with patient symptoms and history",
            "⚠ This is a research prototype, not a medical diagnosis",
        ]

    theme.alert(tone, "Model impression", headline)

    # SIMPLIFIED: Hide detailed confidence band in expander
    with st.expander("Confidence details", expanded=False):
        band = interpretation.get("confidence") or {}
        band_rows = [
            '<div style="font-size:12px;font-weight:700;letter-spacing:.8px;'
            'text-transform:uppercase;color:var(--ts-muted);margin-bottom:6px;">'
            'Confidence assessment</div>'
        ]
        level = str(band.get("level") or "")
        if level:
            band_rows.append(
                theme.level_chip(level, str(band.get("tone") or tone))
            )
        band_explanation = str(band.get("explanation") or "")
        if band_explanation:
            band_rows.append(
                '<div style="color:#334155;font-size:13px;line-height:1.6;'
                f'margin-top:6px;">{html.escape(band_explanation)}</div>'
            )
        band_rows.append(
            theme.kv_row("Probability of the reported class", f"{confidence:.1%}")
        )
        runner_label = band.get("runner_up_label")
        if runner_label:
            runner_probability = float(band.get("runner_up_probability") or 0.0)
            band_rows.append(
                theme.kv_row(
                    "Closest other class", f"{runner_label} at {runner_probability:.1%}"
                )
            )
        if band.get("margin") is not None:
            band_rows.append(
                theme.kv_row("Margin over that class", f"{float(band['margin']):.1%}")
            )
        theme.render_html(
            '<div style="background:#FAFCFE;border:1px solid var(--ts-border);'
            'border-radius:12px;padding:14px 16px;">'
            + "".join(band_rows)
            + "</div>"
        )

    # SIMPLIFIED: Hide focus details in expander
    with st.expander("What the model focused on", expanded=False):
        focus = interpretation.get("focus") or {}
        if focus.get("summary"):
            focus_rows = [
                '<div style="color:#334155;font-size:13.5px;line-height:1.65;">'
                f"{html.escape(str(focus['summary']))}</div>",
            ]
            if focus.get("available") and focus.get("coverage") is not None:
                focus_rows.append(
                    theme.kv_row(
                        "Activated area", f"{float(focus['coverage']):.1%} of the image"
                    )
                )
                focus_rows.append(
                    theme.kv_row("Peak activation", f"{float(focus['peak']):.0%}")
                )
            if focus.get("available") and focus.get("convention"):
                focus_rows.append(
                    f'<div style="color:#5B6B7F;font-size:12.5px;margin-top:8px;">'
                    f"{html.escape(str(focus['convention']))}</div>"
                )
            if focus.get("available") and focus.get("caveat"):
                focus_rows.append(
                    f'<div style="color:#94A3B8;font-size:12px;margin-top:4px;">'
                    f"{html.escape(str(focus['caveat']))}</div>"
                )
            theme.render_html(
                '<div style="background:#fff;border:1px solid '
                'var(--ts-border);border-radius:12px;padding:14px 16px;">'
                + "".join(focus_rows)
                + "</div>"
            )

    # SIMPLIFIED: Show only key notes
    note_items = "".join(f"<li>{note}</li>" for note in notes)
    theme.render_html(
        f"""
        <div style="margin-top:14px;background:#FAFCFE;border:1px solid
                    var(--ts-border);border-radius:12px;padding:16px 18px;">
            <div style="font-size:12px;font-weight:700;letter-spacing:.8px;
                        text-transform:uppercase;color:var(--ts-muted);
                        margin-bottom:8px;">Key considerations</div>
            <ul style="padding-left:18px;margin:0;font-size:13.5px;
                       line-height:1.75;color:#334155;">{note_items}</ul>
        </div>
        """
    )

    # SIMPLIFIED: Hide differential in expander
    differential = interpretation.get("differential") or []
    if differential:
        with st.expander("Why other classes were not chosen", expanded=False):
            for row in differential:
                row_class = str(row.get("class_name") or "")
                theme.render_html(
                    theme.prob_row(
                        str(row.get("label") or theme.class_label(row_class)),
                        float(row.get("probability") or 0.0),
                        theme.class_kind(row_class),
                    )
                )
                if row.get("note"):
                    st.caption(str(row["note"]))

    # SIMPLIFIED: Hide generated flags in expander
    flags = interpretation.get("flags") or []
    if flags:
        with st.expander("Additional warnings", expanded=False):
            for flag in flags:
                theme.alert(
                    "warning", "Check before acting on this screen", html.escape(str(flag))
                )

    disclaimer = str(interpretation.get("disclaimer") or "").strip()
    if not disclaimer:
        disclaimer = "Research prototype. Not for clinical diagnosis."
    theme.render_html(
        f'<div style="margin-top:12px;color:#94A3B8;font-size:12px;'
        f'line-height:1.6;">{html.escape(disclaimer)}</div>'
    )


def render_result_card(analysis: dict[str, Any] | None) -> None:
    """Diagnostic result card (idle placeholder when no analysis exists)."""
    with st.container(border=True, key="card_result"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Step 2 · Result</p>
            <p class="ts-card-title">🧪 Diagnostic result</p>
            """
        )
        if analysis is None:
            theme.render_html(
                """
                <div style="text-align:center;padding:40px 20px 20px 20px;">
                    <div style="font-size:48px;margin-bottom:16px;">⏳</div>
                    <div style="font-weight:700;color:#1E293B;margin-top:12px;font-size:17px;">
                        Awaiting assessment</div>
                    <div style="color:#64748B;font-size:14px;margin-top:6px;line-height:1.5;">
                        Upload an image and press <b>Run assessment</b>.</div>
                </div>
                """
            )
            return

        prediction = analysis["prediction"]
        class_name = str(prediction.get("predicted_class_name", ""))
        confidence = float(prediction.get("confidence", 0.0))
        probabilities = prediction.get("probabilities", {}) or {}
        kind = theme.class_kind(class_name)
        symbol, badge_text = theme.class_badge(class_name)

        # SIMPLIFIED: Make prediction the focal point with large typography
        theme.render_html(
            f"""
            <div class="ts-result ts-result--{kind}" style="padding:32px 24px;">
                <span class="ts-badge ts-badge--{kind}" style="font-size:18px;padding:8px 16px;">
                    {symbol} {badge_text}</span>
                <div class="ts-conf-value" style="font-size:48px;font-weight:800;margin:16px 0 8px 0;">
                    {confidence:.1%}
                </div>
                <div class="ts-conf-label" style="font-size:16px;">Model confidence</div>
                <div class="ts-bar" style="height:12px;margin-top:16px;">
                    <div class="ts-bar-fill ts-bar-fill--{kind}"
                         style="width:{confidence * 100:.1f}%"></div>
                </div>
            </div>
            """
        )

        # SIMPLIFIED: Hide class probabilities in expander
        with st.expander("View all class probabilities", expanded=False):
            class_rows = "".join(
                theme.prob_row(
                    theme.class_label(name),
                    float(probabilities.get(name, 0.0)),
                    theme.class_kind(name),
                )
                for name in theme.ordered_classes(list(probabilities))
            )
            theme.render_html(
                f"""
                <div style="margin-top:12px;">
                    <p class="ts-eyebrow">Class probabilities</p>
                    {class_rows}
                </div>
                """
            )

        # SIMPLIFIED: Hide technical details in expander
        with st.expander("Technical details", expanded=False):
            band = (prediction.get("interpretation") or {}).get("confidence") or {}
            band_html = ""
            if band.get("level"):
                band_tone = str(band.get("tone") or theme.CLASS_TONES.get(kind, "info"))
                band_detail = ""
                if band.get("runner_up_label") and band.get("margin") is not None:
                    band_detail = (
                        '<span style="color:#5B6B7F;font-size:13px;margin-left:8px;">'
                        f"runner-up {html.escape(str(band['runner_up_label']))} at "
                        f"{float(band.get('runner_up_probability') or 0.0):.1%} "
                        f"(margin {float(band['margin']):.1%})</span>"
                    )
                band_html = (
                    '<div style="margin-top:16px;">'
                    f"{theme.level_chip(str(band['level']), band_tone)}"
                    f"{band_detail}</div>"
                )

            theme.render_html(
                f"""
                {band_html}
                <div style="margin-top:18px;">
                    {theme.kv_row("Serving model", str(analysis.get("strategy", "Unknown")))}
                    {theme.kv_row("Analysed at", str(analysis.get("timestamp", "—")))}
                    {theme.kv_row("Latency", f"{analysis.get('latency_seconds', 0.0):.2f}s")}
                </div>
                <div class="ts-card-sub" style="margin-top:12px;">
                    Confidence is the model's softmax probability for the winning
                    class; the band is <b>high</b> from 85%, <b>moderate</b> from
                    60%, and capped at moderate when the closest other class sits
                    within 10 points.
                </div>
                """
            )


def render_explainability_card(analysis: dict[str, Any] | None) -> None:
    """Grad-CAM explainability card with downloadable artefacts."""
    with st.container(border=True, key="card_explain"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Step 3 · Explainability</p>
            <p class="ts-card-title">🔥 Grad-CAM attention map</p>
            <p class="ts-card-sub">Warm (red/orange) regions contributed most
            strongly to the selected class. Original, heatmap and overlay are
            shown side by side.</p>
            """
        )
        data = analysis
        explanation = data.get("explanation") if data else None
        if data and data.get("explain_error"):
            theme.alert(
                "warning",
                "Grad-CAM unavailable",
                "The prediction is valid, but the explanation endpoint failed: "
                f"{html.escape(str(data['explain_error']))}",
            )
        if data is None or not explanation:
            theme.render_html(
                """
                <div style="border:1.5px dashed #CBD9E4;border-radius:12px;
                            padding:32px 20px;text-align:center;background:#FAFCFE;">
                    <div style="font-size:32px;">🔥</div>
                    <div style="font-weight:600;color:#0F172A;margin-top:10px;font-size:16px;">
                        Explanation will appear here</div>
                    <div style="color:#5B6B7F;font-size:14px;margin-top:6px;line-height:1.5;">
                        Grad-CAM outputs are generated alongside the prediction.</div>
                </div>
                """
            )
            return

        prediction = data.get("prediction") or {}
        probabilities = dict(
            explanation.get("probabilities") or prediction.get("probabilities") or {}
        )
        class_names = theme.ordered_classes(list(probabilities))
        target_class = str(explanation.get("target_class_name") or "")
        predicted_class = str(prediction.get("predicted_class_name") or "")

        # SIMPLIFIED: Hide class selector in expander (advanced feature)
        with st.expander("Advanced: View attention for other classes", expanded=False):
            if class_names:
                selector = st.columns([3, 1])
                with selector[0]:
                    choice = st.selectbox(
                        "Class the heatmap explains",
                        ["__predicted__", *class_names],
                        format_func=explain_option_label,
                        key=(
                            "explain_target::"
                            f"{data.get('signature') or data.get('filename')}"
                        ),
                        help="Grad-CAM can be rendered for any class the model "
                        "reports; the prediction itself never changes.",
                    )
                with selector[1]:
                    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
                    recompute = st.button(
                        "🔁 Recompute",
                        width="stretch",
                        help="Re-run Grad-CAM for the selected class.",
                    )
                if recompute:
                    run_explain_for_class(
                        data, None if choice == "__predicted__" else choice
                    )

        heatmap = explanation.get("heatmap_base64")
        overlay = explanation.get("overlay_base64")
        rgb_image = explanation.get("rgb_image_base64")

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # -- Side-by-side clinical view ---------------------------------------
        panels = st.columns(3, gap="medium")
        with panels[0]:
            st.image(
                Image.open(io.BytesIO(data["image_bytes"])),
                caption="Original radiograph",
                width="stretch",
                use_column_width=True,
            )
        with panels[1]:
            if heatmap:
                st.image(
                    Image.open(io.BytesIO(api.decode_base64_png(str(heatmap)))),
                    caption="Activation map",
                    width="stretch",
                    use_column_width=True,
                )
            else:
                st.caption("Heatmap unavailable for this record.")
        with panels[2]:
            if overlay:
                st.image(
                    Image.open(io.BytesIO(api.decode_base64_png(str(overlay)))),
                    caption="Overlay: radiograph + activation",
                    width="stretch",
                    use_column_width=True,
                )
            else:
                st.caption("Overlay unavailable for this record.")

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        # SIMPLIFIED: Hide focus summary and preprocessed image in expander
        with st.expander("Technical details", expanded=False):
            focus = explanation.get("focus") or {}
            if focus.get("summary"):
                theme.render_html(
                    '<div style="margin-top:16px;background:#FEF3C7;'
                    'border:1px solid #FDE68A;border-radius:12px;padding:18px 20px;">'
                    '<div style="font-size:11px;font-weight:700;letter-spacing:0.1em;'
                    'text-transform:uppercase;color:#92400E;margin-bottom:8px;">'
                    'What the model looked at</div>'
                    '<div style="color:#78350F;font-size:14px;line-height:1.6;'
                    'font-weight:500;">'
                    f'{html.escape(str(focus["summary"]))}</div>'
                    '</div>'
                )

            if rgb_image:
                st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
                st.image(
                    Image.open(io.BytesIO(api.decode_base64_png(str(rgb_image)))),
                    caption=(
                        "Resized, normalised tensor rendered back to RGB. The "
                        "heatmap is aligned to this frame, not to the raw file."
                    ),
                    width="stretch",
                )

        theme.render_html('<hr class="ts-divider" style="margin:24px 0;"/>')
        dl_a, dl_b, dl_c = st.columns(3)
        stem = Path(str(data.get("filename", "xray"))).stem
        with dl_a:
            if overlay:
                st.download_button(
                    "⬇ Overlay",
                    data=api.decode_base64_png(str(overlay)),
                    file_name=f"{stem}_gradcam_overlay.png",
                    mime="image/png",
                    width="stretch",
                )
        with dl_b:
            if heatmap:
                st.download_button(
                    "⬇ Heatmap",
                    data=api.decode_base64_png(str(heatmap)),
                    file_name=f"{stem}_gradcam_heatmap.png",
                    mime="image/png",
                    width="stretch",
                )
        with dl_c:
            report = build_summary_report(data)
            st.download_button(
                "⬇ Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"{stem}_explanation.json",
                mime="application/json",
                width="stretch",
            )


def render_clinical_card(analysis: dict[str, Any] | None) -> None:
    """Clinical interpretation summary card."""
    with st.container(border=True, key="card_clinical"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Step 4 · Interpretation</p>
            <p class="ts-card-title">📋 Clinical interpretation summary</p>
            """
        )
        if analysis is None:
            theme.render_html(
                '<p class="ts-card-sub">The structured report, plausibility notes '
                "and export will be generated once an image has been analysed.</p>"
            )
            return

        prediction = analysis["prediction"]
        probabilities = prediction.get("probabilities", {}) or {}
        class_name = str(prediction.get("predicted_class_name", ""))
        confidence = float(prediction.get("confidence", 0.0))

        # One renderer, shared with the Patient History detail view: the stored
        # explanation and a fresh one must look, read and export identically. This
        # is the *prediction's* explanation (with the focus of the prediction's
        # own map), never the focus of a class the reviewer was just inspecting.
        interpretation = api.prediction_interpretation(analysis)
        render_interpretation_block(
            interpretation, class_name, confidence, dict(probabilities)
        )


def build_summary_report(
    analysis: dict[str, Any],
    patient: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the exportable structured report for one assessment."""
    prediction = analysis["prediction"]
    explanation = analysis.get("explanation") or {}
    # Same rule as the storage layer: the prediction's own explanation (the richer
    # /explain copy when it explains the predicted class), never another class's
    # focus borrowed from the Grad-CAM selector.
    interpretation = api.prediction_interpretation(analysis)
    patient = patient or {}
    report: dict[str, Any] = {
        "project": "TBSentinel",
        "record_id": record_id,
        "generated_at": analysis.get("timestamp"),
        "patient": {
            "patient_id": patient.get("patient_id"),
            "name": patient.get("patient_name"),
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "notes": patient.get("notes"),
        },
        "file": analysis.get("filename"),
        "model": analysis.get("strategy"),
        "predicted_class": prediction.get("predicted_class_name"),
        "predicted_class_label": theme.class_label(
            str(prediction.get("predicted_class_name", ""))
        ),
        "confidence": prediction.get("confidence"),
        "probabilities": prediction.get("probabilities"),
        "interpretation": interpretation,
        "focus": interpretation.get("focus"),
        "gradcam": {
            "target_class": explanation.get("target_class_name"),
            "colormap": explanation.get("colormap"),
            "alpha": explanation.get("alpha"),
            "focus": explanation.get("focus"),
        },
        "latency_seconds": analysis.get("latency_seconds"),
        "disclaimer": (
            (interpretation or {}).get("disclaimer")
            or "Research prototype. Not for clinical diagnosis."
        ),
    }
    return report
