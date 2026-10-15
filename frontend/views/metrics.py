"""Performance Metrics — centralized vs federated model comparison."""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from tbui import api, components, metrics, theme

api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
health = components.cached_health(api_url)

components.render_topbar(
    "Performance Metrics",
    "Centralized vs federated model evaluation on the validation split",
    health,
)

rows = metrics.delta_vs_baseline(metrics.load_comparison())
baseline = rows[0]
served = metrics.load_multiclass()

# ---------------------------------------------------------------------------
# Served model — three-class screening (Normal / TB / Pneumonia)
# ---------------------------------------------------------------------------
# This is the model the API loads by default, so it is reported first.

theme.render_html(
    '<p class="ts-eyebrow" style="margin-top:6px;">Served model</p>'
    '<p class="ts-card-title" style="font-size:20px;">'
    "🧪 Three-class screening — Normal / TB / Pneumonia</p>"
    '<p class="ts-card-sub">Metrics of <code>'
    "multiclass_efficientnet_b3_best.pt</code>, the checkpoint the inference "
    "API serves by default (EfficientNet-B3, 300×300 inputs).</p>"
)

if served is None:
    theme.alert(
        "warning",
        "Three-class metrics unavailable",
        "<code>artifacts/outputs/multiclass_test_metrics.json</code> was not "
        "found, so the held-out metrics of the served model cannot be shown. "
        "Run the multi-class evaluation script to regenerate it — no numbers are "
        "estimated here.",
    )
else:
    theme.alert(
        "info",
        "How to read these numbers",
        f"Measured once on the held-out multi-class test split "
        f"(<b>{served['samples']:,} images</b>: "
        + ", ".join(
            f"{int(served['per_class'].get(name, {}).get('support', 0)):,} "
            f"{theme.class_label(name)}"
            for name in served["class_names"]
        )
        + ") at the best-validation checkpoint"
        + (
            f" (epoch {served['checkpoint_epoch']})"
            if served.get("checkpoint_epoch") is not None
            else ""
        )
        + ". Every image is scored as exactly one of the three classes, so "
        "per-class recall — not accuracy — is the number to trust when a "
        "missed case is costly.",
    )

    mc_cols = st.columns(4)
    with mc_cols[0]:
        theme.render_html(
            theme.stat_card(
                "Accuracy (3-class)",
                f"{served['accuracy']:.2%}",
                "🎯",
                foot=f"{served['samples']:,} held-out images",
            )
        )
    with mc_cols[1]:
        theme.render_html(
            theme.stat_card(
                "Macro-F1",
                f"{served['macro_f1']:.4f}",
                "⚖️",
                foot="Unweighted mean over the 3 classes",
                tone="success",
            )
        )
    with mc_cols[2]:
        theme.render_html(
            theme.stat_card(
                "Macro AUC (one-vs-rest)",
                f"{served['macro_auc']:.4f}",
                "📈",
                foot="Ranking quality, threshold-free",
            )
        )
    with mc_cols[3]:
        tb_recall = float(served["per_class"].get("tb", {}).get("recall", 0.0))
        theme.render_html(
            theme.stat_card(
                "TB recall",
                f"{tb_recall:.2%}",
                "⚠️",
                foot="Safety-critical: missed TB cases",
                tone="danger" if tb_recall < 0.9 else "success",
            )
        )

    st.markdown("")

    with st.container(border=True, key="card_multiclass"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Per-class performance</p>
            <p class="ts-card-title">📋 Class-by-class breakdown</p>
            <p class="ts-card-sub">Each column is computed for one class
            against the other two (one-vs-rest AUC) on the held-out test
            split. Support is the number of true images of that class.</p>
            """
        )

        header = (
            "<th>Class</th>"
            + "".join(
                f"<th class='ts-num'>{label}</th>"
                for _, label in (("precision", "Precision"), ("recall", "Recall"),
                                 ("f1", "F1"), ("auc", "AUC-OVR"),
                                 ("support", "Support"))
            )
        )
        body = ""
        for name in theme.ordered_classes(served["class_names"]):
            stats = served["per_class"][name]
            body += f"""
            <tr>
                <td>{theme.badge(name)}</td>
                <td class='ts-num'>{stats['precision']:.4f}</td>
                <td class='ts-num'>{stats['recall']:.4f}</td>
                <td class='ts-num'>{stats['f1']:.4f}</td>
                <td class='ts-num'>{stats['auc']:.4f}</td>
                <td class='ts-num'>{stats['support']:,}</td>
            </tr>"""

        theme.render_html(
            f"""
            <table class="ts-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{body}</tbody>
            </table>
            """
        )

        # Recall bars: the screening-critical view of the same numbers.
        theme.render_html(
            '<p class="ts-eyebrow" style="margin-top:18px;">Compared recall</p>'
            '<p class="ts-card-sub" style="margin-bottom:2px;">Share of each '
            'true class the model finds. Green/red/amber match the result '
            'badges used across the dashboard.</p>'
        )
        for name in theme.ordered_classes(served["class_names"]):
            recall = float(served["per_class"][name]["recall"])
            kind = theme.class_kind(name)
            theme.render_html(
                f"""
                <div class="ts-cmp-row">
                    <div class="ts-cmp-head">
                        <span>{html.escape(theme.class_label(name))}</span>
                        <span>{recall:.2%}</span>
                    </div>
                    <div class="ts-cmp-track">
                        <div class="ts-cmp-fill ts-cmp-fill--{kind}"
                             style="width:{recall * 100:.2f}%"></div>
                    </div>
                </div>
                """
            )

    st.markdown("")

    with st.container(border=True, key="card_multiclass_cm"):
        matrix = served["confusion_matrix"] or []
        classes = theme.ordered_classes(served["class_names"])
        theme.render_html(
            """
            <p class="ts-eyebrow">Error analysis</p>
            <p class="ts-card-title">🧮 Three-class confusion matrix</p>
            <p class="ts-card-sub">Rows are the true class, columns the model's
            prediction. Off-diagonal cells are the confusions, and the pair to
            watch is pneumonia read as normal (a missed opacity).</p>
            """
        )
        if matrix and len(matrix) == len(classes):
            head = "<th></th>" + "".join(
                f"<th class='ts-num'>Pred {html.escape(theme.class_label(name))}</th>"
                for name in classes
            )
            rows_html = ""
            for index, name in enumerate(classes):
                cells = "".join(
                    f"<td class='ts-num'>{int(value)}</td>"
                    for value in matrix[index]
                )
                rows_html += (
                    f"<tr><td><b>True {html.escape(theme.class_label(name))}</b></td>"
                    f"{cells}</tr>"
                )
            theme.render_html(
                f"""
                <table class="ts-table" style="max-width:720px;">
                    <thead><tr>{head}</tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """
            )
        else:
            theme.alert("warning", "Confusion matrix unavailable",
                        "No square matrix was found in the metrics artefact.")

        validation = served.get("validation") or {}
        if validation:
            theme.render_html(
                '<p class="ts-card-sub" style="margin-top:14px;">'
                f"Best-validation checkpoint (epoch "
                f"{served.get('checkpoint_epoch')}): accuracy "
                f"<b>{float(validation.get('accuracy', 0.0)):.4f}</b>, macro-F1 "
                f"<b>{float(validation.get('macro_f1', 0.0)):.4f}</b> on "
                f"{int(validation.get('samples', 0)):,} validation images.</p>"
            )

    st.markdown("")

# ---------------------------------------------------------------------------
# Binary TB benchmark — centralized vs federated
# ---------------------------------------------------------------------------

theme.render_html(
    '<p class="ts-eyebrow" style="margin-top:18px;">Binary TB benchmark</p>'
    '<p class="ts-card-title" style="font-size:20px;">'
    "🏥 Centralized vs federated (TB / Non-TB)</p>"
    '<p class="ts-card-sub">The phase-2 experiment that compared a centralized '
    "TB model with FedAvg and FedProx across simulated hospitals. It is a "
    '<b>binary</b> model, kept here as the federated-learning evidence base '
    "and not the model currently served.</p>"
)

theme.alert(
    "info",
    "How to read these numbers",
    "All metrics are measured on the official <b>1,800-image TBX11K validation "
    "split</b> (1,600 Non-TB / 200 TB) using each strategy's best binary "
    "checkpoint. The federated models were trained across <b>three simulated "
    "hospital clients</b> holding non-IID local shards — no raw images ever left "
    "a client. Sensitivity and specificity below are binary definitions and do "
    "not apply to the three-class model above.",
)

# ---------------------------------------------------------------------------
# Headline cards
# ---------------------------------------------------------------------------

card_cols = st.columns(len(rows))
for col, row in zip(card_cols, rows):
    tone = "" if row["key"] == "centralized" else "success"
    with col:
        theme.render_html(
            theme.stat_card(
                row["name"],
                f"{float(row['accuracy']):.2%}",
                "🎯",
                foot=f"Accuracy · AUC-ROC {float(row['auc']):.4f}",
                tone=tone,
            )
        )

st.markdown("")

# ---------------------------------------------------------------------------
# Full comparison table
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_compare"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Evaluation</p>
        <p class="ts-card-title">📊 Centralized vs Federated comparison</p>
        <p class="ts-card-sub">Best-validation metrics per strategy, with the
        federated delta relative to the centralized baseline.</p>
        """
    )

    metric_labels = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("auc", "AUC-ROC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]

    def _delta_html(row: dict, key: str) -> str:
        if row["key"] == "centralized":
            return '<span class="ts-delta">—</span>'
        delta = float(row.get(f"delta_{key}", 0.0))
        cls = "ts-delta--up" if delta >= 0 else "ts-delta--down"
        sign = "+" if delta >= 0 else ""
        return f'<span class="ts-delta {cls}">{sign}{delta:.4f}</span>'

    body = ""
    for row in rows:
        cells = "".join(
            f"<td class='ts-num'>{float(row[key]):.4f}</td>" for key, _ in metric_labels
        )
        deltas = "".join(
            f"<td class='ts-num'>{_delta_html(row, key)}</td>" for key, _ in metric_labels
        )
        body += f"""
        <tr>
            <td><b>{html.escape(row['name'])}</b><br/>
                <span style="color:#5B6B7F;font-size:11.5px;">
                {html.escape(str(row.get('source') or ''))}</span></td>
            {cells}{deltas}
        </tr>"""

    value_headers = "".join(
        f"<th class='ts-num'>{label}</th>" for _, label in metric_labels
    )
    delta_headers = "".join(
        f"<th class='ts-num' style='color:#0E7490;'>&Delta; {label}</th>"
        for _, label in metric_labels
    )

    theme.render_html(
        f"""
        <table class="ts-table">
            <thead>
                <tr>
                    <th rowspan="2" style="vertical-align:bottom;">Model</th>
                    <th colspan="{len(metric_labels)}" class="ts-num"
                        style="text-align:right;">Metric value</th>
                    <th colspan="{len(metric_labels)}" class="ts-num"
                        style="text-align:right;color:#0E7490;">
                        &Delta; vs centralized baseline</th>
                </tr>
                <tr>{value_headers}{delta_headers}</tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
        """
    )

# ---------------------------------------------------------------------------
# Per-metric bars
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_bars"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Metric profiles</p>
        <p class="ts-card-title">📈 Metric-by-metric comparison</p>
        <p class="ts-card-sub">Each bar shows the metric value (0–1). Blue is the
        centralized baseline, teal FedAvg and slate FedProx.</p>
        """
    )
    bar_classes = {"centralized": "cen", "fedavg": "fed", "fedprox": "fedp"}
    for key, label in metric_labels:
        theme.render_html(f'<p style="font-size:13.5px;font-weight:700;'
                          f'color:#0F172A;margin:14px 0 6px 0;">{label}</p>')
        for row in rows:
            value = float(row[key])
            fill = bar_classes.get(row["key"], "cen")
            theme.render_html(
                f"""
                <div class="ts-cmp-row">
                    <div class="ts-cmp-head">
                        <span>{html.escape(row['name'])}</span>
                        <span>{value:.4f}</span>
                    </div>
                    <div class="ts-cmp-track">
                        <div class="ts-cmp-fill ts-cmp-fill--{fill}"
                             style="width:{value * 100:.2f}%"></div>
                    </div>
                </div>
                """
            )

# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_confusion"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Error analysis</p>
        <p class="ts-card-title">🧮 Confusion matrices</p>
        <p class="ts-card-sub">Rows are the true class, columns the prediction
        (Non-TB, then TB) on the 1,800-image validation split.</p>
        """
    )
    cm_cols = st.columns(len(rows))
    for col, row in zip(cm_cols, rows):
        matrix = row.get("confusion_matrix") or [[0, 0], [0, 0]]
        with col:
            theme.render_html(
                f"""
                <div style="background:#FAFCFE;border:1px solid var(--ts-border);
                            border-radius:12px;padding:14px;">
                    <div style="font-weight:700;font-size:13.5px;color:#0F172A;
                                margin-bottom:10px;text-align:center;">
                        {html.escape(row['name'])}</div>
                    <table class="ts-table" style="font-size:12.5px;">
                        <thead>
                            <tr><th></th><th class="ts-num">Pred NB</th>
                                <th class="ts-num">Pred TB</th></tr>
                        </thead>
                        <tbody>
                            <tr><td><b>True NB</b></td>
                                <td class="ts-num">{matrix[0][0]}</td>
                                <td class="ts-num">{matrix[0][1]}</td></tr>
                            <tr><td><b>True TB</b></td>
                                <td class="ts-num">{matrix[1][0]}</td>
                                <td class="ts-num">{matrix[1][1]}</td></tr>
                        </tbody>
                    </table>
                </div>
                """
            )

# ---------------------------------------------------------------------------
# What the metrics mean
# ---------------------------------------------------------------------------

theme.render_html(
    """
    <p class="ts-eyebrow" style="margin-top:18px;">Reference</p>
    <p class="ts-card-title">📖 What these metrics mean</p>
    """
)
help_cols = st.columns(2)
for index, key in enumerate(metrics.METRIC_KEYS):
    title, description = metrics.METRIC_HELP[key]
    with help_cols[index % 2]:
        theme.render_html(
            f"""
            <div style="background:#fff;border:1px solid var(--ts-border);
                        border-radius:12px;padding:14px 16px;margin-bottom:10px;
                        box-shadow:var(--ts-shadow);">
                <div style="font-weight:700;font-size:13.5px;color:#0F172A;
                            margin-bottom:4px;">{html.escape(title)}</div>
                <div style="font-size:13px;line-height:1.65;color:#334155;">
                    {html.escape(description)}</div>
            </div>
            """
        )

st.markdown("")

with st.expander("🧠 Why federated models trail the centralized baseline", expanded=False):
    st.markdown(
        """
Both federated strategies trail the centralized baseline on accuracy, macro-F1,
AUC and sensitivity — and this is expected. The centralized model saw all
**6,600 training images** every epoch, while each federated client trains on a
small, skewed local shard for a single epoch per round before FedAvg/FedProx
aggregates the updates.

**Sensitivity is the weak metric for every model**: only 200 of the 1,800
validation images are TB-positive, and the extreme Dirichlet partition gives one
client TB-only data. In a screening workflow, sensitivity should be treated as
the primary metric — a federated deployment would typically lower the decision
threshold to recover recall at the cost of some specificity.
        """
    )

theme.render_html(
    '<p class="ts-card-sub" style="text-align:center;margin-top:12px;">'
    "Three-class metrics loaded from "
    "<code>artifacts/outputs/multiclass_test_metrics.json</code>; binary "
    "comparison from <code>artifacts/outputs/federated_*_results.json</code> "
    "and <code>docs/federated_learning.md</code>.</p>"
)
