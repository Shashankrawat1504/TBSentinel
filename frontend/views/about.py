"""About & System Info — project description, model card, disclaimer, stack."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from tbui import api, components, theme

APP_VERSION = "1.0.0"

api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
health = components.cached_health(api_url)
summary = api.health_summary(health)

components.render_topbar(
    "About & System Info",
    "Project description, model card, disclaimer and tech stack",
    health,
)

# ---------------------------------------------------------------------------
# Project description
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_project"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Project</p>
        <p class="ts-card-title">🫁 About TBSentinel</p>
        """
    )
    st.markdown(
        """
**TBSentinel** is a privacy-preserving, federated multimodal clinical decision
support system for the detection of **tuberculosis (TB)** from chest X-ray
images.

The system enables multiple healthcare institutions to collaboratively train a
robust deep learning model **without sharing raw patient data**, addressing
critical challenges of data privacy, data heterogeneity and limited labeled
medical datasets in resource-constrained settings.

**What this prototype demonstrates**

- **Federated learning** — FedAvg and FedProx aggregation across simulated
  hospital clients (Flower), with a centralized baseline for comparison.
- **Explainable AI** — Grad-CAM attention maps generated for every prediction so
  clinicians can see *where* the model looked.
- **Clinical workflow** — patient-linked assessments, local record storage,
  searchable history and exportable structured summaries.
- **Transparent evaluation** — accuracy, macro-F1, AUC-ROC, sensitivity and
  specificity published for every training strategy.

> Built as a **research prototype** to support early TB triage discussions —
> not as a substitute for radiologist interpretation or laboratory confirmation.
        """
    )

# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

model_left, model_right = st.columns([1, 1], gap="large")

with model_left:
    with st.container(border=True, key="card_model"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Model card</p>
            <p class="ts-card-title">🧠 Model information</p>
            """
        )
        theme.kv_list(
            [
                ("Architecture", "EfficientNet-B3 (ImageNet-pretrained)"),
                ("Served task", f"{summary['task_label']} — {summary['class_list']}"),
                ("Input", "Frontal chest radiograph, resized to 300×300"),
                (
                    "Training data",
                    "12,029 images: TBX11K TB + RSNA Normal / Lung Opacity",
                ),
                ("Splits (train/val/test)", "8,419 / 1,805 / 1,805"),
                ("Federated clients", "3 simulated hospitals (non-IID Dirichlet)"),
                ("Aggregation", "FedAvg and FedProx (µ = 0.01)"),
                ("Explainability", "Grad-CAM (auto-detected target layer)"),
                ("Serving", "FastAPI — /predict and /explain endpoints"),
            ]
        )

with model_right:
    with st.container(border=True, key="card_runtime"):
        theme.render_html(
            """
            <p class="ts-eyebrow">Live system</p>
            <p class="ts-card-title">⚙️ Current runtime</p>
            """
        )
        theme.kv_list(
            [
                ("App version", APP_VERSION),
                ("Inference API", api_url),
                ("API status", "Online" if summary["api_online"] else "Offline"),
                ("Model loaded", "Yes" if summary["model_loaded"] else "No"),
                ("Serving model", summary["strategy"]),
                ("Checkpoint", summary["checkpoint_name"]),
                ("Compute device", summary["device"]),
                ("Python", platform.python_version()),
                ("Platform", platform.platform(terse=True)),
            ]
        )

# ---------------------------------------------------------------------------
# Tech stack
# ---------------------------------------------------------------------------

with st.container(border=True, key="card_stack"):
    theme.render_html(
        """
        <p class="ts-eyebrow">Engineering</p>
        <p class="ts-card-title">🛠️ Technical stack</p>
        """
    )
    stack_cols = st.columns(2)
    with stack_cols[0]:
        st.markdown(
            """
**Frontend**
- Streamlit multi-page dashboard (this app)
- Custom clinical design system (theme + components)

**Backend**
- FastAPI inference service (`/health`, `/predict`, `/explain`)
- PyTorch + `timm` EfficientNet-B3
- OpenCV + PyTorch-Grad-CAM
            """
        )
    with stack_cols[1]:
        st.markdown(
            """
**Machine learning**
- PyTorch 2.x training pipelines
- Flower federated simulation (FedAvg / FedProx)
- Weights & Biases experiment tracking

**Data & storage**
- TBX11K chest X-ray benchmark
- SQLite local record store
- Docker / Docker Compose deployment
            """
        )

theme.render_html(
    f"""
    <div style="text-align:center;margin-top:16px;color:#5B6B7F;font-size:12.5px;">
        TBSentinel v{APP_VERSION} · Federated Multimodal CDSS ·
        EfficientNet-B3 + Grad-CAM · Built with Streamlit, FastAPI and Flower
    </div>
    """
)
