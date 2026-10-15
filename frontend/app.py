"""TBSentinel Clinical Decision Support System — dashboard entrypoint.

A professional multi-page medical dashboard built on Streamlit. This file is
the application frame only: it configures the page, loads the clinical theme,
renders the persistent sidebar (navigation + system health) and routes to the
current view. All presentation, API and storage logic lives in ``frontend/tbui``.

Pages
------
* Dashboard          — overview, system status, KPIs, recent assessments
* New Assessment     — patient details, upload, prediction, Grad-CAM, save
* Patient History    — searchable record browser with detail view + export
* Performance Metrics— centralized vs federated (FedAvg/FedProx) comparison
* About              — project, model card, disclaimer, tech stack

Run from the project root with::

    python -m src.inference.main          # start the FastAPI backend first
    streamlit run frontend/app.py         # then this dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``frontend/`` importable so pages can ``from tbui import ...`` even when
# Streamlit executes this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from auth import check_session_timeout, get_current_user, is_authenticated, login, logout, render_login_page
from tbui import api, components, storage, theme

st.set_page_config(
    page_title="TBSentinel | TB Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

# If not authenticated, show login page
if not is_authenticated():
    # Change layout for login page
    st.set_page_config(
        page_title="Login | TBSentinel",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Hide sidebar for login
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_login_page()
    st.stop()

# Check session timeout
check_session_timeout()

PAGES_DIR = Path(__file__).resolve().parent / "views"

# (title, icon, file, subtitle) — order defines sidebar navigation order.
PAGE_SPECS: list[tuple[str, str, str, str]] = [
    (
        "Dashboard",
        ":material/dashboard:",
        "dashboard.py",
        "",
    ),
    (
        "New Assessment",
        ":material/monitor_heart:",
        "assessment.py",
        "",
    ),
    (
        "Patient History",
        ":material/history:",
        "history.py",
        "",
    ),
    (
        "Performance Metrics",
        ":material/monitoring:",
        "metrics.py",
        "",
    ),
    (
        "About & System Info",
        ":material/info:",
        "about.py",
        "",
    ),
]


def build_navigation() -> tuple[object, list[st.Page]]:
    """Create the hidden router plus the Page objects used by the sidebar nav."""
    pages = [
        st.Page(
            str(PAGES_DIR / filename),
            title=title,
            icon=icon,
            default=(index == 0),
        )
        for index, (title, icon, filename, _sub) in enumerate(PAGE_SPECS)
    ]
    # Navigation is rendered manually in the sidebar so the brand block can sit
    # above the menu; st.page_link still routes through these registered pages.
    current = st.navigation(pages, position="hidden")
    return current, pages


def render_sidebar_nav(pages: list[st.Page]) -> None:
    """Custom sidebar navigation (brand → menu) for the app frame."""
    with st.sidebar:
        theme.render_html(
            """
            <div class="ts-brand-block">
                <div class="ts-brand-mark">
                    <span style="color:#fff;font-weight:800;font-size:16px;">T</span>
                </div>
                <div>
                    <div class="ts-brand-name">TBSentinel</div>
                    <div class="ts-brand-sub">Clinical Decision Support</div>
                </div>
            </div>
            <hr class="ts-divider" style="margin:10px 0 16px 0;"/>
            <p class="ts-nav-label">Clinical workflow</p>
            """
        )
        for page, spec in zip(pages[:3], PAGE_SPECS[:3]):
            st.page_link(page, label=spec[0], icon=spec[1], help=spec[3],
                         width="stretch")
        theme.render_html(
            '<p class="ts-nav-label" style="margin-top:16px;">Model &amp; system</p>'
        )
        for page, spec in zip(pages[3:], PAGE_SPECS[3:]):
            st.page_link(page, label=spec[0], icon=spec[1], help=spec[3],
                         width="stretch")
        theme.render_html('<hr class="ts-divider" style="margin:16px 0;"/>')

        # User info and logout
        current_user = get_current_user()
        if current_user:
            st.markdown(f"**Logged in as:** {current_user}")

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        if st.button("Logout", icon=":material/logout:", use_container_width=True):
            logout()
            st.success("Logged out successfully!")
            st.rerun()


def main() -> None:
    """Application frame: theme → router → sidebar → current page."""
    theme.inject_styles()

    # Ensure the local record store exists before any page touches it.
    try:
        storage.init_db()
    except OSError:
        pass

    current, pages = build_navigation()

    api_url = st.session_state.get("api_url", api.DEFAULT_API_URL)
    health = components.cached_health(api_url)

    render_sidebar_nav(pages)
    # Sidebar status + connection panel (shared across all pages).
    new_url = components.render_sidebar(health)
    if new_url != api_url:
        health = components.cached_health(new_url)

    current.run()
    components.render_footer()


if __name__ == "__main__":
    main()
