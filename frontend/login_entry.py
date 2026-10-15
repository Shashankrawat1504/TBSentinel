"""Legacy standalone login page for TBSentinel (kept only for reference).

This page renders the login form but cannot take the user anywhere: the real
application, including its authentication gate, is ``frontend/app.py``. Run
that instead — from here, always log in through the app itself.

    streamlit run frontend/app.py      # the dashboard (correct entry point)
    streamlit run frontend/login_entry.py   # this legacy page, for reference only
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make frontend/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from auth import is_authenticated, render_login_page


def main() -> None:
    """Render the login page or redirect if already authenticated."""
    # Set page config for login page
    st.set_page_config(
        page_title="Login | TBSentinel",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.warning(
        "Legacy page: this is not the application. Start the dashboard with "
        "`streamlit run frontend/app.py` — it has the same login gate and then "
        "opens the real interface.",
        icon="⚠️",
    )

    # Hide the sidebar for login page
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

    # If already authenticated, show redirect instructions
    if is_authenticated():
        st.success("You are already logged in!")
        st.info("Please open the main dashboard in a new terminal:")
        st.code("streamlit run frontend/app.py", language="bash")
        st.markdown("---")
        if st.button("Open Dashboard in New Tab", type="primary"):
            st.info("Please run the command above in a new terminal")
        return

    # Render the login form
    render_login_page()


if __name__ == "__main__":
    main()
