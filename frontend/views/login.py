"""Login page for TBSentinel dashboard.

This is the first page users see. It requires authentication before
allowing access to any other dashboard pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make frontend/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from auth import is_authenticated, render_login_page


def main() -> None:
    """Render the login page."""
    # Set page config for login page
    st.set_page_config(
        page_title="Login | TBSentinel",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed",
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
        st.success("✅ You are already logged in!")
        st.info("Please open the main dashboard in a new terminal:")
        st.code("streamlit run frontend/app.py", language="bash")
        st.markdown("---")
        if st.button("Open Dashboard", type="primary"):
            st.info("Please run the command above in a new terminal")
        return

    # Render the login form
    render_login_page()


if __name__ == "__main__":
    main()
