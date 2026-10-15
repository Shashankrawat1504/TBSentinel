"""Authentication module for TBSentinel dashboard.

Provides login functionality, session management, and authentication guards
for protecting dashboard pages.
"""

from __future__ import annotations

import datetime
from typing import Literal

import streamlit as st

from auth_config import (
    AUTH_CREDENTIALS,
    SESSION_KEYS,
    SESSION_TIMEOUT_MINUTES,
    USING_DEFAULT_CREDENTIALS,
)


def is_authenticated() -> bool:
    """Check if the current session is authenticated."""
    return st.session_state.get(SESSION_KEYS["authenticated"], False)


def check_session_timeout() -> bool:
    """Check if the session has timed out and logout if so."""
    if not is_authenticated():
        return False

    login_time = st.session_state.get(SESSION_KEYS["login_time"])
    if login_time is None:
        logout()
        return False

    # Check if session has expired
    elapsed = (datetime.datetime.now() - login_time).total_seconds()
    if elapsed > SESSION_TIMEOUT_MINUTES * 60:
        logout()
        st.warning("Session expired. Please login again.")
        return False

    return True


def login(username: str, password: str) -> Literal[True]:
    """Attempt to authenticate with the provided credentials.

    Args:
        username: Username to authenticate
        password: Password to authenticate

    Returns:
        True if authentication successful, raises error otherwise

    Raises:
        ValueError: If credentials are invalid
    """
    if username == AUTH_CREDENTIALS["username"] and password == AUTH_CREDENTIALS["password"]:
        st.session_state[SESSION_KEYS["authenticated"]] = True
        st.session_state[SESSION_KEYS["username"]] = username
        st.session_state[SESSION_KEYS["login_time"]] = datetime.datetime.now()
        return True
    else:
        raise ValueError("Invalid username or password")


def logout() -> None:
    """Clear the current authentication session."""
    for key in SESSION_KEYS.values():
        if key in st.session_state:
            del st.session_state[key]


def get_current_user() -> str | None:
    """Get the currently authenticated username."""
    return st.session_state.get(SESSION_KEYS["username"])


def render_login_page() -> None:
    """Render the login page with professional medical styling."""
    # Inject professional theme styles
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        [data-testid="stHeader"] { background: transparent; padding: 0; }
        [data-testid="stHeader"] > div { display: none; }
        #MainMenu, footer { visibility: hidden; height: 0; }
        [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 500px;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            padding: 12px 24px;
            border: 1px solid #0A2647;
            transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 14px;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0A2647 0%, #2C74B3 100%);
            color: #fff;
            border: 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            padding: 12px 16px;
            font-size: 14px;
            transition: all .2s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0A2647;
            box-shadow: 0 0 0 3px rgba(10, 38, 71, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Professional login container
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:32px;">
            <div style="font-size:64px;margin-bottom:16px;">🏥</div>
            <h1 style="color:#0A2647;font-size:32px;font-weight:800;margin:0 0 8px 0;letter-spacing:-0.02em;">
                TBSentinel
            </h1>
            <p style="color:#64748B;font-size:15px;font-weight:500;margin:0;">
                Clinical Decision Support System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Login form
    with st.form("login_form", clear_on_submit=False, border=True):
        st.markdown(
            """
            <p style="color:#1E293B;font-size:14px;font-weight:600;margin-bottom:16px;">
                Sign in to continue
            </p>
            """,
            unsafe_allow_html=True,
        )

        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password",
        )

        remember_me = st.checkbox("Remember me", value=False, key="remember_me")

        submitted = st.form_submit_button(
            "Sign In",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                try:
                    login(username, password)
                    st.success("✅ Login successful! Redirecting to dashboard...")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # Disclaimer
    st.markdown(
        """
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-left:4px solid #D97706;
                    border-radius:12px;padding:16px;margin-top:24px;">
            <div style="color:#92400E;font-size:13px;font-weight:700;margin-bottom:4px;">
                ⚠️ Research Prototype – Authorized users only
            </div>
            <div style="color:#78350F;font-size:12px;line-height:1.6;">
                This is a research prototype for demonstration purposes only.
                Do not use for clinical diagnosis. Access is restricted to authorized personnel.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if USING_DEFAULT_CREDENTIALS:
        st.caption(
            "Prototype note: the shipped demo credentials are still active. "
            "Set TBSENTINEL_PASSWORD (and TBSENTINEL_USERNAME) before showing "
            "this dashboard to anyone else."
        )

    # Footer
    st.markdown(
        """
        <div style="text-align:center;color:#94A3B8;font-size:12px;margin-top:32px;padding-top:24px;
                    border-top:1px solid #E2E8F0;">
            TBSentinel v1.0.0 · Research Prototype · Not for Clinical Use
        </div>
        """,
        unsafe_allow_html=True,
    )
