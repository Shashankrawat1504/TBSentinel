"""Authentication configuration for TBSentinel dashboard.

For this research prototype, credentials come from here or — preferably — from
the environment, so the shared demo password never has to be edited into (or
committed to) the source tree:

    TBSENTINEL_USERNAME=dr.smith TBSENTINEL_PASSWORD='<strong-password>' \
        streamlit run frontend/app.py

In production, this should be replaced with a proper authentication system
(e.g., OAuth, database-backed auth, or enterprise identity provider).
"""

from __future__ import annotations

import os

# Default credentials for the research prototype (see the module docstring for
# the environment overrides). TODO: Replace with proper authentication.
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "tbsentinel2026"

USERNAME_ENV = "TBSENTINEL_USERNAME"
PASSWORD_ENV = "TBSENTINEL_PASSWORD"

AUTH_CREDENTIALS = {
    "username": os.environ.get(USERNAME_ENV, DEFAULT_USERNAME),
    "password": os.environ.get(PASSWORD_ENV, DEFAULT_PASSWORD),
}

# True while the shipped demo password is still in use, so the UI can remind the
# operator to set TBSENTINEL_PASSWORD before showing the dashboard to anyone.
USING_DEFAULT_CREDENTIALS = (
    AUTH_CREDENTIALS["password"] == DEFAULT_PASSWORD
    and AUTH_CREDENTIALS["username"] == DEFAULT_USERNAME
)

# Session state keys
SESSION_KEYS = {
    "authenticated": "authenticated",
    "username": "username",
    "login_time": "login_time",
}

# Security settings
SESSION_TIMEOUT_MINUTES = 60  # Auto-logout after 60 minutes of inactivity
