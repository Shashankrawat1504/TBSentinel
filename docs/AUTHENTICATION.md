# TBSentinel Authentication Guide

## Overview

The TBSentinel dashboard includes a lightweight authentication layer to protect access to the clinical decision support interface. This is suitable for a research prototype and should be replaced with a proper authentication system for production deployment.

## Default Credentials

- **Username**: `admin`
- **Password**: `tbsentinel2026`

These are demo credentials shipped for the prototype. Set
`TBSENTINEL_USERNAME` / `TBSENTINEL_PASSWORD` before showing the dashboard to
anyone else (see [Configuration](#configuration)).

## How It Works

### Login Flow

1. User accesses the application via `streamlit run frontend/app.py`
2. The app frame (`frontend/app.py`) checks the session first: if it is not
   authenticated it renders the login form (`auth.render_login_page()`) and
   calls `st.stop()`, so no page view ever executes
3. User enters credentials
4. If credentials match, session state is set and the dashboard renders
5. Authenticated pages are also reached through `st.navigation`, so they cannot
   be routed to without passing the guard in step 2

### Session Management

- Sessions are stored in Streamlit's session state
- Sessions automatically expire after 60 minutes of inactivity
- Session timeout is checked on each page load
- Logout button in the sidebar clears the session

### Protected Pages

All dashboard pages are protected:
- Dashboard
- New Assessment
- Patient History
- Performance Metrics
- About & System Info

The guard is centralised: `frontend/app.py` authenticates before it registers
or runs any page, which is stronger than checking inside each view because a
newly added page is private by default.

## Configuration

### Changing Credentials

Preferred: set the environment variables, so the password never lands in the
source tree or in shell history of the repository.

```bash
# Windows (PowerShell)
$env:TBSENTINEL_USERNAME = "your_username"; $env:TBSENTINEL_PASSWORD = "your_password"
streamlit run frontend/app.py
```

```bash
# macOS / Linux / Git Bash
export TBSENTINEL_USERNAME=your_username
export TBSENTINEL_PASSWORD='your_password'
streamlit run frontend/app.py
```

`frontend/auth_config.py` reads those variables and falls back to the demo
credentials, so no code change is needed. While the demo password is still in
use the login page shows a reminder to replace it.

Editing the fallback defaults in `frontend/auth_config.py` also works, but the
values end up in version control:

```python
DEFAULT_USERNAME = "your_username"
DEFAULT_PASSWORD = "your_password"
```

### Session Timeout

Edit `frontend/auth_config.py`:

```python
SESSION_TIMEOUT_MINUTES = 60  # Change to desired timeout
```

## Running the Application

There is one entry point, and it is authenticated:

```bash
streamlit run frontend/app.py
```

The login gate lives inside `app.py`, so there is no separate "authenticated"
and "unauthenticated" way to start the dashboard — starting it without logging
in is not possible.

### Binding

`.streamlit/config.toml` pins Streamlit to `127.0.0.1:8501`, so the dashboard is
reachable only from this machine. Because the login gate is a single shared
password, that is the intended default. For a deliberate LAN demo override it
per run:

```bash
streamlit run frontend/app.py --server.address 0.0.0.0
```

## Files Structure

```
frontend/
├── auth_config.py          # Credentials (env overrides), session keys, timeout
├── auth.py                 # Authentication logic + login page rendering
├── app.py                  # Entry point, auth gate, sidebar, page router
├── login_entry.py          # Legacy standalone login page (not needed)
└── views/
    ├── dashboard.py        # Protected page
    ├── assessment.py       # Protected page
    ├── history.py          # Protected page
    ├── metrics.py          # Protected page
    └── about.py            # Protected page
```

## Security Considerations

### Current Implementation (Prototype)

- Credentials stored in plain text in a config file
- No password hashing
- No rate limiting
- No audit logging
- Session state only (no persistent storage)

### Production Recommendations

For production deployment, implement:

1. **Password Hashing**: Use bcrypt or Argon2 for password storage
2. **Database Authentication**: Store user credentials in a secure database
3. **OAuth/OIDC**: Integrate with enterprise identity providers (Azure AD, Okta, etc.)
4. **Rate Limiting**: Prevent brute force attacks
5. **Audit Logging**: Track login attempts and access
6. **HTTPS**: Use TLS for all communications
7. **Multi-Factor Authentication**: Add 2FA for sensitive operations
8. **Session Management**: Use secure session cookies with HttpOnly and Secure flags

## Troubleshooting

### Login Not Working

1. Check `TBSENTINEL_USERNAME` / `TBSENTINEL_PASSWORD` in your environment, then
   the fallbacks in `frontend/auth_config.py`
2. Ensure you are running `streamlit run frontend/app.py` (there is no second
   entry point that changes the login behaviour)
3. Check the browser console for errors
4. Clear the Streamlit cache: `streamlit cache clear`

### Session Expiring Too Quickly

Increase `SESSION_TIMEOUT_MINUTES` in `frontend/auth_config.py`.

### Can't Access Any Page

Ensure the FastAPI backend is running:
```bash
python -m src.inference.main
```

## Customization

### Changing Login Page Design

Edit the `render_login_page()` function in `frontend/auth.py` to customize:
- Colors and styling
- Logo and branding
- Disclaimer text
- Form layout

### Adding "Remember Me" Functionality

The "Remember me" checkbox is already present in the login form. To implement persistent sessions:
1. Store a secure token in browser cookies/local storage
2. Check for the token on page load
3. Validate token against a database
4. Auto-login if token is valid

This requires additional implementation beyond the current prototype.

## Support

For issues or questions, refer to the main project README or open an issue in the repository.
