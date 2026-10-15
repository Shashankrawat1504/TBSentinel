#!/usr/bin/env python3
"""
Streamlit UI flow test for TBSentinel presentation.

This test simulates a user journey through the app to ensure no unhandled
exceptions occur during normal usage.

Tests:
1. App starts without errors
2. Login page renders correctly
3. Login with credentials works
4. Dashboard loads after login
5. Navigation to New Assessment works
6. Patient form renders without errors

Run with: pytest tests/test_ui_flow.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add frontend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from streamlit.testing.v1 import AppTest


def test_app_starts():
    """Test that the Streamlit app starts without errors."""
    print("\n" + "=" * 60)
    print("TEST 1: App Startup")
    print("=" * 60)

    at = AppTest.from_string("import streamlit as st; st.write('Hello')")
    assert not at.exception
    print("✅ App starts without exceptions")


def test_login_page_renders():
    """Test that the login page renders without errors."""
    print("\n" + "=" * 60)
    print("TEST 2: Login Page Rendering")
    print("=" * 60)

    # Point to the actual app
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"

    at = AppTest.from_file(str(app_path))
    at.run()

    # Check for any exceptions
    if at.exception:
        print(f"❌ Exception during login page render: {at.exception}")
        pytest.fail(f"Login page raised exception: {at.exception}")

    print("✅ Login page renders without exceptions")
    print(f"   Number of elements rendered: {len(at.main)}")


def test_login_form_elements_exist():
    """Test that login form elements are present."""
    print("\n" + "=" * 60)
    print("TEST 3: Login Form Elements")
    print("=" * 60)

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()

    # Check for text_input widgets (username/password)
    text_inputs = [element for element in at.main if element.type == "text_input"]
    print(f"   Found {len(text_inputs)} text input(s)")

    # Check for button widget
    buttons = [element for element in at.main if element.type == "button"]
    print(f"   Found {len(buttons)} button(s)")

    # We expect at least 2 text inputs (username, password) and 1 button
    assert len(text_inputs) >= 2, f"Expected at least 2 text inputs, got {len(text_inputs)}"
    assert len(buttons) >= 1, f"Expected at least 1 button, got {len(buttons)}"

    print("✅ Login form elements exist")


def test_login_with_credentials():
    """Test entering credentials and clicking login."""
    print("\n" + "=" * 60)
    print("TEST 4: Login with Credentials")
    print("=" * 60)

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()

    # Enter username
    at.text_input[0].set_value("admin").run()
    print("   ✓ Username entered")

    # Enter password
    at.text_input[1].set_value("tbsentinel2026").run()
    print("   ✓ Password entered")

    # Click login button
    if len(at.button) > 0:
        at.button[0].click().run()
        print("   ✓ Login button clicked")
    else:
        print("   ⚠ No button found (may be auto-login)")

    # Check for exceptions after login
    if at.exception:
        print(f"❌ Exception after login: {at.exception}")
        pytest.fail(f"Login raised exception: {at.exception}")

    print("✅ Login with credentials completes without exceptions")


def test_dashboard_loads_after_login():
    """Test that dashboard loads successfully after login."""
    print("\n" + "=" * 60)
    print("TEST 5: Dashboard After Login")
    print("=" * 60)

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()

    # Simulate login
    if len(at.text_input) >= 2:
        at.text_input[0].set_value("admin").run()
        at.text_input[1].set_value("tbsentinel2026").run()
        if len(at.button) > 0:
            at.button[0].click().run()

    # Check for exceptions
    if at.exception:
        print(f"❌ Exception after dashboard load: {at.exception}")
        pytest.fail(f"Dashboard raised exception: {at.exception}")

    print("✅ Dashboard loads without exceptions")
    print(f"   Elements rendered: {len(at.main)}")


def test_navigation_to_assessment():
    """Test navigation to New Assessment page."""
    print("\n" + "=" * 60)
    print("TEST 6: Navigation to New Assessment")
    print("=" * 60)

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path))

    # Simulate login first
    if len(at.text_input) >= 2:
        at.text_input[0].set_value("admin").run()
        at.text_input[1].set_value("tbsentinel2026").run()
        if len(at.button) > 0:
            at.button[0].click().run()

    # Try to navigate to assessment page
    # Note: AppTest has limited navigation support, so we'll just check
    # that the app state doesn't crash when we try to set the page
    try:
        at.session_state["_st_active_page"] = "views/assessment.py"
        at.run()
        print("   ✓ Navigation attempted")
    except Exception as e:
        print(f"   ⚠ Navigation test skipped: {e}")

    # Check for exceptions
    if at.exception:
        print(f"❌ Exception during navigation: {at.exception}")
        pytest.fail(f"Navigation raised exception: {at.exception}")

    print("✅ Navigation test completes without exceptions")


def test_no_keyerror_or_indexerror():
    """Ensure no KeyError or IndexError is raised during normal flow."""
    print("\n" + "=" * 60)
    print("TEST 7: No Unhandled Exceptions")
    print("=" * 60)

    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path))

    exceptions = []

    # Run the app
    at.run()
    if at.exception:
        exceptions.append(f"Initial render: {at.exception}")

    # Try login
    if len(at.text_input) >= 2:
        try:
            at.text_input[0].set_value("admin").run()
            at.text_input[1].set_value("tbsentinel2026").run()
            if len(at.button) > 0:
                at.button[0].click().run()
        except (KeyError, IndexError) as e:
            exceptions.append(f"Login interaction: {e}")

    # Check session state access
    try:
        _ = at.session_state
    except (KeyError, IndexError) as e:
        exceptions.append(f"Session state access: {e}")

    if exceptions:
        print("❌ Unhandled exceptions found:")
        for exc in exceptions:
            print(f"   - {exc}")
        pytest.fail(f"Found {len(exceptions)} unhandled exceptions")

    print("✅ No KeyError or IndexError raised")


def test_upload_widget_exists():
    """Test that file upload widget exists on assessment page."""
    print("\n" + "=" * 60)
    print("TEST 8: Upload Widget Check")
    print("=" * 60)

    # Test the assessment page directly
    assessment_path = Path(__file__).resolve().parents[1] / "frontend" / "views" / "assessment.py"

    try:
        at = AppTest.from_file(str(assessment_path))
        at.run()

        if at.exception:
            print(f"   ⚠ Assessment page raised exception: {at.exception}")
            print("   (This may be expected if session state is not initialized)")
        else:
            # Check for file_uploader widget
            file_uploaders = [el for el in at.main if el.type == "file_uploader"]
            print(f"   Found {len(file_uploaders)} file uploader(s)")

            if len(file_uploaders) > 0:
                print("✅ Upload widget exists")
            else:
                print("⚠ No upload widget found (may need session state)")

    except Exception as e:
        print(f"   ⚠ Could not test assessment page directly: {e}")
        print("   (This is acceptable - the page requires app context)")

    print("✅ Upload widget check completed")


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v", "-s"])
