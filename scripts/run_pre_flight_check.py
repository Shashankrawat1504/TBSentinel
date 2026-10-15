#!/usr/bin/env python3
"""
Pre-flight health check for TBSentinel presentation.

This script verifies:
1. FastAPI backend is healthy (/health endpoint)
2. Prediction endpoint works (/predict)
3. Explanation endpoint works (/explain)
4. SQLite database is accessible and writable

Run this before your presentation to ensure everything is working.
"""

from __future__ import annotations

import io
import sqlite3
import sys
from pathlib import Path

import requests
from PIL import Image

# Configuration
API_URL = "http://127.0.0.1:8000"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_records" / "assessments.db"


def create_dummy_image() -> bytes:
    """Create a simple dummy PNG image for testing."""
    # Create a 300x300 white image
    img = Image.new("RGB", (300, 300), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def test_health_endpoint() -> bool:
    """Test the /health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed (HTTP {response.status_code})")
            print(f"   API online: {data.get('api_online', 'unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', 'unknown')}")
            print(f"   Device: {data.get('device', 'unknown')}")
            print(f"   Checkpoint: {data.get('checkpoint_name', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed (HTTP {response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {API_URL}")
        print("   Make sure the FastAPI server is running:")
        print("   python -m src.inference.main")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_predict_endpoint() -> bool:
    """Test the /predict endpoint with a dummy image."""
    print("\n" + "=" * 60)
    print("TEST 2: Prediction Endpoint")
    print("=" * 60)
    try:
        image_bytes = create_dummy_image()
        files = {"file": ("test_dummy.png", image_bytes, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction endpoint passed (HTTP {response.status_code})")
            print(f"   Predicted class: {data.get('predicted_class_name', 'unknown')}")
            print(f"   Confidence: {data.get('confidence', 0.0):.2%}")
            print(f"   Classes found: {list(data.get('probabilities', {}).keys())}")
            return True
        else:
            print(f"❌ Prediction endpoint failed (HTTP {response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
        return False


def test_explain_endpoint() -> bool:
    """Test the /explain endpoint with a dummy image."""
    print("\n" + "=" * 60)
    print("TEST 3: Explanation Endpoint")
    print("=" * 60)
    try:
        image_bytes = create_dummy_image()
        files = {"file": ("test_dummy.png", image_bytes, "image/png")}
        response = requests.post(f"{API_URL}/explain", files=files, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Explanation endpoint passed (HTTP {response.status_code})")
            print(f"   Has heatmap: {'heatmap_base64' in data}")
            print(f"   Has overlay: {'overlay_base64' in data}")
            print(f"   Has focus summary: {'focus' in data and data.get('focus', {}).get('summary')}")
            return True
        else:
            print(f"❌ Explanation endpoint failed (HTTP {response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Explanation endpoint error: {e}")
        return False


def test_database_access() -> bool:
    """Test SQLite database read/write access."""
    print("\n" + "=" * 60)
    print("TEST 4: Database Access")
    print("=" * 60)

    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  Database does not exist, will create: {DB_PATH}")

    try:
        # Test read connection
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Check if assessments table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='assessments'"
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            print("   Creating assessments table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    patient_id TEXT,
                    patient_name TEXT,
                    age INTEGER,
                    gender TEXT,
                    notes TEXT,
                    filename TEXT,
                    predicted_class TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            print("   ✅ Table created successfully")

        # Test write
        cursor.execute(
            "INSERT OR IGNORE INTO assessments (record_id, patient_id, patient_name, predicted_class, confidence) "
            "VALUES ('TEST-PREFLIGHT-001', 'TEST-001', 'Test Patient', 'normal', 0.95)"
        )
        conn.commit()
        print("   ✅ Write test passed")

        # Test read
        cursor.execute("SELECT COUNT(*) FROM assessments WHERE record_id = 'TEST-PREFLIGHT-001'")
        count = cursor.fetchone()[0]
        print(f"   ✅ Read test passed (found {count} test record(s))")

        # Cleanup test record
        cursor.execute("DELETE FROM assessments WHERE record_id = 'TEST-PREFLIGHT-001'")
        conn.commit()
        print("   ✅ Cleanup test passed")

        conn.close()
        print(f"✅ Database access test passed: {DB_PATH}")
        return True

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Database access error: {e}")
        return False


def main() -> int:
    """Run all pre-flight checks."""
    print("\n" + "=" * 60)
    print("TBSentinel Pre-Flight Health Check")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Database: {DB_PATH}")

    results = {
        "Health Endpoint": test_health_endpoint(),
        "Prediction Endpoint": test_predict_endpoint(),
        "Explanation Endpoint": test_explain_endpoint(),
        "Database Access": test_database_access(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 All checks passed! System is ready for presentation.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
