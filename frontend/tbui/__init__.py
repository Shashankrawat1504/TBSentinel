"""Shared presentation, API and storage layer for the TBSentinel dashboard.

Modules
-------
theme      Clinical design system (CSS + HTML building blocks).
api        FastAPI backend client (health, predict, explain).
storage    Local SQLite persistence for saved patient assessments.
metrics    Loads experiment metrics from ``artifacts/outputs``.
components Reusable Streamlit widgets (sidebar, result cards, ...).
"""

__all__ = ["api", "components", "metrics", "storage", "theme"]
