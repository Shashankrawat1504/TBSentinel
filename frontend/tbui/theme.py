"""Clinical design system for the TBSentinel dashboard.

Professional enterprise-grade visual language for clinical decision support.
Single source of truth for colors, typography, CSS and UI components.
"""

from __future__ import annotations

import html
import textwrap

# ---------------------------------------------------------------------------
# Design tokens - Professional Clinical Palette
# ---------------------------------------------------------------------------

# Deep navy primary for clinical trustworthiness
PRIMARY = "#0A2647"
PRIMARY_DARK = "#051826"
PRIMARY_SOFT = "#E8F0F7"
PRIMARY_LIGHT = "#144272"

# Clean clinical accent
ACCENT = "#2C74B3"
ACCENT_HOVER = "#205081"

# Professional text colors
INK = "#1E293B"
INK_SOFT = "#334155"
MUTED = "#64748B"
MUTED_LIGHT = "#94A3B8"

# Sterile surface colors
BG = "#F8FAFC"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F1F5F9"

# Clinical status colors
DANGER = "#DC2626"
DANGER_SOFT = "#FEF2F2"
DANGER_BORDER = "#FECACA"

SUCCESS = "#059669"
SUCCESS_SOFT = "#ECFDF5"
SUCCESS_BORDER = "#A7F3D0"

WARNING = "#D97706"
WARNING_SOFT = "#FFFBEB"
WARNING_BORDER = "#FDE68A"

INFO = "#2563EB"
INFO_SOFT = "#EFF6FF"
INFO_BORDER = "#BFDBFE"

# Border and divider
BORDER = "#E2E8F0"
BORDER_LIGHT = "#F1F5F9"

# Rounded corners - professional but approachable
RADIUS_SM = "8px"
RADIUS_MD = "12px"
RADIUS_LG = "16px"
RADIUS_XL = "20px"

# Professional shadows
SHADOW_SM = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
SHADOW_MD = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
SHADOW_LG = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
SHADOW_XL = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"

# Clinical class vocabulary
CLASS_LABELS = {
    "normal": "Normal",
    "tb": "Tuberculosis",
    "pneumonia": "Pneumonia",
    "non_tb": "Non-TB (binary model)",
}
CLASS_ORDER = ("normal", "tb", "pneumonia")
CLASS_TONES = {
    "normal": "success",
    "tb": "error",
    "pneumonia": "warning",
    "non_tb": "success",
}
CLASS_BADGES = {
    "normal": ("✓", "Normal"),
    "tb": ("⚠", "Tuberculosis"),
    "pneumonia": ("▲", "Pneumonia"),
    "non_tb": ("✓", "Non-TB"),
}


def class_kind(class_name: str | None) -> str:
    """Normalise a model class name into a badge/bar kind."""
    name = (class_name or "").strip().casefold()
    return name if name in CLASS_LABELS else "idle"


def class_label(class_name: str | None) -> str:
    """Human-readable label for a model class name."""
    name = (class_name or "").strip()
    return CLASS_LABELS.get(name.casefold(), name or "Unknown")


def class_badge(class_name: str | None) -> tuple[str, str]:
    """Return the ``(symbol, text)`` pair for a result badge."""
    kind = class_kind(class_name)
    if kind == "idle":
        return ("•", class_label(class_name).upper())
    symbol, text = CLASS_BADGES[kind]
    return (symbol, text)


def ordered_classes(class_names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Order a model's class names clinically (Normal → TB → Pneumonia first)."""
    names = [str(name) for name in (class_names or []) if str(name).strip()]
    known = [name for name in CLASS_ORDER if name in names]
    return known + [name for name in names if name not in known]


LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 48 48" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0A2647"/>
      <stop offset="100%" style="stop-color:#2C74B3"/>
    </linearGradient>
  </defs>
  <rect width="48" height="48" rx="12" fill="url(#logoGrad)"/>
  <path d="M24 12c-3.5 2.6-6.9 3.8-9.7 3.8v8.5c0 5.8 4 9.8 9.7 11.7 5.7-1.8 9.7-5.9 9.7-11.7V15.8C30.9 15.8 27.5 14.6 24 12z"
        fill="rgba(255,255,255,0.15)" stroke="rgba(255,255,255,0.4)" stroke-width="1.5"/>
  <path d="M16 23h4l2-4.5 3 8.5 2-4h5"
        stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Professional Stylesheet
# ---------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ts-primary: #0A2647;
    --ts-primary-dark: #051826;
    --ts-primary-soft: #E8F0F7;
    --ts-primary-light: #144272;
    --ts-accent: #2C74B3;
    --ts-accent-hover: #205081;
    --ts-ink: #1E293B;
    --ts-ink-soft: #334155;
    --ts-muted: #64748B;
    --ts-muted-light: #94A3B8;
    --ts-border: #E2E8F0;
    --ts-border-light: #F1F5F9;
    --ts-bg: #F8FAFC;
    --ts-surface: #FFFFFF;
    --ts-surface-alt: #F1F5F9;
    --ts-danger: #DC2626;
    --ts-danger-soft: #FEF2F2;
    --ts-danger-border: #FECACA;
    --ts-success: #059669;
    --ts-success-soft: #ECFDF5;
    --ts-success-border: #A7F3D0;
    --ts-warn: #D97706;
    --ts-warn-soft: #FFFBEB;
    --ts-warn-border: #FDE68A;
    --ts-info: #2563EB;
    --ts-info-soft: #EFF6FF;
    --ts-info-border: #BFDBFE;
    --ts-radius-sm: 8px;
    --ts-radius-md: 12px;
    --ts-radius-lg: 16px;
    --ts-radius-xl: 20px;
    --ts-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --ts-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --ts-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --ts-shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

button, input, textarea, select { font-family: inherit; }

.stApp { background: var(--ts-bg); }
[data-testid="stHeader"] { background: transparent; padding: 0; }
[data-testid="stHeader"] > div { display: none; }
#MainMenu, footer { visibility: hidden; height: 0; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
.block-container { 
    padding-top: 2rem; 
    padding-bottom: 3rem; 
    padding-left: 2rem; 
    padding-right: 2rem;
    max-width: 1400px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { 
    background: #FFFFFF; 
    border-right: 1px solid var(--ts-border);
    box-shadow: var(--ts-shadow-sm);
}
[data-testid="stSidebar"] .block-container { 
    padding-top: 1.5rem; 
    padding-left: 1.25rem; 
    padding-right: 1.25rem;
    padding-bottom: 1rem;
}
[data-testid="stSidebarNav"] { padding-right: 0; }
[data-testid="stSidebarNav"] ul { gap: 4px; }
[data-testid="stSidebarNav"] ul li { margin-left: 0; margin-right: 0; }
[data-testid="stSidebarNav"] ul li a {
    border-radius: var(--ts-radius-md); 
    border: 1px solid transparent;
    padding: 10px 14px; 
    font-weight: 600; 
    font-size: 14px;
    color: var(--ts-ink-soft); 
    transition: all .2s cubic-bezier(0.4, 0, 0.2, 1); 
    min-height: 0;
    letter-spacing: -0.01em;
}
[data-testid="stSidebarNav"] ul li a:hover {
    background: var(--ts-primary-soft); 
    color: var(--ts-primary);
    border-color: var(--ts-border);
    transform: translateX(2px);
}
[data-testid="stSidebarNav"] ul li a[aria-current="page"],
[data-testid="stSidebarNav"] ul li a.active {
    background: linear-gradient(135deg, var(--ts-primary) 0%, var(--ts-accent) 100%);
    border-color: transparent; 
    color: #fff;
    box-shadow: var(--ts-shadow-md);
    transform: translateX(4px);
}
.ts-nav-label {
    color: var(--ts-muted); 
    font-size: 11px; 
    font-weight: 700;
    letter-spacing: 0.1em; 
    text-transform: uppercase;
    margin: 16px 0 8px 4px;
}
.ts-brand-block { 
    display: flex; 
    align-items: center; 
    gap: 12px; 
    margin-bottom: 8px; 
    padding-bottom: 16px;
    border-bottom: 1px solid var(--ts-border-light);
}
.ts-brand-mark {
    width: 44px; 
    height: 44px; 
    flex: 0 0 44px; 
    border-radius: var(--ts-radius-md);
    background: linear-gradient(135deg, var(--ts-primary) 0%, var(--ts-accent) 100%);
    display: flex; 
    align-items: center; 
    justify-content: center;
    box-shadow: var(--ts-shadow-md);
}
.ts-brand-name { 
    font-weight: 800; 
    font-size: 18px; 
    color: var(--ts-ink); 
    line-height: 1.2; 
    letter-spacing: -0.02em;
}
.ts-brand-sub {
    font-size: 11px; 
    color: var(--ts-muted); 
    letter-spacing: 0.05em;
    text-transform: uppercase; 
    font-weight: 600;
}

/* ---------- Topbar ---------- */
.ts-topbar {
    display: flex; 
    align-items: center; 
    justify-content: space-between;
    gap: 20px; 
    flex-wrap: wrap;
    background: var(--ts-surface); 
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg); 
    padding: 16px 24px;
    box-shadow: var(--ts-shadow-sm); 
    margin-bottom: 24px;
}
.ts-topbar-title { 
    font-size: 20px; 
    font-weight: 800; 
    color: var(--ts-ink); 
    letter-spacing: -0.02em; 
    margin: 0; 
}
.ts-topbar-sub { 
    font-size: 13px; 
    color: var(--ts-muted); 
    margin: 2px 0 0 0; 
    font-weight: 500;
}
.ts-topbar-right { 
    display: flex; 
    align-items: center; 
    gap: 10px; 
    flex-wrap: wrap; 
}

/* ---------- Hero ---------- */
.ts-hero {
    display: flex; 
    align-items: center; 
    justify-content: space-between;
    gap: 24px; 
    flex-wrap: wrap;
    background: linear-gradient(135deg, var(--ts-primary) 0%, var(--ts-accent) 100%);
    border-radius: var(--ts-radius-xl); 
    padding: 32px 36px;
    box-shadow: var(--ts-shadow-lg); 
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.ts-hero::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.ts-hero-left { 
    display: flex; 
    align-items: center; 
    gap: 20px; 
    position: relative;
    z-index: 1;
}
.ts-logo {
    width: 64px; 
    height: 64px; 
    flex: 0 0 64px; 
    border-radius: var(--ts-radius-lg);
    background: rgba(255, 255, 255, .12);
    display: flex; 
    align-items: center; 
    justify-content: center;
    border: 1px solid rgba(255, 255, 255, .2);
    backdrop-filter: blur(10px);
}
.ts-brand { 
    color: #fff; 
    font-size: 32px; 
    font-weight: 800; 
    letter-spacing: -0.03em; 
    line-height: 1.1; 
    margin: 0; 
}
.ts-tagline { 
    color: rgba(255, 255, 255, .85); 
    font-size: 14px; 
    font-weight: 500; 
    margin-top: 6px;
    line-height: 1.5;
}
.ts-hero-right { 
    display: flex; 
    align-items: center; 
    gap: 10px; 
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.ts-chip {
    background: rgba(255, 255, 255, .15); 
    color: #fff;
    border: 1px solid rgba(255, 255, 255, .25);
    padding: 8px 16px; 
    border-radius: 999px;
    font-size: 12px; 
    font-weight: 600; 
    letter-spacing: 0.05em;
    text-transform: uppercase; 
    white-space: nowrap;
    backdrop-filter: blur(10px);
}

/* ---------- Cards ---------- */
.ts-card-title {
    display: flex; 
    align-items: center; 
    gap: 10px;
    font-size: 16px; 
    font-weight: 700; 
    color: var(--ts-ink);
    margin: 0 0 6px 0; 
    letter-spacing: -0.01em;
}
.ts-card-sub { 
    color: var(--ts-muted); 
    font-size: 13px; 
    margin: 0 0 16px 0; 
    line-height: 1.5;
}
.ts-eyebrow {
    color: var(--ts-primary); 
    font-size: 11px; 
    font-weight: 700;
    letter-spacing: 0.1em; 
    text-transform: uppercase; 
    margin: 0 0 8px 0;
}
.ts-divider { 
    height: 1px; 
    background: var(--ts-border); 
    margin: 20px 0; 
    border: 0; 
}

/* Targeted card styling using component keys */
[class*="st-key-card_"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border) !important;
    border-radius: var(--ts-radius-lg) !important;
    padding: 24px !important;
    box-shadow: var(--ts-shadow-sm);
    transition: all .25s cubic-bezier(0.4, 0, 0.2, 1);
}
[class*="st-key-card_"]:hover {
    box-shadow: var(--ts-shadow-lg);
    border-color: var(--ts-muted-light) !important;
    transform: translateY(-2px);
}
[class*="st-key-card_"] [data-testid="stImage"] img { 
    width: 100%; 
    border-radius: var(--ts-radius-md);
}

/* ---------- KPI / stat cards ---------- */
.ts-stat {
    background: var(--ts-surface); 
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg); 
    padding: 20px 24px;
    box-shadow: var(--ts-shadow-sm); 
    height: 100%;
    transition: all .25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.ts-stat::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--ts-primary), var(--ts-accent));
    opacity: 0;
    transition: opacity .25s ease;
}
.ts-stat:hover { 
    box-shadow: var(--ts-shadow-lg);
    transform: translateY(-3px);
}
.ts-stat:hover::before {
    opacity: 1;
}
.ts-stat-head { 
    display: flex; 
    align-items: center; 
    justify-content: space-between; 
    gap: 12px; 
}
.ts-stat-label {
    font-size: 11px; 
    font-weight: 700; 
    letter-spacing: 0.08em;
    text-transform: uppercase; 
    color: var(--ts-muted);
}
.ts-stat-icon {
    width: 40px; 
    height: 40px; 
    border-radius: var(--ts-radius-md); 
    font-size: 18px;
    display: flex; 
    align-items: center; 
    justify-content: center;
    background: var(--ts-primary-soft);
    color: var(--ts-primary);
}
.ts-stat-value {
    font-size: 32px; 
    font-weight: 800; 
    color: var(--ts-ink);
    letter-spacing: -0.02em; 
    line-height: 1; 
    margin-top: 12px;
}
.ts-stat-foot { 
    font-size: 12px; 
    color: var(--ts-muted); 
    margin-top: 6px; 
    font-weight: 500;
}
.ts-stat--danger .ts-stat-icon { 
    background: var(--ts-danger-soft); 
    color: var(--ts-danger);
}
.ts-stat--danger::before {
    background: linear-gradient(90deg, var(--ts-danger), #EF4444);
}
.ts-stat--success .ts-stat-icon { 
    background: var(--ts-success-soft); 
    color: var(--ts-success);
}
.ts-stat--success::before {
    background: linear-gradient(90deg, var(--ts-success), #10B981);
}
.ts-stat--warn .ts-stat-icon { 
    background: var(--ts-warn-soft); 
    color: var(--ts-warn);
}
.ts-stat--warn::before {
    background: linear-gradient(90deg, var(--ts-warn), #F59E0B);
}

/* ---------- Result badge & confidence ---------- */
.ts-result {
    border-radius: var(--ts-radius-lg); 
    padding: 24px; 
    border: 1px solid var(--ts-border);
    background: #FAFAFA;
}
.ts-result--tb { 
    background: var(--ts-danger-soft); 
    border-color: var(--ts-danger-border); 
    border-left: 5px solid var(--ts-danger); 
}
.ts-result--non_tb, .ts-result--normal { 
    background: var(--ts-success-soft); 
    border-color: var(--ts-success-border); 
    border-left: 5px solid var(--ts-success); 
}
.ts-result--pneumonia { 
    background: var(--ts-warn-soft); 
    border-color: var(--ts-warn-border); 
    border-left: 5px solid var(--ts-warn); 
}
.ts-badge {
    display: inline-flex; 
    align-items: center; 
    gap: 8px;
    padding: 10px 18px; 
    border-radius: 999px;
    font-size: 14px; 
    font-weight: 700; 
    letter-spacing: 0.02em;
}
.ts-badge--tb { 
    background: var(--ts-danger-soft); 
    color: var(--ts-danger); 
    border: 1px solid var(--ts-danger-border); 
}
.ts-badge--non_tb, .ts-badge--normal { 
    background: var(--ts-success-soft); 
    color: var(--ts-success); 
    border: 1px solid var(--ts-success-border); 
}
.ts-badge--pneumonia { 
    background: var(--ts-warn-soft); 
    color: var(--ts-warn); 
    border: 1px solid var(--ts-warn-border); 
}
.ts-badge--idle { 
    background: var(--ts-surface-alt); 
    color: var(--ts-muted); 
    border: 1px solid var(--ts-border); 
}
.ts-conf-value { 
    font-size: 44px; 
    font-weight: 800; 
    letter-spacing: -0.03em; 
    color: var(--ts-ink); 
    line-height: 1; 
    margin: 18px 0 4px 0; 
}
.ts-conf-label { 
    font-size: 12px; 
    font-weight: 600; 
    letter-spacing: 0.08em; 
    text-transform: uppercase; 
    color: var(--ts-muted); 
}
.ts-bar { 
    height: 12px; 
    border-radius: 999px; 
    background: var(--ts-surface-alt); 
    overflow: hidden; 
    margin-top: 12px; 
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
}
.ts-bar-fill { 
    height: 100%; 
    border-radius: 999px; 
    transition: width .6s cubic-bezier(0.4, 0, 0.2, 1); 
}
.ts-bar-fill--tb { 
    background: linear-gradient(90deg, #F87171, var(--ts-danger)); 
}
.ts-bar-fill--non_tb, .ts-bar-fill--normal { 
    background: linear-gradient(90deg, #34D399, var(--ts-success)); 
}
.ts-bar-fill--pneumonia { 
    background: linear-gradient(90deg, #FBBF24, var(--ts-warn)); 
}
.ts-bar-fill--primary { 
    background: linear-gradient(90deg, #60A5FA, var(--ts-primary)); 
}
.ts-bar-fill--accent { 
    background: linear-gradient(90deg, #2DD4BF, var(--ts-accent)); 
}
.ts-prob-row { margin-top: 16px; }
.ts-prob-head {
    display: flex; 
    justify-content: space-between;
    font-size: 13px; 
    font-weight: 600; 
    color: var(--ts-ink); 
    margin-bottom: 6px;
}
.ts-prob-head span:last-child { 
    font-variant-numeric: tabular-nums; 
    color: var(--ts-muted); 
}

/* ---------- Alerts ---------- */
.ts-alert {
    border-radius: var(--ts-radius-md); 
    padding: 16px 20px; 
    margin: 8px 0 16px 0;
    font-size: 14px; 
    line-height: 1.6; 
    border: 1px solid transparent;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.ts-alert-title { 
    font-weight: 700; 
    margin-bottom: 4px; 
    display: flex; 
    gap: 8px; 
    align-items: center;
    font-size: 14px;
}
.ts-alert--error { 
    background: var(--ts-danger-soft); 
    border-color: var(--ts-danger-border); 
    color: #991B1B; 
}
.ts-alert--warning { 
    background: var(--ts-warn-soft); 
    border-color: var(--ts-warn-border); 
    color: #92400E; 
}
.ts-alert--info { 
    background: var(--ts-info-soft); 
    border-color: var(--ts-info-border); 
    color: #1E40AF; 
}
.ts-alert--success { 
    background: var(--ts-success-soft); 
    border-color: var(--ts-success-border); 
    color: #065F46; 
}
.ts-alert code {
    background: rgba(0, 0, 0, .06); 
    padding: 2px 8px; 
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; 
    font-size: 13px;
}

/* ---------- Status pills ---------- */
.ts-pill {
    display: inline-flex; 
    align-items: center; 
    gap: 8px;
    border-radius: 999px; 
    padding: 8px 14px; 
    font-size: 13px; 
    font-weight: 600;
}
.ts-pill--ok { 
    background: var(--ts-success-soft); 
    color: var(--ts-success); 
    border: 1px solid var(--ts-success-border); 
}
.ts-pill--down { 
    background: var(--ts-danger-soft); 
    color: var(--ts-danger); 
    border: 1px solid var(--ts-danger-border); 
}
.ts-pill--neutral { 
    background: var(--ts-surface-alt); 
    color: var(--ts-muted); 
    border: 1px solid var(--ts-border); 
}
.ts-dot { 
    width: 8px; 
    height: 8px; 
    border-radius: 50%; 
    display: inline-block; 
}
.ts-dot--ok { 
    background: var(--ts-success); 
    box-shadow: 0 0 0 3px rgba(5, 150, 105, .2); 
}
.ts-dot--down { 
    background: var(--ts-danger); 
    box-shadow: 0 0 0 3px rgba(220, 38, 38, .2); 
}

/* ---------- Key/value lists ---------- */
.ts-kv { 
    display: flex; 
    justify-content: space-between; 
    gap: 12px; 
    font-size: 13px; 
    padding: 10px 0; 
    border-bottom: 1px dashed var(--ts-border-light); 
}
.ts-kv:last-child { border-bottom: 0; }
.ts-kv-key { 
    color: var(--ts-muted); 
    font-weight: 500; 
}
.ts-kv-value { 
    color: var(--ts-ink); 
    font-weight: 600; 
    text-align: right; 
    word-break: break-word; 
}

/* ---------- Metric comparison bars ---------- */
.ts-cmp-row { margin: 12px 0 16px 0; }
.ts-cmp-head { 
    display: flex; 
    justify-content: space-between; 
    font-size: 13px; 
    font-weight: 600; 
    color: var(--ts-ink); 
    margin-bottom: 6px; 
}
.ts-cmp-head span:last-child { 
    color: var(--ts-muted); 
    font-variant-numeric: tabular-nums; 
}
.ts-cmp-track { 
    height: 10px; 
    border-radius: 999px; 
    background: var(--ts-surface-alt); 
    overflow: hidden; 
    position: relative; 
}
.ts-cmp-fill { 
    height: 100%; 
    border-radius: 999px; 
}
.ts-cmp-fill--cen { 
    background: linear-gradient(90deg, #60A5FA, var(--ts-primary)); 
}
.ts-cmp-fill--fed { 
    background: linear-gradient(90deg, #2DD4BF, var(--ts-accent)); 
}
.ts-cmp-fill--fedp { 
    background: linear-gradient(90deg, #94A3B8, var(--ts-muted)); 
}
.ts-cmp-fill--normal { 
    background: linear-gradient(90deg, #34D399, var(--ts-success)); 
}
.ts-cmp-fill--tb { 
    background: linear-gradient(90deg, #F87171, var(--ts-danger)); 
}
.ts-cmp-fill--pneumonia { 
    background: linear-gradient(90deg, #FBBF24, var(--ts-warn)); 
}
.ts-delta { 
    font-size: 12px; 
    font-weight: 700; 
    font-variant-numeric: tabular-nums; 
}
.ts-delta--down { color: var(--ts-danger); }
.ts-delta--up { color: var(--ts-success); }

/* ---------- Tables ---------- */
.ts-table { 
    width: 100%; 
    border-collapse: separate; 
    border-spacing: 0; 
    font-size: 13px; 
}
.ts-table thead th {
    text-align: left; 
    color: var(--ts-muted); 
    font-size: 11px; 
    font-weight: 700;
    letter-spacing: 0.05em; 
    text-transform: uppercase; 
    padding: 12px 16px;
    border-bottom: 2px solid var(--ts-border); 
    background: var(--ts-surface-alt);
}
.ts-table tbody td { 
    padding: 14px 16px; 
    border-bottom: 1px solid var(--ts-border-light); 
    color: var(--ts-ink); 
}
.ts-table tbody tr:last-child td { border-bottom: 0; }
.ts-table tbody tr:hover td { 
    background: var(--ts-surface-alt); 
}
.ts-table .ts-num { 
    text-align: right; 
    font-variant-numeric: tabular-nums; 
    font-weight: 600; 
}
.ts-table thead th.ts-num { text-align: right; }

/* ---------- Footer ---------- */
.ts-footer {
    text-align: center; 
    color: var(--ts-muted); 
    font-size: 12px;
    padding: 24px 16px 8px 16px; 
    line-height: 1.7;
}
.ts-footer strong { color: var(--ts-ink); }

/* ---------- Widget polish ---------- */
.stButton > button {
    border-radius: var(--ts-radius-md); 
    font-weight: 600; 
    padding: 12px 24px;
    border: 1px solid var(--ts-primary); 
    transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 14px;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--ts-primary) 0%, var(--ts-accent) 100%);
    color: #fff; 
    border: 0; 
    box-shadow: var(--ts-shadow-md);
}
.stButton > button[kind="primary"]:hover { 
    filter: brightness(1.05); 
    transform: translateY(-1px);
    box-shadow: var(--ts-shadow-lg);
}
.stButton > button[kind="secondary"] { 
    background: #fff; 
    color: var(--ts-primary); 
}
.stButton > button[kind="secondary"]:hover { 
    background: var(--ts-primary-soft); 
}
.stButton > button:disabled {
    border-color: var(--ts-border); 
    background: var(--ts-surface-alt); 
    color: var(--ts-muted-light);
    box-shadow: none; 
    transform: none; 
    filter: none;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--ts-surface); 
    border: 2px dashed var(--ts-border); 
    border-radius: var(--ts-radius-lg);
    transition: all .2s ease;
    padding: 32px;
}
[data-testid="stFileUploaderDropzone"]:hover { 
    border-color: var(--ts-accent); 
    background: var(--ts-primary-soft); 
}

div[role="tablist"] { 
    border-bottom: 2px solid var(--ts-border); 
    gap: 6px; 
}
button[role="tab"] { 
    font-weight: 600; 
    font-size: 14px; 
    color: var(--ts-muted); 
    padding: 12px 20px;
}
button[role="tab"][aria-selected="true"] { 
    color: var(--ts-primary); 
    border-bottom: 2px solid var(--ts-primary);
}

[data-testid="stImage"] img {
    border-radius: var(--ts-radius-md); 
    border: 1px solid var(--ts-border); 
    background: #fff;
    box-shadow: var(--ts-shadow-sm);
}
[data-testid="stMetricValue"] { 
    font-size: 28px; 
    font-weight: 800; 
    color: var(--ts-ink);
}
[data-testid="stDataFrame"] { 
    border-radius: var(--ts-radius-lg); 
    overflow: hidden; 
    border: 1px solid var(--ts-border); 
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Input fields ---------- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    border-radius: var(--ts-radius-md);
    border: 1px solid var(--ts-border);
    padding: 10px 14px;
    font-size: 14px;
    transition: all .2s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div > select:focus {
    border-color: var(--ts-primary);
    box-shadow: 0 0 0 3px rgba(10, 38, 71, 0.1);
}

/* ---------- Grad-CAM display ---------- */
[class*="st-key-gradcam_"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 20px;
    box-shadow: var(--ts-shadow-sm);
}
[class*="st-key-gradcam_"] [data-testid="stImage"] {
    border-radius: var(--ts-radius-md);
    overflow: hidden;
}
[class*="st-key-gradcam_"] [data-testid="stImage"] img {
    border: none;
    border-radius: var(--ts-radius-md);
}

/* ---------- Enhanced metrics panel styling ---------- */
[class*="st-key-metrics_"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Patient form card styling ---------- */
[class*="st-key-card_patient"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Upload card styling ---------- */
[class*="st-key-card_upload"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Result card styling ---------- */
[class*="st-key-card_result"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Explainability card styling ---------- */
[class*="st-key-card_explain"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Clinical card styling ---------- */
[class*="st-key-card_clinical"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Save card styling ---------- */
[class*="st-key-card_save"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Recent records card styling ---------- */
[class*="st-key-card_recent"] {
    background: var(--ts-surface);
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-lg);
    padding: 24px;
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Status widget polishing ---------- */
[data-testid="stStatusWidget"] {
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-md);
    background: var(--ts-surface);
    box-shadow: var(--ts-shadow-sm);
}

/* ---------- Expander styling ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--ts-border);
    border-radius: var(--ts-radius-md);
    background: var(--ts-surface);
    box-shadow: var(--ts-shadow-sm);
}
[data-testid="stExpander"] > div > div > button {
    font-weight: 600;
    color: var(--ts-ink);
}

/* ---------- Divider styling ---------- */
hr .ts-divider {
    border: 0;
    height: 1px;
    background: var(--ts-border);
    margin: 20px 0;
}

/* ---------- Caption and help text ---------- */
.stCaption {
    color: var(--ts-muted);
    font-size: 12px;
    font-weight: 500;
}

/* ---------- Number input styling ---------- */
.stNumberInput > div > div > input {
    border-radius: var(--ts-radius-md);
    border: 1px solid var(--ts-border);
    padding: 10px 14px;
    font-size: 14px;
    transition: all .2s ease;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--ts-primary);
    box-shadow: 0 0 0 3px rgba(10, 38, 71, 0.1);
}

/* ---------- Checkbox styling ---------- */
.stCheckbox > label {
    font-weight: 500;
    color: var(--ts-ink);
}
.stCheckbox > label > div {
    border: 2px solid var(--ts-border);
    border-radius: 6px;
}

/* ---------- Toast notifications ---------- */
[data-testid="stToast"] {
    border-radius: var(--ts-radius-md);
    box-shadow: var(--ts-shadow-lg);
}

/* ---------- Session state improvements ---------- */
.stDownloadButton > button {
    border-radius: var(--ts-radius-md);
    font-weight: 600;
    padding: 10px 20px;
    border: 1px solid var(--ts-border);
    transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 13px;
}
.stDownloadButton > button:hover {
    background: var(--ts-primary-soft);
    border-color: var(--ts-accent);
}
</style>
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_html(markup: str) -> None:
    """Render a de-indented HTML block (indented text would be shown as code)."""
    import streamlit as st

    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def inject_styles() -> None:
    """Load the clinical theme stylesheet."""
    render_html(STYLES)


def alert(kind: str, title: str, body: str = "") -> None:
    """Render a styled alert card (kind: error, warning, info, success)."""
    icons = {"error": "⛔", "warning": "⚠️", "info": "ℹ️", "success": "✅"}
    icon = icons.get(kind, "ℹ️")
    render_html(
        f"""
        <div class="ts-alert ts-alert--{kind}">
            <div class="ts-alert-title"><span>{icon}</span><span>{html.escape(title)}</span></div>
            <div>{body}</div>
        </div>
        """
    )


def kv_row(key: str, value: str) -> str:
    """Single key/value row for the structured list."""
    return f'<div class="ts-kv"><span class="ts-kv-key">{html.escape(key)}</span><span class="ts-kv-value">{html.escape(value)}</span></div>'


def kv_list(pairs: list[tuple[str, str]]) -> None:
    """Render a structured key/value list."""
    render_html("".join(kv_row(k, v) for k, v in pairs))


def pill(status: str, text: str) -> str:
    """Status pill component."""
    dot = "ok" if status == "ok" else "down" if status == "down" else "neutral"
    pill_class = f"ts-pill--{status}"
    dot_class = f"ts-dot--{dot}"
    return f'<span class="ts-pill {pill_class}"><span class="ts-dot {dot_class}"></span>{html.escape(text)}</span>'


def stat_card(
    label: str,
    value: str,
    icon: str,
    foot: str = "",
    tone: str = "",
) -> str:
    """KPI stat card HTML."""
    tone_class = f"ts-stat--{tone}" if tone else ""
    foot_html = f'<div class="ts-stat-foot">{html.escape(foot)}</div>' if foot else ""
    return f"""
    <div class="ts-stat {tone_class}">
        <div class="ts-stat-head">
            <span class="ts-stat-label">{html.escape(label)}</span>
            <span class="ts-stat-icon">{icon}</span>
        </div>
        <div class="ts-stat-value">{html.escape(value)}</div>
        {foot_html}
    </div>
    """


def prob_row(label: str, probability: float, kind: str) -> str:
    """Probability bar row for class distribution."""
    bar_class = f"ts-bar-fill--{kind}"
    return f"""
    <div class="ts-prob-row">
        <div class="ts-prob-head">
            <span>{html.escape(label)}</span>
            <span>{probability:.1%}</span>
        </div>
        <div class="ts-bar">
            <div class="ts-bar-fill {bar_class}" style="width:{probability * 100:.1f}%"></div>
        </div>
    </div>
    """


def level_chip(level: str, tone: str) -> str:
    """Confidence level chip."""
    tone_classes = {
        "success": "ts-pill--ok",
        "error": "ts-pill--down",
        "warning": "ts-pill--neutral",
        "info": "ts-pill--neutral",
    }
    pill_class = tone_classes.get(tone, "ts-pill--neutral")
    return f'<span class="ts-pill {pill_class}">{html.escape(level)}</span>'


def badge(class_name: str, compact: bool = False) -> str:
    """Class badge component."""
    kind = class_kind(class_name)
    symbol, text = class_badge(class_name)
    badge_class = f"ts-badge--{kind}"
    if compact:
        return f'<span class="ts-badge {badge_class}" style="padding:6px 12px;font-size:12px;">{symbol} {html.escape(text)}</span>'
    return f'<span class="ts-badge {badge_class}">{symbol} {html.escape(text)}</span>'


def render_footer() -> None:
    """Page footer with the standing research disclaimer."""
    render_html(
        """
        <div class="ts-footer">
            <strong>Research Prototype – Not for Clinical Diagnosis</strong><br/>
            TBSentinel v1.0.0 · Federated Multimodal CDSS · EfficientNet-B3 + Grad-CAM ·
            Built with Streamlit, FastAPI and Flower
        </div>
        """
    )
