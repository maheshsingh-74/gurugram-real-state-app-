"""
Design system and shared branding for EstateIQ Gurugram — Real Estate Intelligence Portal.

Modern dark-mode portal aesthetics inspired by leading market real estate platforms:
- Canvas: Deep luxury midnight canvas (#0b0f19)
- Cards & Surfaces: Sleek dark slate surfaces (#131d31) with crisp borders (#1e293b)
- Brand Primary: Vibrant Royal Blue & Azure (#2563eb / #38bdf8)
- Headings & Text: High-contrast crisp white (#ffffff / #f8fafc) and muted slate (#94a3b8)
- Badges: Translucent neon-accent pills with glowing borders

Exported helpers
────────────────
  apply_theme()               → CSS + sidebar (call once per page)
  render_hero(title, tagline)  → Portal hero section with background image
  render_gradient_divider()    → Styled separator
  render_footer()              → Portal footer
  render_kpi_cards(kpis)       → Row of metric stat cards
  render_prediction_result()   → Valuation panel (without R2/MAE)
  render_rec_card()            → Property listing card
  render_feature_cards()       → Feature showcase grid
  col_label(name)              → Human-readable column label
  PLOTLY_LAYOUT                → Dict to spread into fig.update_layout()
"""

import base64
import os

import streamlit as st

# ── Human-readable column labels ──────────────────────────────────────────

COLUMN_LABELS = {
    "bedRoom": "🛏️ Bedrooms",
    "bathroom": "🚿 Bathrooms",
    "built_up_area": "📐 Built-up Area (sq ft)",
    "servant room": "🏠 Servant Room",
    "store room": "📦 Store Room",
    "property_type": "🏢 Property Type",
    "sector": "📍 Sector",
    "furnishing_type": "🪑 Furnishing",
    "balcony": "🌿 Balconies",
    "agePossession": "📅 Possession Status",
    "luxury_category": "✨ Luxury Tier",
    "floor_category": "🏗️ Floor Level",
}


def col_label(name):
    """Get human-readable label for a column, falling back to title case."""
    return COLUMN_LABELS.get(name, name.replace("_", " ").title())


# ── Plotly portal dark template ───────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="'Inter', 'Open Sans', sans-serif", size=13),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", tickfont=dict(color="#94a3b8")),
    colorway=["#38bdf8", "#2563eb", "#f59e0b", "#10b981", "#8b5cf6",
              "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#ec4899"],
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#131d31", font_color="#ffffff", bordercolor="#38bdf8"),
)


# ── SVGs ──────────────────────────────────────────────────────────────────

ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="38" height="38">
  <defs>
    <linearGradient id="eqDarkGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#38bdf8"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="96" height="96" rx="16" fill="url(#eqDarkGrad)"/>
  <path d="M48 20 L76 42 L68 42 L68 76 L28 76 L28 42 L20 42 Z" fill="#ffffff" opacity="0.95"/>
  <rect x="42" y="52" width="12" height="24" rx="2" fill="#0f172a"/>
  <circle cx="58" cy="38" r="4" fill="#38bdf8"/>
</svg>
"""

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 84" width="290" height="68">
  <defs>
    <linearGradient id="logoDarkGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1e40af"/>
      <stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>
  </defs>
  <rect x="0" y="2" width="80" height="80" rx="14" fill="url(#logoDarkGrad)"/>
  <path d="M40 18 L66 38 L58 38 L58 68 L22 68 L22 38 L14 38 Z" fill="#ffffff"/>
  <rect x="34" y="46" width="12" height="22" rx="2" fill="#0f172a"/>
  <circle cx="50" cy="32" r="3.5" fill="#38bdf8"/>
  <text x="96" y="46" font-family="'Inter', 'Open Sans', sans-serif"
        font-size="29" font-weight="800" fill="#ffffff" letter-spacing="-0.5">EstateIQ</text>
  <text x="97" y="66" font-family="'Inter', sans-serif"
        font-size="10" font-weight="700" letter-spacing="2.2" fill="#38bdf8">REAL ESTATE INTELLIGENCE</text>
</svg>
"""


# ── Image helpers ─────────────────────────────────────────────────────────

def _image_to_base64(path):
    """Read an image file and return its base64-encoded data URI."""
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{data}"


# ── CSS design system (Dark Portal Edition — v2 with image heroes) ────────

DESIGN_CSS = """
/* ─── BASE STYLING (DARK PORTAL) ─── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Open Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Inter', 'Open Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.015em;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b0f19; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ─── SIDEBAR (DARK PORTAL) ─── */
[data-testid="stSidebar"] {
    background: #0d1322 !important;
    border-right: 1px solid #1e293b !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebarNav"] a {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 0.9rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(37, 99, 255, 0.12) !important;
    color: #38bdf8 !important;
}
[data-testid="stSidebarNav"] li { margin-bottom: 0.2rem; }

.sidebar-brand {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.8rem 0; margin-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}
.sidebar-brand-name {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800; font-size: 1.25rem;
    color: #ffffff;
}
.sidebar-caption {
    color: #38bdf8; font-size: 0.72rem; margin-top: -0.1rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
}

/* ─── HERO IMAGE BANNER (99acres-style) ─── */
.hero-image-wrap {
    position: relative;
    width: 100%;
    min-height: 340px;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1.8rem;
    background-size: cover;
    background-position: center 40%;
    background-repeat: no-repeat;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
}
.hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        180deg,
        rgba(11, 15, 25, 0.35) 0%,
        rgba(11, 15, 25, 0.55) 40%,
        rgba(11, 15, 25, 0.92) 100%
    );
    z-index: 1;
}
.hero-content {
    position: relative;
    z-index: 2;
    padding: 2.8rem 2.4rem 2.2rem 2.4rem;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    min-height: 340px;
}
.hero-badge-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #38bdf8; padding: 0.35rem 1rem;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600;
    margin-bottom: 0.8rem;
    width: fit-content;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
.hero-page-title {
    font-size: 2.6rem; font-weight: 800;
    color: #ffffff !important; margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
    line-height: 1.15;
}
.hero-tagline {
    color: #e2e8f0; font-size: 1.08rem;
    margin: 0; max-width: 680px; line-height: 1.6;
    text-shadow: 0 1px 6px rgba(0, 0, 0, 0.4);
}
.hero-stats-row {
    display: flex; gap: 2rem; margin-top: 1.2rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex; flex-direction: column;
    padding: 0.6rem 1rem;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    min-width: 120px;
}
.hero-stat-value {
    font-size: 1.4rem; font-weight: 800; color: #ffffff;
    line-height: 1.2;
}
.hero-stat-label {
    font-size: 0.75rem; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-top: 0.15rem;
}

/* ─── FALLBACK HERO (no image) ─── */
.hero-portal-wrap {
    background: linear-gradient(135deg, #091328 0%, #0f2756 50%, #1a4494 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 2.2rem 1.8rem 2rem 1.8rem;
    margin-bottom: 1.8rem;
    color: #ffffff;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
}
.hero-portal-wrap::after {
    content: '';
    position: absolute; right: -40px; bottom: -40px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}

/* ─── NAVIGATION TABS BAR ─── */
.search-tabs-bar {
    display: flex; gap: 0.5rem; margin-bottom: 1.2rem;
    border-bottom: 2px solid #1e293b; padding-bottom: 0.5rem;
}
.search-tab-item {
    font-size: 0.92rem; font-weight: 700; color: #94a3b8;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid transparent;
    padding: 0.45rem 1rem; border-radius: 6px; cursor: default;
    transition: all 0.2s ease;
}
.search-tab-item:hover {
    color: #f8fafc; background: rgba(255, 255, 255, 0.06);
}
.search-tab-item.active {
    color: #38bdf8; background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* ─── FEATURE SHOWCASE CARDS ─── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}
@media (max-width: 768px) {
    .feature-grid { grid-template-columns: 1fr; }
    .hero-page-title { font-size: 1.8rem !important; }
    .hero-content { padding: 1.5rem 1.2rem 1.5rem 1.2rem !important; min-height: 260px !important; }
    .hero-image-wrap { min-height: 260px !important; }
}
@media (max-width: 1024px) and (min-width: 769px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}
.feature-card {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    background: #131d31;
    border: 1px solid #1e293b;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 48px rgba(37, 99, 235, 0.25);
    border-color: rgba(56, 189, 248, 0.4);
}
.feature-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    cursor: pointer;
    height: 100%;
}
.feature-card-image {
    width: 100%;
    height: 160px;
    object-fit: cover;
    display: block;
    filter: brightness(0.7);
    transition: filter 0.3s ease;
}
.feature-card:hover .feature-card-image {
    filter: brightness(0.85);
}
.feature-card-body {
    padding: 1.2rem 1.3rem 1.3rem 1.3rem;
}
.feature-card-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
}
.feature-card-title {
    font-size: 1.12rem; font-weight: 800; color: #ffffff;
    margin-bottom: 0.3rem;
}
.feature-card-desc {
    font-size: 0.88rem; color: #94a3b8; line-height: 1.5;
}
.feature-card-tag {
    display: inline-block;
    margin-top: 0.8rem;
    background: rgba(37, 99, 235, 0.18);
    color: #60a5fa;
    padding: 0.3rem 0.85rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid rgba(37, 99, 235, 0.3);
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: all 0.2s ease;
}
.feature-card:hover .feature-card-tag {
    background: #2563eb;
    color: #ffffff;
    border-color: #3b82f6;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.5);
}

/* ─── CLEAN DARK CARDS ─── */
.glass-card, .portal-card {
    background: #131d31 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.glass-card:hover, .portal-card:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2) !important;
}

/* ─── VALUATION / PREDICTION PANEL ─── */
.prediction-panel {
    background: #131d31 !important;
    border: 2px solid #2563eb !important;
    border-radius: 14px !important;
    padding: 2.4rem 1.8rem !important;
    text-align: center;
    margin: 1.6rem 0 !important;
    box-shadow: 0 8px 40px rgba(37, 99, 235, 0.25),
                0 0 60px rgba(37, 99, 235, 0.08) !important;
    position: relative;
    overflow: hidden;
}
.prediction-panel::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 100%, rgba(37, 99, 235, 0.06) 0%, transparent 50%);
    pointer-events: none;
}
.prediction-header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff;
    padding: 0.35rem 1.2rem; border-radius: 6px;
    font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
}
.price-label {
    font-size: 0.9rem; color: #94a3b8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem;
}
.price-main {
    font-size: 3.4rem; font-weight: 800;
    color: #ffffff !important; line-height: 1.15;
    letter-spacing: -0.02em; margin: 0.2rem 0;
}
.price-per-sqft {
    display: inline-block;
    background: rgba(37, 99, 235, 0.18); color: #60a5fa;
    padding: 0.4rem 1rem; border-radius: 6px;
    font-size: 0.95rem; font-weight: 700; margin: 0.4rem 0 1.2rem 0;
    border: 1px solid rgba(37, 99, 235, 0.35);
}
.price-range {
    display: flex; justify-content: center; gap: 3rem; margin-top: 1rem;
    padding-top: 1rem; border-top: 1px solid #1e293b;
}
.price-range-item { text-align: center; }
.price-range-value {
    font-size: 1.35rem; font-weight: 700; color: #38bdf8;
}
.price-range-label {
    font-size: 0.82rem; color: #94a3b8; margin-top: 0.2rem; font-weight: 600;
}

/* ─── KPI METRICS with animation ─── */
@keyframes kpiSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 1rem; margin-bottom: 1.8rem;
}
.kpi-card {
    background: #131d31;
    border: 1px solid #1e293b;
    border-top: 3px solid #38bdf8;
    border-radius: 10px; padding: 1.2rem 1rem; text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease;
    animation: kpiSlideUp 0.5s ease-out both;
}
.kpi-card:nth-child(1) { animation-delay: 0s; }
.kpi-card:nth-child(2) { animation-delay: 0.08s; }
.kpi-card:nth-child(3) { animation-delay: 0.16s; }
.kpi-card:nth-child(4) { animation-delay: 0.24s; }
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    border-top-color: #60a5fa;
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
.kpi-value {
    font-size: 1.6rem; font-weight: 800; color: #ffffff; line-height: 1.2;
}
.kpi-label {
    font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; font-weight: 600;
}

/* ─── RECOMMENDATION LISTING CARDS ─── */
@keyframes recCardSlideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.rec-card {
    background: #131d31;
    border: 1px solid #1e293b;
    border-radius: 10px; padding: 1.25rem 1.4rem; margin-bottom: 0.85rem;
    border-left: 5px solid #2563eb;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, border-color 0.25s ease;
    animation: recCardSlideIn 0.4s ease-out both;
}
.rec-card:hover {
    transform: translateX(6px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    border-color: #3b82f6;
}
.rec-card.type-location  { border-left-color: #38bdf8; }
.rec-card.type-budget    { border-left-color: #f59e0b; }
.rec-card.type-config    { border-left-color: #10b981; }
.rec-card.type-landmark  { border-left-color: #8b5cf6; }
.rec-card.type-match     { border-left-color: #2563eb; }

.rec-name {
    font-weight: 700; font-size: 1.08rem; color: #ffffff; margin-bottom: 0.2rem;
}
.rec-locality {
    font-size: 0.88rem; color: #94a3b8; font-weight: 500;
}
.rec-price {
    font-weight: 800; color: #38bdf8; font-size: 1.15rem;
}
.bhk-badge {
    display: inline-block;
    background: rgba(37, 99, 235, 0.18); color: #93c5fd;
    padding: 0.2rem 0.6rem; border-radius: 4px;
    font-size: 0.78rem; font-weight: 700; margin-right: 0.3rem;
    border: 1px solid rgba(37, 99, 235, 0.3);
}
.verified-tag {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15); color: #34d399;
    padding: 0.15rem 0.5rem; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; margin-left: 0.4rem;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.featured-tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.08));
    color: #fbbf24;
    padding: 0.15rem 0.55rem; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; margin-left: 0.4rem;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.rec-link {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff !important;
    padding: 0.4rem 1rem; border-radius: 6px;
    text-decoration: none; font-size: 0.84rem; font-weight: 700;
    transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
}
.rec-link:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}

/* ─── PRIMARY BUTTONS ─── */
.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.01em !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* ─── METRICS & FORMS ─── */
[data-testid="stMetric"] {
    background: #131d31 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important; padding: 0.9rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}
[data-testid="stForm"] {
    border: 1px solid #1e293b !important;
    border-radius: 10px !important; padding: 1.6rem !important;
    background: #131d31 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}

/* ─── EXPANDERS ─── */
[data-testid="stExpander"] {
    background: #131d31 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #f8fafc !important;
}

/* ─── DIVIDER & FOOTER ─── */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155 30%, #38bdf8 50%, #334155 70%, transparent);
    margin: 1.8rem 0; border: none;
    opacity: 0.6;
}
.site-footer {
    text-align: center; padding: 2rem 0 1.5rem 0; margin-top: 3rem;
    border-top: 1px solid #1e293b;
    color: #64748b; font-size: 0.84rem;
}
.footer-brand {
    color: #94a3b8; font-weight: 700; margin-bottom: 0.3rem; font-size: 0.95rem;
}

/* ─── TIMELINE ─── */
.timeline { position: relative; padding-left: 2rem; margin: 1.5rem 0; }
.timeline::before {
    content: ''; position: absolute; left: 0.5rem; top: 0; bottom: 0;
    width: 2px; background: #2563eb;
}
.timeline-item { position: relative; padding-bottom: 1.5rem; padding-left: 1rem; }
.timeline-item::before {
    content: ''; position: absolute; left: -1.55rem; top: 0.35rem;
    width: 10px; height: 10px; border-radius: 50%;
    background: #38bdf8; border: 2px solid #0b0f19;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.4);
}
.timeline-title { font-weight: 700; color: #ffffff; margin-bottom: 0.25rem; }
.timeline-desc { color: #94a3b8; font-size: 0.9rem; line-height: 1.5; }
.stat-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(37, 99, 235, 0.15); border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 6px; padding: 0.4rem 0.8rem; margin: 0.2rem;
    font-size: 0.88rem; color: #60a5fa; font-weight: 600;
}

/* ─── STREAMLIT TABS STYLING (DARK PORTAL) ─── */
@keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.6rem;
    border-bottom: 2px solid #1e293b;
    padding-bottom: 0.4rem;
    background: transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    padding: 0.6rem 1.3rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(56, 189, 248, 0.35) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(37, 99, 235, 0.2) !important;
    color: #38bdf8 !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.2) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #38bdf8 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    animation: tabFadeIn 0.35s ease-out;
}

/* ─── GLOBAL SELECTBOX & INPUT CONTROLS ─── */
div[data-baseweb="select"] > div {
    background-color: #131f37 !important;
    border: 1.5px solid #2563eb !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
    min-height: 42px !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #38bdf8 !important;
    background-color: #182848 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.35) !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {
    color: #ffffff !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] svg {
    fill: #38bdf8 !important;
    width: 20px !important;
    height: 20px !important;
}
div[data-baseweb="popover"], ul[data-baseweb="menu"] {
    background-color: #0f172a !important;
    border: 1.5px solid #38bdf8 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important;
}
ul[data-baseweb="menu"] li {
    color: #f1f5f9 !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}
"""



# ══════════════════════════════════════════════════════════════════════════
# Public helpers
# ══════════════════════════════════════════════════════════════════════════

def apply_theme():
    """Inject the portal dark design system CSS + Google Fonts + sidebar branding."""
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800'
        '&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{DESIGN_CSS}</style>", unsafe_allow_html=True)
    _render_sidebar()


def _render_sidebar():
    """Styled sidebar branding."""
    st.sidebar.markdown(
        f'<div class="sidebar-brand">{ICON_SVG}'
        f'<div><div class="sidebar-brand-name">EstateIQ</div>'
        f'<div class="sidebar-caption">Gurugram Property Intelligence</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_hero(title, tagline, show_logo=False, image_path=None,
                hero_stats=None):
    """Render a portal header banner.

    If *image_path* points to a valid image, the hero uses a full-bleed
    background image with a dark gradient overlay (99acres-style).
    Otherwise falls back to a gradient-only banner.

    *hero_stats* is an optional list of (value, label) tuples shown as
    frosted-glass stat pills at the bottom of the hero.
    """
    logo_html = f'<div style="margin-bottom:0.8rem;">{LOGO_SVG}</div>' if show_logo else ""
    badge_html = '<div class="hero-badge-pill">✓ Gurugram Real Estate Intelligence</div>'

    # Stats row
    stats_html = ""
    if hero_stats:
        items = "".join(
            f'<div class="hero-stat">'
            f'<div class="hero-stat-value">{v}</div>'
            f'<div class="hero-stat-label">{l}</div></div>'
            for v, l in hero_stats
        )
        stats_html = f'<div class="hero-stats-row">{items}</div>'

    # Try image hero
    data_uri = _image_to_base64(image_path) if image_path else None

    if data_uri:
        st.markdown(
            f'<div class="hero-image-wrap" style="background-image:url({data_uri});">'
            f'<div class="hero-overlay"></div>'
            f'<div class="hero-content">'
            f'{badge_html}'
            f'{logo_html}'
            f'<div class="hero-page-title">{title}</div>'
            f'<div class="hero-tagline">{tagline}</div>'
            f'{stats_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Fallback gradient hero
        st.markdown(
            f'<div class="hero-portal-wrap">'
            f'{badge_html}'
            f'{logo_html}'
            f'<div class="hero-page-title">{title}</div>'
            f'<div class="hero-tagline">{tagline}</div>'
            f'{stats_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_feature_cards(cards):
    """Render a grid of feature showcase cards.

    *cards* is a list of dicts:
        {"icon": "🏠", "title": "...", "desc": "...",
         "tag": "EXPLORE →", "image": "/path/to/img.jpg", "link": "/Analytics"}
    """
    cards_html = ""
    for c in cards:
        img_html = ""
        if c.get("image"):
            data_uri = _image_to_base64(c["image"])
            if data_uri:
                img_html = f'<img class="feature-card-image" src="{data_uri}" alt="{c["title"]}" />'
            else:
                # Gradient placeholder
                img_html = (
                    '<div class="feature-card-image" style="'
                    'background: linear-gradient(135deg, #0f2756 0%, #1a4494 100%);'
                    'height:160px;"></div>'
                )
        tag_html = f'<div class="feature-card-tag">{c.get("tag", "EXPLORE →")}</div>' if c.get("tag") else ""
        card_content = (
            f'<div class="feature-card">'
            f'{img_html}'
            f'<div class="feature-card-body">'
            f'<div class="feature-card-icon">{c.get("icon", "")}</div>'
            f'<div class="feature-card-title">{c["title"]}</div>'
            f'<div class="feature-card-desc">{c["desc"]}</div>'
            f'{tag_html}'
            f'</div></div>'
        )

        link = c.get("link")
        target = c.get("target", "_self")
        if link:
            cards_html += f'<a href="{link}" target="{target}" class="feature-card-link">{card_content}</a>'
        else:
            cards_html += card_content

    st.markdown(f'<div class="feature-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_gradient_divider():
    """A clean separator divider with subtle gradient."""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def render_footer():
    """Portal footer."""
    st.markdown(
        '<div class="site-footer">'
        '<div class="footer-brand">EstateIQ Gurugram · Real Estate Intelligence Portal</div>'
        'Estimates are ML model-generated for research and decision-support. '
        'Verified against sector records in Gurugram, Haryana. © 2026'
        '</div>',
        unsafe_allow_html=True,
    )


def render_kpi_cards(kpis):
    """Render a responsive row of KPI stat cards."""
    cards_html = ""
    for icon, value, label in kpis:
        cards_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>'
        )
    st.markdown(f'<div class="kpi-row">{cards_html}</div>', unsafe_allow_html=True)


def render_prediction_result(low, point, high, r2=None, mae=None, built_up_area=None):
    """Render the valuation panel without R2/MAE validation box."""
    pps_html = ""
    if built_up_area and float(built_up_area) > 0:
        pps = (point * 1e7) / float(built_up_area)
        pps_html = f'<div class="price-per-sqft">Avg Rate: ₹ {pps:,.0f} / sq ft</div>'

    st.markdown(f"""
    <div class="prediction-panel">
        <div class="prediction-header-badge">✓ ESTIMATED MARKET VALUE</div>
        <div class="price-label">Expected Property Price</div>
        <div class="price-main">₹ {point:.2f} Cr</div>
        {pps_html}
        <div class="price-range">
            <div class="price-range-item">
                <div class="price-range-value">₹ {low:.2f} Cr</div>
                <div class="price-range-label">Fair Low</div>
            </div>
            <div class="price-range-item">
                <div class="price-range-value">₹ {point:.2f} Cr</div>
                <div class="price-range-label">Fair Value</div>
            </div>
            <div class="price-range-item">
                <div class="price-range-value">₹ {high:.2f} Cr</div>
                <div class="price-range-label">Fair High</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


_REC_CARD_TYPES = {
    "📍": "location",
    "💰": "budget",
    "🏠": "config",
    "🗺️": "landmark",
    "✨": "match",
}


def get_card_type(section_title):
    """Infer the card CSS type from the section title's leading emoji."""
    for emoji, css_type in _REC_CARD_TYPES.items():
        if section_title.startswith(emoji):
            return css_type
    return "match"


def render_rec_card(name, locality, bhk_configs, min_price, max_price, link,
                    card_type="match", featured=False):
    """Render a property listing card."""
    bhk_html = "".join(f'<span class="bhk-badge">{b}</span>' for b in (bhk_configs or []))
    if min_price is not None and max_price is not None:
        import math
        if not (math.isnan(min_price) or math.isnan(max_price)):
            price_html = f'<span class="rec-price">₹ {min_price:.2f} – {max_price:.2f} Cr</span>'
        else:
            price_html = '<span class="rec-price" style="color:#64748b;">Price on request</span>'
    else:
        price_html = '<span class="rec-price" style="color:#64748b;">Price on request</span>'

    import urllib.parse
    clean_locality = str(locality or 'Gurugram').replace('Sector ', 'Sector-')
    search_query = urllib.parse.quote_plus(f"{name} {clean_locality} Gurgaon property 99acres")
    safe_link = f"https://www.google.com/search?q={search_query}"

    link_html = f'<a class="rec-link" href="{safe_link}" target="_blank" rel="noopener noreferrer">View Details ↗</a>'

    featured_html = '<span class="featured-tag">★ FEATURED</span>' if featured else ""

    st.markdown(f"""
    <div class="rec-card type-{card_type}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
                    flex-wrap:wrap; gap:0.6rem;">
            <div style="flex:1; min-width:220px;">
                <div class="rec-name">{name} <span class="verified-tag">✓ RERA / Verified</span>{featured_html}</div>
                <div class="rec-locality">📍 {locality or 'Gurugram'}</div>
                <div style="margin-top:0.5rem;">{bhk_html}</div>
            </div>
            <div style="text-align:right;">
                {price_html}
                <div style="margin-top:0.4rem;">{link_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_brand():
    """Legacy alias."""
    _render_sidebar()


def render_header(tagline="Smart algorithmic price estimates for the Gurugram residential market"):
    """Legacy alias."""
    render_hero("EstateIQ Gurugram", tagline, show_logo=True)