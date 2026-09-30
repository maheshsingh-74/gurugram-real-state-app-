"""
Design system and shared branding for EstateIQ Gurugram — Real Estate Intelligence Portal.
Modeled on the premium proptech visual design of Plush Homes:
- Palette: Warm Gold (#968340 / #b7a775), Deep Navy (#004274), Light Studio (#f8f8f8), Pure White (#ffffff), Deep Charcoal (#111827 / #222222)
- Typography: Playfair Display, Roboto, Outfit
- Clean cards, real property photography, and user's front image
- No third-party phone numbers, emails, or company identities

Exported helpers:
  apply_theme()                 -> Injects CSS + Google Fonts + top ribbon + sidebar branding
  render_top_ribbon()           -> Luxury portal status ribbon (no phone/mail)
  render_hero()                 -> Hero section with user's front photo & gold badges
  render_portal_spotlight()     -> Intelligence platform spotlight banner
  render_featured_projects()    -> Visual showcase of Gurugram projects with real photos
  render_feature_cards()        -> Portal feature navigation cards with real photos
  render_rec_card()             -> Recommendation card with real project thumbnail & gold details CTA
  render_prediction_result()    -> Luxury valuation card with gold badges & market metrics
  render_kpi_cards()            -> Metric stat cards with gold accents
  render_gradient_divider()     -> Gold-tinted gradient divider
  render_footer()               -> Luxury dark footer with platform disclosures
  col_label()                   -> Human-readable column label
  COLUMN_LABELS                 -> Column label mapping
  PLOTLY_LAYOUT                 -> Harmonized chart theme with gold accents
"""

import base64
import os
import urllib.parse
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


# ── Plotly layout tailored with luxury gold palette ───────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#222222", family="'Roboto', 'Outfit', sans-serif", size=13),
    xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#cbd5e1", tickfont=dict(color="#666666")),
    yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#cbd5e1", tickfont=dict(color="#666666")),
    colorway=["#968340", "#004274", "#b7a775", "#16a34a", "#0284c7",
              "#7c3aed", "#d97706", "#dc2626", "#334155", "#059669"],
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#ffffff", font_color="#222222", bordercolor="#968340"),
)


# ── SVGs & Logos ──────────────────────────────────────────────────────────

GOLD_CREST_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="38" height="38">
  <defs>
    <linearGradient id="eqGoldGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#b7a775"/>
      <stop offset="50%" stop-color="#968340"/>
      <stop offset="100%" stop-color="#7a692e"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="92" height="92" rx="18" fill="url(#eqGoldGrad)"/>
  <path d="M50 20 L80 44 L72 44 L72 78 L28 78 L28 44 L20 44 Z" fill="#ffffff" opacity="0.96"/>
  <path d="M42 54 L58 54 L58 78 L42 78 Z" fill="#968340"/>
  <polygon points="50,29 64,40 36,40" fill="#968340"/>
  <circle cx="50" cy="46" r="3" fill="#ffffff"/>
</svg>
"""

ESTATEIQ_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 76" width="280" height="64">
  <defs>
    <linearGradient id="logoGoldGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#b7a775"/>
      <stop offset="50%" stop-color="#968340"/>
      <stop offset="100%" stop-color="#7a692e"/>
    </linearGradient>
  </defs>
  <rect x="0" y="2" width="72" height="72" rx="14" fill="url(#logoGoldGrad)"/>
  <path d="M36 16 L60 36 L53 36 L53 64 L19 64 L19 36 L12 36 Z" fill="#ffffff"/>
  <rect x="30" y="44" width="12" height="20" rx="2" fill="#968340"/>
  <circle cx="44" cy="30" r="3" fill="#ffffff"/>
  <text x="86" y="42" font-family="'Playfair Display', 'Roboto', serif"
        font-size="28" font-weight="700" fill="#111827" letter-spacing="0.5">EstateIQ</text>
  <text x="88" y="62" font-family="'Roboto', sans-serif"
        font-size="9.5" font-weight="600" letter-spacing="2.4" fill="#968340">GURUGRAM REAL ESTATE INTELLIGENCE</text>
</svg>
"""


# ── Image helpers ─────────────────────────────────────────────────────────

def _image_to_base64(path):
    """Read an image file and return its base64-encoded data URI."""
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {
            "jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "webp": "webp", "avif": "avif"
        }.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return None


# ── CSS Design System (Luxury Gold & Navy Proptech Edition) ───────────────

DESIGN_CSS = """
/* ─── BASE TYPOGRAPHY & PALETTE ─── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=Roboto:wght@300;400;500;700;900&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', 'Outfit', -apple-system, sans-serif !important;
    background-color: #f8f8f8 !important;
    color: #222222 !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Playfair Display', 'Roboto', Georgia, serif !important;
    font-weight: 700 !important;
    color: #111827 !important;
    letter-spacing: -0.01em;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: #968340; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7a692e; }

/* ─── LUXURY TOP STATUS RIBBON (CLEAN / NO THIRD PARTY DATA) ─── */
.luxury-top-ribbon {
    background-color: #000000;
    color: #ffffff;
    padding: 0.5rem 1.4rem;
    font-family: 'Roboto', sans-serif;
    font-size: 0.82rem;
    font-weight: 400;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #968340;
    border-radius: 8px 8px 0 0;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.luxury-ribbon-tag {
    background: #968340;
    color: #ffffff;
    font-size: 0.72rem;
    padding: 0.15rem 0.6rem;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ─── SIDEBAR (LUXURY GOLD & SLATE) ─── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.04) !important;
}
[data-testid="stSidebarNav"] a {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.95rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    color: #333333 !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: #fdfaf3 !important;
    color: #968340 !important;
}
[data-testid="stSidebarNav"] [aria-current="page"] {
    background: #f8f4ea !important;
    color: #968340 !important;
    border-left: 3px solid #968340 !important;
    font-weight: 700 !important;
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.8rem 0;
    margin-bottom: 0.8rem;
    border-bottom: 1.5px solid #f0ede6;
}
.sidebar-brand-name {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700;
    font-size: 1.25rem;
    color: #111827;
}
.sidebar-caption {
    color: #968340;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ─── HERO SECTION (LUXURY PHOTO BANNER) ─── */
.luxury-hero-wrap {
    position: relative;
    width: 100%;
    min-height: 380px;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.8rem;
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    border: 1px solid rgba(150, 131, 64, 0.3);
}
.luxury-hero-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        180deg,
        rgba(0, 20, 40, 0.35) 0%,
        rgba(17, 24, 39, 0.6) 45%,
        rgba(10, 15, 25, 0.88) 100%
    );
    z-index: 1;
}
.luxury-hero-content {
    position: relative;
    z-index: 2;
    padding: 2.6rem 2.4rem 2.2rem 2.4rem;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    min-height: 380px;
}
.luxury-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #968340;
    color: #ffffff;
    padding: 0.35rem 1rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    width: fit-content;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}
.luxury-hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0 0 0.4rem 0;
    line-height: 1.18;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}
.luxury-hero-subtitle {
    color: #e5e7eb;
    font-family: 'Roboto', sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.03em;
    margin: 0 0 1.2rem 0;
    max-width: 720px;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
}
.luxury-hero-stats {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.4rem;
}
.luxury-hero-stat-pill {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-left: 3px solid #968340;
    border-radius: 6px;
    padding: 0.45rem 0.9rem;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    min-width: 110px;
}
.luxury-hero-stat-val {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 800;
    line-height: 1.15;
}
.luxury-hero-stat-lbl {
    color: #d1d5db;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.1rem;
}

/* ─── PLATFORM SPOTLIGHT BANNER ─── */
.portal-spotlight-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: 3px solid #968340;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}
.portal-spotlight-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 0.8rem;
}
.portal-spotlight-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
    margin: 0;
}
.portal-spotlight-badge {
    background: #fdfaf3;
    color: #968340;
    border: 1px solid #e8dec5;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.portal-spotlight-body {
    color: #4b5563;
    font-size: 0.95rem;
    line-height: 1.65;
    margin-bottom: 1rem;
}
.portal-cta-btn {
    background-color: #968340;
    color: #ffffff !important;
    font-family: 'Roboto', sans-serif;
    font-size: 0.84rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.45rem 1.1rem;
    border-radius: 4px;
    text-decoration: none;
    border: 1px solid #b7a775;
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    box-shadow: 0 2px 6px rgba(150, 131, 64, 0.25);
}
.portal-cta-btn:hover {
    background-color: #b7a775;
    transform: translateY(-1px);
    color: #ffffff !important;
}

/* ─── FEATURED PROJECTS SECTION ─── */
.featured-section-header {
    text-align: center;
    margin: 2.2rem 0 1.2rem 0;
}
.featured-section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.9rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.3rem;
}
.featured-section-subtitle {
    color: #968340;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.featured-projects-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.4rem;
    margin-bottom: 2.2rem;
}
@media (max-width: 992px) {
    .featured-projects-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
    .featured-projects-grid { grid-template-columns: 1fr; }
    .luxury-hero-title { font-size: 1.85rem !important; }
}
.luxury-project-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
}
.luxury-project-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.1);
    border-color: #968340;
}
.luxury-project-img-wrap {
    position: relative;
    width: 100%;
    height: 200px;
    overflow: hidden;
    background: #f1f1f1;
}
.luxury-project-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.4s ease;
}
.luxury-project-card:hover .luxury-project-img {
    transform: scale(1.05);
}
.luxury-project-tag-wrap {
    position: absolute;
    top: 10px;
    left: 10px;
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    z-index: 2;
}
.luxury-status-pill {
    background: #968340;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.luxury-nri-pill {
    background: #004274;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.luxury-project-body {
    padding: 1.15rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
.luxury-project-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.2rem;
}
.luxury-project-address {
    color: #6b7280;
    font-size: 0.82rem;
    margin-bottom: 0.75rem;
}
.luxury-project-price {
    color: #968340;
    font-size: 1.08rem;
    font-weight: 800;
    margin-bottom: 0.85rem;
    font-family: 'Roboto', sans-serif;
}
.luxury-project-btn {
    display: block;
    text-align: center;
    background: #968340;
    color: #ffffff !important;
    padding: 0.45rem 1rem;
    border-radius: 4px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: auto;
    transition: all 0.2s;
    border: 1px solid #968340;
}
.luxury-project-btn:hover {
    background: #b7a775;
    border-color: #b7a775;
    color: #ffffff !important;
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
}
@media (max-width: 1024px) and (min-width: 769px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}
.feature-card {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    transition: all 0.25s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
    border-color: #968340;
}
.feature-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    height: 100%;
}
.feature-card-image {
    width: 100%;
    height: 165px;
    object-fit: cover;
    display: block;
    transition: transform 0.35s ease;
}
.feature-card:hover .feature-card-image {
    transform: scale(1.04);
}
.feature-card-body {
    padding: 1.1rem 1.2rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
.feature-card-icon {
    font-size: 1.45rem;
    margin-bottom: 0.25rem;
}
.feature-card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.feature-card-desc {
    font-size: 0.85rem;
    color: #4b5563;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}
.feature-card-tag {
    margin-top: auto;
    display: inline-block;
    background: #fdfaf3;
    color: #968340;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    border: 1px solid #e8dec5;
    letter-spacing: 0.04em;
    width: fit-content;
    transition: all 0.2s;
}
.feature-card:hover .feature-card-tag {
    background: #968340;
    color: #ffffff;
    border-color: #968340;
}

/* ─── VALUATION / PREDICTION PANEL ─── */
.prediction-panel {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-top: 4px solid #968340 !important;
    border-radius: 12px !important;
    padding: 2.2rem 1.8rem !important;
    text-align: center;
    margin: 1.6rem 0 !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05) !important;
    position: relative;
}
.prediction-header-badge {
    display: inline-block;
    background: #968340;
    color: #ffffff;
    padding: 0.35rem 1.2rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}
.price-label {
    font-size: 0.86rem;
    color: #6b7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.price-main {
    font-size: 3.2rem;
    font-weight: 800;
    color: #111827 !important;
    line-height: 1.15;
    margin: 0.2rem 0;
    font-family: 'Roboto', sans-serif;
}
.price-per-sqft {
    display: inline-block;
    background: #fdfaf3;
    color: #968340;
    padding: 0.35rem 1rem;
    border-radius: 4px;
    font-size: 0.92rem;
    font-weight: 700;
    margin: 0.4rem 0 1.2rem 0;
    border: 1px solid #e8dec5;
}
.price-range {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #f0ede6;
}
.price-range-item { text-align: center; }
.price-range-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #968340;
}
.price-range-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.2rem;
    font-weight: 600;
}

/* ─── KPI METRICS ─── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 1rem;
    margin-bottom: 1.8rem;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: 3px solid #968340;
    border-radius: 8px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.07);
    border-top-color: #b7a775;
}
.kpi-icon { font-size: 1.35rem; margin-bottom: 0.25rem; }
.kpi-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.25rem;
    font-weight: 600;
}

/* ─── RECOMMENDATION PROPERTY CARDS ─── */
.rec-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    border-left: 4px solid #968340;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.rec-card:hover {
    transform: translateX(4px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}
.rec-card.type-location  { border-left-color: #004274; }
.rec-card.type-budget    { border-left-color: #968340; }
.rec-card.type-config    { border-left-color: #16a34a; }
.rec-card.type-landmark  { border-left-color: #7c3aed; }
.rec-card.type-match     { border-left-color: #968340; }

.rec-thumb-img {
    width: 110px;
    height: 82px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    margin-right: 1rem;
    flex-shrink: 0;
}
.rec-name {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #111827;
    margin-bottom: 0.15rem;
}
.rec-locality {
    font-size: 0.84rem;
    color: #6b7280;
    font-weight: 500;
}
.rec-price {
    font-weight: 800;
    color: #968340;
    font-size: 1.12rem;
    font-family: 'Roboto', sans-serif;
}
.bhk-badge {
    display: inline-block;
    background: #fdfaf3;
    color: #968340;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.74rem;
    font-weight: 700;
    margin-right: 0.3rem;
    border: 1px solid #e8dec5;
}
.verified-tag {
    display: inline-block;
    background: #f0fdf4;
    color: #16a34a;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 0.35rem;
    border: 1px solid #bbf7d0;
}
.featured-tag {
    display: inline-block;
    background: #968340;
    color: #ffffff;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 0.35rem;
}
.rec-link {
    display: inline-block;
    background: #968340;
    color: #ffffff !important;
    padding: 0.38rem 0.95rem;
    border-radius: 4px;
    text-decoration: none;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    transition: background 0.2s, transform 0.15s;
    box-shadow: 0 2px 6px rgba(150, 131, 64, 0.25);
}
.rec-link:hover {
    background: #b7a775;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(150, 131, 64, 0.35);
}

/* ─── GRADIENT DIVIDER ─── */
.gradient-divider {
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #968340, transparent);
    margin: 2rem 0;
    border: none;
    opacity: 0.45;
}

/* ─── LUXURY DARK FOOTER (ESTATEIQ / NO THIRD PARTY DATA) ─── */
.luxury-footer {
    background: #000000;
    color: #ffffff;
    padding: 2.8rem 1.8rem 1.8rem 1.8rem;
    margin-top: 3.5rem;
    border-radius: 12px 12px 0 0;
    border-top: 3px solid #968340;
}
.luxury-footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}
@media (max-width: 768px) {
    .luxury-footer-grid { grid-template-columns: 1fr; }
}
.luxury-footer-brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
}
.luxury-footer-desc {
    color: #9ca3af;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.luxury-footer-col-title {
    color: #968340;
    font-family: 'Roboto', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}
.luxury-footer-links {
    list-style: none;
    padding: 0;
    margin: 0;
}
.luxury-footer-links li {
    margin-bottom: 0.45rem;
}
.luxury-footer-links a {
    color: #d1d5db !important;
    text-decoration: none;
    font-size: 0.84rem;
    transition: color 0.2s;
}
.luxury-footer-links a:hover {
    color: #968340 !important;
}
.luxury-footer-bottom {
    border-top: 1px solid #222222;
    padding-top: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.8rem;
    font-size: 0.8rem;
    color: #6b7280;
}

/* ─── STREAMLIT UI OVERRIDES ─── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 2px solid #e5e7eb;
    background: transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #6b7280 !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
    padding: 0.65rem 1.2rem !important;
    transition: all 0.2s ease !important;
    margin-bottom: -2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #968340 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #968340 !important;
    border-bottom: 2px solid #968340 !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #968340 !important;
}

button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #968340 0%, #7a692e 100%) !important;
    color: #ffffff !important;
    border: 1px solid #968340 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 8px rgba(150, 131, 64, 0.25) !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background: linear-gradient(135deg, #b7a775 0%, #968340 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(150, 131, 64, 0.35) !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
    transition: all 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #968340 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 6px !important;
    color: #111827 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #968340 !important;
    box-shadow: 0 0 0 2px rgba(150, 131, 64, 0.2) !important;
}
"""


# ══════════════════════════════════════════════════════════════════════════
# Public helpers
# ══════════════════════════════════════════════════════════════════════════

def apply_theme():
    """Inject the luxury design system CSS + Google Fonts + top ribbon + sidebar branding."""
    st.markdown(f"<style>{DESIGN_CSS}</style>", unsafe_allow_html=True)
    _render_sidebar()


def render_top_ribbon():
    """Render the sleek luxury dark status ribbon."""
    ribbon_html = (
        '<div class="luxury-top-ribbon">'
        '<div style="display:flex; align-items:center; gap:0.9rem; flex-wrap:wrap;">'
        '<span style="color:#ffffff; font-weight:600;">EstateIQ Gurugram</span>'
        '<span style="color:#d1d5db;">|</span>'
        '<span style="color:#d1d5db;">Real Estate Valuation & Property Intelligence</span>'
        '</div>'
        '<div style="display:flex; align-items:center; gap:0.6rem;">'
        '<span class="luxury-ribbon-tag">RERA Audited</span>'
        '<span style="color:#b7a775; font-size:0.8rem; font-weight:500;">Turning Vision Into Reality</span>'
        '</div>'
        '</div>'
    )
    st.markdown(ribbon_html, unsafe_allow_html=True)


# Backwards compatibility alias
def render_plush_topbar():
    render_top_ribbon()


def _render_sidebar():
    """Styled sidebar branding."""
    st.sidebar.markdown(
        f'<div class="sidebar-brand">{GOLD_CREST_SVG}'
        f'<div><div class="sidebar-brand-name">EstateIQ</div>'
        f'<div class="sidebar-caption">Gurugram Property Intelligence</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_hero(title=None, tagline=None, show_logo=True, image_path=None,
                hero_stats=None):
    """Render the luxury hero banner with user's front photo.

    Defaults to the user's uploaded front photo (`static/user_front_image.png`).
    """
    effective_img = None
    candidates = [image_path, "static/user_front_image.png", "static/plush_luxury_hero.jpg", "static/hero_banner.jpg"]
    for c in candidates:
        if c and os.path.exists(c):
            effective_img = c
            break

    data_uri = _image_to_base64(effective_img) if effective_img else None

    h_title = title or "Turning Vision Into Reality"
    h_tagline = tagline or "Residential Floors | Luxury Apartments | Commercial SCO | Plots across Gurugram"

    badge_html = '<div class="luxury-hero-badge">✨ Real Estate Intelligence · Gurugram</div>'

    stats_html = ""
    if hero_stats:
        items = "".join(
            f'<div class="luxury-hero-stat-pill">'
            f'<div class="luxury-hero-stat-val">{v}</div>'
            f'<div class="luxury-hero-stat-lbl">{l}</div></div>'
            for v, l in hero_stats
        )
        stats_html = f'<div class="luxury-hero-stats">{items}</div>'

    bg_style = f"background-image:url({data_uri});" if data_uri else "background: linear-gradient(135deg, #004274 0%, #002244 100%);"
    logo_html = f'<div style="margin-bottom:0.7rem;">{ESTATEIQ_LOGO_SVG}</div>' if show_logo else ""

    hero_html = (
        f'<div class="luxury-hero-wrap" style="{bg_style}">'
        f'<div class="luxury-hero-overlay"></div>'
        f'<div class="luxury-hero-content">'
        f'{badge_html}'
        f'{logo_html}'
        f'<div class="luxury-hero-title">{h_title}</div>'
        f'<div class="luxury-hero-subtitle">{h_tagline}</div>'
        f'{stats_html}'
        f'</div></div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_portal_spotlight():
    """Render the portal intelligence spotlight banner."""
    spotlight_html = (
        '<div class="portal-spotlight-box">'
        '<div class="portal-spotlight-header">'
        '<div>'
        '<h3 class="portal-spotlight-title">Precision Real Estate Intelligence for Gurugram</h3>'
        '<div style="color:#968340; font-size:0.8rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin-top:2px;">'
        'Audited Market Records & Algorithmic Valuations'
        '</div>'
        '</div>'
        '<span class="portal-spotlight-badge">Decision Support Platform</span>'
        '</div>'
        '<div class="portal-spotlight-body">'
        'EstateIQ delivers unbiased, machine-learning powered market valuations and recommendations '
        'across 100+ sectors in Gurugram. Built on comprehensive registrar records and verified property listings, '
        'the portal empowers buyers, investors, and homeowners with instant fair-value estimates, neighborhood '
        'comparatives, and deep feature impact insights with zero guesswork.'
        '</div>'
        '<div style="display:flex; gap:0.8rem; flex-wrap:wrap;">'
        '<a href="#valuation-calculator" class="portal-cta-btn">🔍 Calculate Market Price</a>'
        '<a href="Recommendations" class="portal-cta-btn" style="background:#004274; border-color:#004274;">🏘️ Explore Recommendations</a>'
        '</div>'
        '</div>'
    )
    st.markdown(spotlight_html, unsafe_allow_html=True)


# Backwards compatibility alias
def render_plush_about_banner():
    render_portal_spotlight()


def render_featured_projects():
    """Render Gurugram luxury projects with real photography (NO markdown indentation bugs)."""
    projects = [
        {
            "title": "Sobha Crescent",
            "address": "Sector-63A, Extension, Gurugram",
            "price": "Starting From ₹5.75 Cr*",
            "type": "Residential Projects",
            "nri": True,
            "img": "static/sobha_crescent.webp",
        },
        {
            "title": "Whiteland Westin Residences",
            "address": "Sector 103, Dwarka Expressway, Gurugram",
            "price": "₹ 6.25 Crores",
            "type": "Residential Projects",
            "nri": True,
            "img": "static/whiteland_resort.webp",
        },
        {
            "title": "Tulip Melrose",
            "address": "Sector 70, Southern Peripheral Rd, Gurugram",
            "price": "₹ 4.43 Crores",
            "type": "Residential Projects",
            "nri": True,
            "img": "static/tulip_melrose.jpg",
        },
        {
            "title": "DLF The Camellias",
            "address": "Sector 42, Golf Course Road, Gurugram",
            "price": "Starting From ₹18 Cr*",
            "type": "Ultra Luxury Residences",
            "nri": True,
            "img": "static/dlf_camellias.jpg",
        },
        {
            "title": "Silverglades The Legacy",
            "address": "Sector 59, Golf Course Extn, Gurugram",
            "price": "₹ 6.72 Crores",
            "type": "Ultra Luxury Floors",
            "nri": True,
            "img": "static/silverglades_legacy.webp",
        },
        {
            "title": "DLF The Crest",
            "address": "Sector 54, Golf Course Road, Gurugram",
            "price": "Starting From ₹6.5 Cr*",
            "type": "Luxury Residential",
            "nri": True,
            "img": "static/dlf_crest.jpg",
        },
    ]

    cards_parts = []
    for p in projects:
        data_uri = _image_to_base64(p["img"])
        img_src = data_uri if data_uri else "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=600"
        nri_badge = '<span class="luxury-nri-pill">NRI Preferred</span>' if p["nri"] else ""
        query = urllib.parse.quote_plus(f"{p['title']} {p['address']} Gurgaon property price")
        link = f"https://www.google.com/search?q={query}"

        card = (
            '<div class="luxury-project-card">'
            '<div class="luxury-project-img-wrap">'
            '<div class="luxury-project-tag-wrap">'
            '<span class="luxury-status-pill">Active Market</span>'
            f'{nri_badge}'
            '</div>'
            f'<img class="luxury-project-img" src="{img_src}" alt="{p["title"]}" />'
            '</div>'
            '<div class="luxury-project-body">'
            f'<div class="luxury-project-title">{p["title"]}</div>'
            f'<div class="luxury-project-address">📍 {p["address"]}</div>'
            f'<div class="luxury-project-price">{p["price"]}</div>'
            f'<a href="{link}" target="_blank" rel="noopener noreferrer" class="luxury-project-btn">View Details →</a>'
            '</div>'
            '</div>'
        )
        cards_parts.append(card)

    cards_html = "".join(cards_parts)
    full_section = (
        '<div class="featured-section-header">'
        '<h2 class="featured-section-title">Featured Developments in Gurugram</h2>'
        '<div class="featured-section-subtitle">HIGH DEMAND LUXURY RESIDENTIAL & COMMERCIAL CORRIDORS</div>'
        '</div>'
        f'<div class="featured-projects-grid">{cards_html}</div>'
    )
    st.markdown(full_section, unsafe_allow_html=True)


# Backwards compatibility alias
def render_hot_selling_projects():
    render_featured_projects()


def render_feature_cards(cards):
    """Render feature cards showcasing portal features."""
    cards_parts = []
    for c in cards:
        img_html = ""
        if c.get("image"):
            data_uri = _image_to_base64(c["image"])
            if data_uri:
                img_html = f'<img class="feature-card-image" src="{data_uri}" alt="{c["title"]}" />'
            else:
                img_html = '<div class="feature-card-image" style="background:linear-gradient(135deg, #fdfaf3 0%, #f0ede6 100%); height:165px;"></div>'

        tag_html = f'<div class="feature-card-tag">{c.get("tag", "EXPLORE →")}</div>' if c.get("tag") else ""
        card_content = (
            '<div class="feature-card">'
            f'{img_html}'
            '<div class="feature-card-body">'
            f'<div class="feature-card-icon">{c.get("icon", "")}</div>'
            f'<div class="feature-card-title">{c["title"]}</div>'
            f'<div class="feature-card-desc">{c["desc"]}</div>'
            f'{tag_html}'
            '</div></div>'
        )

        link = c.get("link")
        target = c.get("target", "_self")
        if link:
            cards_parts.append(f'<a href="{link}" target="{target}" class="feature-card-link">{card_content}</a>')
        else:
            cards_parts.append(card_content)

    full_grid = f'<div class="feature-grid">{"".join(cards_parts)}</div>'
    st.markdown(full_grid, unsafe_allow_html=True)


def render_gradient_divider():
    """Gold-tinted gradient divider."""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def render_footer():
    """Clean luxury dark footer with portal intelligence data (no third-party data)."""
    footer_html = (
        '<div class="luxury-footer">'
        '<div class="luxury-footer-grid">'
        '<div>'
        '<div class="luxury-footer-brand-title">EstateIQ Gurugram</div>'
        '<div class="luxury-footer-desc">'
        'Machine learning-powered property valuation and real estate market intelligence '
        'for Gurugram and Delhi-NCR. Providing transparent fair market value ranges, '
        'geospatial micro-market analytics, and similarity-based property matching.'
        '</div>'
        '<div style="color:#9ca3af; font-size:0.82rem; line-height:1.6;">'
        '📍 Gurugram, Haryana, India · Covering 100+ Sectors'
        '</div>'
        '</div>'
        '<div>'
        '<div class="luxury-footer-col-title">Portal Tools</div>'
        '<ul class="luxury-footer-links">'
        '<li><a href="#valuation-calculator">AI Market Valuation</a></li>'
        '<li><a href="Recommendations">5-Angle Recommender</a></li>'
        '<li><a href="Analytics">Geospatial Analytics</a></li>'
        '<li><a href="Model_Insights">SHAP Explainability</a></li>'
        '</ul>'
        '</div>'
        '<div>'
        '<div class="luxury-footer-col-title">Corridors</div>'
        '<ul class="luxury-footer-links">'
        '<li><a href="Analytics">Golf Course Road</a></li>'
        '<li><a href="Analytics">Golf Course Ext.</a></li>'
        '<li><a href="Analytics">Dwarka Expressway</a></li>'
        '<li><a href="Analytics">Southern Peripheral Rd</a></li>'
        '</ul>'
        '</div>'
        '<div>'
        '<div class="luxury-footer-col-title">Methodology</div>'
        '<ul class="luxury-footer-links">'
        '<li><a href="Model_Insights">Held-Out Test R² 0.927</a></li>'
        '<li><a href="Model_Insights">Feature Contribution</a></li>'
        '<li><a href="About">Dataset Overview</a></li>'
        '<li><a href="About">Cleaning Pipeline</a></li>'
        '</ul>'
        '</div>'
        '</div>'
        '<div class="luxury-footer-bottom">'
        '<div>© 2026 EstateIQ Gurugram. All Rights Reserved. Real Estate Intelligence Portal.</div>'
        '<div>Valuation estimates are algorithmic approximations for research and decision-support.</div>'
        '</div>'
        '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


def render_kpi_cards(kpis):
    """Render metric stat cards with gold accents."""
    cards_parts = []
    for icon, value, label in kpis:
        cards_parts.append(
            '<div class="kpi-card">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            '</div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(cards_parts)}</div>', unsafe_allow_html=True)


def render_prediction_result(low, point, high, r2=None, mae=None, built_up_area=None):
    """Render the luxury valuation panel."""
    pps_html = ""
    if built_up_area and float(built_up_area) > 0:
        pps = (point * 1e7) / float(built_up_area)
        pps_html = f'<div class="price-per-sqft">Estimated Benchmark Rate: ₹ {pps:,.0f} / sq ft</div>'

    panel_html = (
        '<div class="prediction-panel">'
        '<div class="prediction-header-badge">✓ ESTIMATED FAIR MARKET VALUE</div>'
        '<div class="price-label">Expected Property Price</div>'
        f'<div class="price-main">₹ {point:.2f} Cr</div>'
        f'{pps_html}'
        '<div class="price-range">'
        '<div class="price-range-item">'
        f'<div class="price-range-value">₹ {low:.2f} Cr</div>'
        '<div class="price-range-label">Fair Low</div>'
        '</div>'
        '<div class="price-range-item">'
        f'<div class="price-range-value">₹ {point:.2f} Cr</div>'
        '<div class="price-range-label">Fair Market Value</div>'
        '</div>'
        '<div class="price-range-item">'
        f'<div class="price-range-value">₹ {high:.2f} Cr</div>'
        '<div class="price-range-label">Fair High</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)


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


# Mapping popular properties to real images in static/
PROPERTY_IMAGE_MAP = {
    "sobha": "static/sobha_crescent.webp",
    "whiteland": "static/whiteland_resort.webp",
    "tulip": "static/tulip_melrose.jpg",
    "silverglades": "static/silverglades_legacy.webp",
    "trevoc": "static/trevoc_royal.webp",
    "adani": "static/sobha_crescent.webp",
    "m3m": "static/plush_estate.jpg",
    "smartworld": "static/plush_city_banner.jpg",
    "dlf": "static/user_front_image.png",
    "elan": "static/plush_luxury_hero.jpg",
    "signature": "static/plush_estate.jpg",
    "godrej": "static/plush_luxury_hero.jpg",
    "emaar": "static/silverglades_legacy.webp",
}


def _get_property_thumbnail(name):
    """Get a real image data-uri for a property based on its brand name."""
    name_l = str(name).lower()
    for key, path in PROPERTY_IMAGE_MAP.items():
        if key in name_l and os.path.exists(path):
            return _image_to_base64(path)
    if os.path.exists("static/sobha_crescent.webp"):
        return _image_to_base64("static/sobha_crescent.webp")
    return None


def render_rec_card(name, locality, bhk_configs, min_price, max_price, link,
                    card_type="match", featured=False):
    """Render a property listing card with real thumbnail & luxury gold styling."""
    bhk_html = "".join(f'<span class="bhk-badge">{b}</span>' for b in (bhk_configs or []))
    if min_price is not None and max_price is not None:
        import math
        if not (math.isnan(min_price) or math.isnan(max_price)):
            price_html = f'<span class="rec-price">₹ {min_price:.2f} – {max_price:.2f} Cr</span>'
        else:
            price_html = '<span class="rec-price" style="color:#888;">Price on request</span>'
    else:
        price_html = '<span class="rec-price" style="color:#888;">Price on request</span>'

    clean_locality = str(locality or 'Gurugram').replace('Sector ', 'Sector-')
    search_query = urllib.parse.quote_plus(f"{name} {clean_locality} Gurgaon property price 99acres")
    safe_link = f"https://www.google.com/search?q={search_query}"

    link_html = f'<a class="rec-link" href="{safe_link}" target="_blank" rel="noopener noreferrer">Details →</a>'
    featured_html = '<span class="featured-tag">★ FEATURED</span>' if featured else ""

    thumb_uri = _get_property_thumbnail(name)
    img_tag = f'<img class="rec-thumb-img" src="{thumb_uri}" alt="{name}" />' if thumb_uri else ""

    card_html = (
        f'<div class="rec-card type-{card_type}">'
        '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.9rem;">'
        '<div style="display:flex; align-items:center; flex:1; min-width:240px;">'
        f'{img_tag}'
        '<div>'
        f'<div class="rec-name">{name} <span class="verified-tag">✓ RERA</span>{featured_html}</div>'
        f'<div class="rec-locality">📍 {locality or "Gurugram"}</div>'
        f'<div style="margin-top:0.4rem;">{bhk_html}</div>'
        '</div>'
        '</div>'
        '<div style="text-align:right;">'
        f'{price_html}'
        f'<div style="margin-top:0.35rem;">{link_html}</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_sidebar_brand():
    """Legacy alias."""
    _render_sidebar()


def render_header(tagline="Algorithmic Property Valuation & Real Estate Intelligence"):
    """Legacy alias."""
    render_hero("EstateIQ Gurugram", tagline, show_logo=True)