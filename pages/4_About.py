"""
About page - the story behind this project, the data, and me.
"""

import streamlit as st

from branding import apply_theme, render_hero, render_gradient_divider, render_footer

YOUR_NAME = "Mahesh Singh Beniwal"
YOUR_EMAIL = "maheshbeniwal74@gmail.com"
GITHUB_URL = "https://github.com/maheshsingh-74/gurugram-real-state-app-"

st.set_page_config(page_title="EstateIQ Gurugram | About", page_icon="👋", layout="wide")
apply_theme()

render_hero(
    "👋 About This Project",
    "Why I built EstateIQ Gurugram, where the data came from, and how it all fits together",
)

# ── Intro ─────────────────────────────────────────────────────────────
st.markdown(f"""
Hi, I'm **{YOUR_NAME}** — this project started as my data science capstone, built around one question I kept
running into while looking at Gurugram real estate listings online: *could I actually predict what a property
should cost*, using the same information a buyer sees on a listing page?
""")

render_gradient_divider()

# ── Data source ───────────────────────────────────────────────────────
st.markdown("### 📊 Where the data comes from")

st.markdown("""
Everything here is aggregated from verified Gurugram residential market listings. The data used is as per
2023–2024 pricing; real-time pricing may differ. Two separate datasets went into this:

- A larger property listings dataset (property type, sector, BHK, bathrooms, built-up area, furnishing, floor,
  luxury tier, possession status, and price) — powers the **price predictor**.
- A smaller, richer dataset of ~250 individual apartment complexes, with their nearby landmarks, distances,
  facilities/amenities, and BHK-wise price ranges — powers the **recommendations** and the **amenities word cloud**.
""")

# ── Data challenges as stat badges ────────────────────────────────────
st.markdown("**Real-world data challenges I tackled:**")
st.markdown("""
<div style="margin: 0.5rem 0 1.5rem 0;">
    <span class="stat-badge">🏷️ 10 spellings of "IGI Airport etc."</span>
    <span class="stat-badge">📏 6+ distance unit formats</span>
    <span class="stat-badge">💰 Mixed ₹L/Cr in same field</span>
    <span class="stat-badge">🧹 94 single-price entries rescued</span>
    <span class="stat-badge">📋 Stray header row in CSV</span>
    <span class="stat-badge">🗺️ 20% distances mis-parsed originally</span>
</div>
""", unsafe_allow_html=True)

render_gradient_divider()

# ── Pipeline timeline ─────────────────────────────────────────────────
st.markdown("### 🔧 The Pipeline")

st.markdown("""
<div class="timeline">
    <div class="timeline-item">
        <div class="timeline-title">1. Cleaning & EDA</div>
        <div class="timeline-desc">
            Fixed the issues above, removed genuine outliers, understood what actually
            drives price in this market — sector and built-up area, unsurprisingly, matter a lot.
        </div>
    </div>
    <div class="timeline-item">
        <div class="timeline-title">2. Feature Engineering</div>
        <div class="timeline-desc">
            Encoding categorical fields properly, without accidentally duplicating information
            across encodings (sector/agePossession were double-encoded — caught and fixed).
        </div>
    </div>
    <div class="timeline-item">
        <div class="timeline-title">3. Model Comparison</div>
        <div class="timeline-desc">
            Tuned XGBoost, LightGBM, and CatBoost with randomized search + 5-fold CV,
            evaluated on a held-out test set none of them saw during tuning. XGBoost came out ahead.
        </div>
    </div>
    <div class="timeline-item">
        <div class="timeline-title">4. Recommender System</div>
        <div class="timeline-desc">
            Built independently on the amenities/landmark dataset, using facility overlap,
            price/area similarity, and real geographic distance — not just "properties in the same sector."
        </div>
    </div>
    <div class="timeline-item">
        <div class="timeline-title">5. This App</div>
        <div class="timeline-desc">
            The predictor, analytics, and recommender — all wired together in a Streamlit multi-page app
            with a shared design system and cross-page state management.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

render_gradient_divider()

# ── Quick note on numbers ─────────────────────────────────────────────
st.markdown("### 📝 A Quick Note on the Numbers")

st.markdown("""
The price predictor's estimate is shown as a **range**, not a single confident number, because it isn't one —
the model has a real, measured error margin on data it never trained on, and I'd rather show that honestly than
hide it behind a single impressive-looking figure. Treat every prediction here as a **starting point for
negotiation**, not a professional valuation.
""")

render_gradient_divider()

# ── Contact ───────────────────────────────────────────────────────────
st.markdown("### 🤝 Get in Touch")

st.markdown("""
If you spot something off in the data, want to talk about how this was built, or just want to connect:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f'<div class="glass-card" style="text-align:center; padding:1.2rem;">'
        f'<div style="font-size:1.5rem; margin-bottom:0.3rem;">📧</div>'
        f'<a href="mailto:{YOUR_EMAIL}" style="color:#34d399; text-decoration:none; font-weight:500;">'
        f'{YOUR_EMAIL}</a></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="glass-card" style="text-align:center; padding:1.2rem;">'
        f'<div style="font-size:1.5rem; margin-bottom:0.3rem;">💻</div>'
        f'<a href="{GITHUB_URL}" target="_blank" style="color:#3b82f6; text-decoration:none; font-weight:500;">'
        f'Project Repo on GitHub</a></div>',
        unsafe_allow_html=True,
    )

render_footer()
