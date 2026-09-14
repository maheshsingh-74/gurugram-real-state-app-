"""
Recommendations page - property recommendations based on five different
similarity angles. All inputs live on the main page (sidebar is nav-only).
If reached via the "Yes, show me recommendations" button on the predictor
page, this runs automatically using that search.
"""

import os

import joblib
import pandas as pd
import streamlit as st

from branding import (
    apply_theme, render_hero, render_gradient_divider, render_footer,
    render_rec_card, get_card_type, render_kpi_cards,
)

ARTIFACT_PATH = os.path.join("recommender_artifacts", "recommender_artifacts.pkl")

st.set_page_config(page_title="EstateIQ Gurugram | Recommendations", page_icon="🏘️", layout="wide")
apply_theme()

render_hero(
    "🏘️ Property Recommendations",
    "Five different angles on what might suit you — not just one ranked list",
    image_path="static/property_card_bg.jpg",
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(f"Couldn't find {ARTIFACT_PATH}. Run recommender_system.py first.")
        st.stop()
    return joblib.load(ARTIFACT_PATH)


artifacts = load_artifacts()
meta = artifacts["properties_meta"]
location_df = artifacts["location_df"]
landmarks = artifacts["landmarks"]
combined_sim = artifacts["combined_similarity"]
property_names = artifacts["property_names"]

overall_min = float(meta["MinPriceCr"].min(skipna=True))
overall_max = float(meta["MaxPriceCr"].max(skipna=True))
localities = ["Any"] + sorted(meta["Locality"].dropna().unique().tolist())
all_bhk = sorted({b for configs in meta["BHKConfigs"] for b in configs})
all_types = sorted({t for types in meta["PropertyTypes"] for t in types})


def generate_sections(locality=None, property_types=None, bhk=None,
                      budget_range=None, landmark=None, top_n=5):
    sections = {}  # title -> (note, dataframe)
    base = meta.copy()
    if property_types:
        base = base[base["PropertyTypes"].apply(lambda ts: any(t in ts for t in property_types))]
    if bhk:
        base = base[base["BHKConfigs"].apply(lambda cs: any(b in cs for b in bhk))]

    has_locality = bool(locality and locality != "Any")

    if has_locality:
        sec = base[base["Locality"].str.contains(locality, case=False, na=False)]
        sections[f"📍 More options in {locality}"] = (
            "Same area, matching your other filters.", sec.head(top_n))

    if budget_range:
        lo, hi = budget_range
        sec = base[base["MinPriceCr"].notna() & base["MaxPriceCr"].notna()]
        sec = sec[(sec["MaxPriceCr"] >= lo) & (sec["MinPriceCr"] <= hi)]
        if has_locality:
            sec = sec[~sec["Locality"].str.contains(locality, case=False, na=False)]
        sections["💰 Similar budget, other areas"] = (
            "Same price range — worth comparing against elsewhere in Gurugram.", sec.head(top_n))

    if (bhk or property_types) and has_locality:
        sec = base[~base["Locality"].str.contains(locality, case=False, na=False)]
        sections["🏠 Same configuration, different areas"] = (
            "Same BHK/type as your search, but in other localities.", sec.head(top_n))

    if landmark and landmark != "Any" and landmark in location_df.columns:
        ordered_names = location_df[landmark].sort_values().index.tolist()
        sec = base.set_index("PropertyName").reindex(ordered_names).dropna(
            subset=["Locality"]).reset_index()
        sections[f"🗺️ Closest to {landmark}"] = (
            f"Ranked purely by distance to {landmark}, regardless of area.", sec.head(top_n))

    anchor_candidates = base.copy()
    if budget_range:
        mid = sum(budget_range) / 2
        anchor_candidates = anchor_candidates[anchor_candidates["MinPriceCr"].notna()]
        if len(anchor_candidates):
            anchor_candidates = anchor_candidates.assign(
                _d=(((anchor_candidates["MinPriceCr"] + anchor_candidates["MaxPriceCr"]) / 2)
                    - mid).abs()
            ).sort_values("_d")
    if len(anchor_candidates):
        anchor_name = anchor_candidates.iloc[0]["PropertyName"]
        if anchor_name in property_names:
            idx = property_names.index(anchor_name)
            scores = [(i, s) for i, s in enumerate(combined_sim[idx]) if i != idx]
            scores.sort(key=lambda x: -x[1])
            top_names = [property_names[i] for i, _ in scores[:top_n]]
            sec = meta.set_index("PropertyName").reindex(top_names).reset_index()
            sections[f"✨ Best overall match to {anchor_name}"] = (
                "Blends facilities, price/area profile, and nearby-location distances.", sec)

    return sections


def render_sections(sections):
    """Render all recommendation sections with styled cards."""
    if not sections:
        st.warning("No sections could be generated — try widening your filters.")
        return
    any_results = False
    for title, (note, df) in sections.items():
        card_type = get_card_type(title)

        st.markdown(f'<div class="rec-section-header">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-section-note">{note}</div>', unsafe_allow_html=True)

        if df.empty:
            st.caption("No matches for this angle specifically.")
            continue
        any_results = True
        for _, row in df.iterrows():
            render_rec_card(
                name=row["PropertyName"],
                locality=row.get("Locality", ""),
                bhk_configs=row.get("BHKConfigs", []),
                min_price=row.get("MinPriceCr"),
                max_price=row.get("MaxPriceCr"),
                link=row.get("Link"),
                card_type=card_type,
            )
        render_gradient_divider()

    if not any_results:
        st.info("None of the five angles found a match — try a wider budget or fewer filters.")


# ── Auto-run path (arrived from Price Predictor) ──────────────────────
came_from_prediction = bool(st.session_state.get("auto_run_recommendations"))
if came_from_prediction:
    default_locality = st.session_state.get("rec_sector", "Any")
    default_types = st.session_state.get("rec_property_types", [])
    default_bhk = st.session_state.get("rec_bhk", [])
    default_budget = st.session_state.get("rec_budget", (overall_min, overall_max))
    st.session_state["auto_run_recommendations"] = False
else:
    default_locality, default_types, default_bhk = "Any", [], []
    default_budget = (overall_min, overall_max)

if came_from_prediction:
    st.success(
        "✨ Showing recommendations based on your price prediction search. "
        "Adjust below and search again anytime."
    )
    sections = generate_sections(
        locality=default_locality if default_locality in localities else "Any",
        property_types=[t for t in default_types if t in all_types],
        bhk=[b for b in default_bhk if b in all_bhk],
        budget_range=(max(overall_min, default_budget[0]),
                      min(overall_max, default_budget[1])),
        landmark=None,
    )
    render_sections(sections)
    st.caption("Want to refine this search? Use the form below.")

# ── Dataset summary ──────────────────────────────────────────────────
render_kpi_cards([
    ("🏠", str(len(meta)), "Properties"),
    ("📍", str(len(localities) - 1), "Localities"),
    ("💰", f"₹ {overall_min:.1f}–{overall_max:.1f}", "Price Range (Cr)"),
    ("🏷️", str(len(all_types)), "Property Types"),
])

# ── Search form ──────────────────────────────────────────────────────
with st.form("search_form"):
    c1, c2 = st.columns(2)
    with c1:
        locality_choice = st.selectbox(
            "📍 Sector / Area", localities,
            index=localities.index(default_locality) if default_locality in localities else 0,
        )
        type_choice = st.multiselect(
            "🏢 Property type", all_types,
            default=[t for t in default_types if t in all_types],
        )
    with c2:
        bhk_choice = st.multiselect(
            "🛏️ BHK", all_bhk,
            default=[b for b in default_bhk if b in all_bhk],
        )
        landmark_choice = st.selectbox(
            "🗺️ Nearby landmark (optional)", ["Any"] + landmarks,
        )

    budget_choice = st.slider(
        "💰 Budget (₹ Cr)",
        min_value=round(overall_min, 2), max_value=round(overall_max, 2),
        value=(round(default_budget[0], 2), round(default_budget[1], 2)),
        step=0.1,
    )

    submitted = st.form_submit_button("🔍 Get recommendations", type="primary")

if submitted:
    sections = generate_sections(
        locality=locality_choice, property_types=type_choice, bhk=bhk_choice,
        budget_range=budget_choice, landmark=landmark_choice,
    )
    render_gradient_divider()
    render_sections(sections)

render_footer()
