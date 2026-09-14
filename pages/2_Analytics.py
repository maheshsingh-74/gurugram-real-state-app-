"""
Analytics page - market insights and data exploration.

Covers: KPI summary, geo map, area-vs-price scatter, BHK-by-sector pie,
bedroom-vs-price box plot, flat-vs-house distribution, amenities word cloud.

Uses TWO different source files:
  - gurgaon_properties_post_feature_selection_v2.csv (price model training data)
  - appartments.csv (recommender data, for the amenities word cloud)
"""

import ast
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk

from branding import apply_theme, render_hero, render_gradient_divider, render_footer, render_kpi_cards, PLOTLY_LAYOUT

PRICE_CSV = "gurgaon_properties_post_feature_selection_v2.csv"
APPARTMENTS_CSV = "appartments.csv"
SECTOR_COORDS_CSV = "sector_coordinates.csv"

# same outlier rows the price model itself was trained without
OUTLIER_INDICES = [44, 333, 501, 589, 641, 673, 845, 854, 949, 1018, 1104, 1165, 1208,
                   1278, 1294, 1365, 1432, 1508, 1545, 1551, 1558, 1579, 1680, 1705,
                   1762, 1781, 1825, 2049, 2098, 2143, 2153, 2252, 2391, 2434, 2478,
                   2496, 2579, 2670, 2672, 2673, 2676, 2710, 2743, 2775, 2869, 2948,
                   2952, 2969, 3167, 3249, 3275, 3388]

st.set_page_config(page_title="EstateIQ Gurugram | Analytics", page_icon="📊", layout="wide")
apply_theme()

render_hero(
    "📊 Market Analytics",
    "Explore the data behind the price model and recommendations",
    image_path="static/analytics_banner.jpg",
)


@st.cache_data
def load_price_data():
    if not os.path.exists(PRICE_CSV):
        return None
    df = pd.read_csv(PRICE_CSV)
    df = df.drop(index=[i for i in OUTLIER_INDICES if i in df.index]).reset_index(drop=True)
    return df


@st.cache_data
def load_facilities_text():
    if not os.path.exists(APPARTMENTS_CSV):
        return None, None
    df = pd.read_csv(APPARTMENTS_CSV)
    header_dupe = df.apply(lambda r: list(r.astype(str)) == list(df.columns), axis=1)
    df = df.loc[~header_dupe].reset_index(drop=True)

    def parse_facilities(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []

    all_facilities = []
    for val in df["TopFacilities"]:
        all_facilities.extend(parse_facilities(val))
    # underscore multi-word facility names so WordCloud treats each as one token
    text = " ".join(f.replace(" ", "_") for f in all_facilities)
    return text, all_facilities


def styled_plotly(fig, **extra_layout):
    """Apply the shared dark Plotly layout to any figure."""
    layout = {**PLOTLY_LAYOUT, **extra_layout}
    fig.update_layout(**layout)
    return fig


def _show_sector_bar(df):
    """Fallback: top 25 sectors by average price as a horizontal bar chart."""
    sector_avg = df.groupby("sector")["price"].mean().sort_values(ascending=False).head(25).reset_index()
    fig = px.bar(
        sector_avg, x="price", y="sector", orientation="h",
        labels={"price": "Avg Price (Cr)", "sector": ""},
        color="price", color_continuous_scale=["#93c5fd", "#2563eb", "#1e40af"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600,
                      coloraxis_showscale=False)
    styled_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


price_df = load_price_data()

if price_df is None:
    st.error(
        f"Couldn't find **{PRICE_CSV}** in this folder. Copy it here (same file "
        "finalmodel.py trains on) to see the sector/price/BHK charts below."
    )
else:
    # ─── KPI Summary Cards ────────────────────────────────────────────
    n_properties = len(price_df)
    avg_price = price_df["price"].mean()
    median_price = price_df["price"].median()
    n_sectors = price_df["sector"].nunique()
    top_sector = price_df.groupby("sector")["price"].mean().idxmax()
    price_range = f"₹ {price_df['price'].min():.2f} – {price_df['price'].max():.1f} Cr"

    render_kpi_cards([
        ("🏠", "4,500+", "Properties Analyzed"),
        ("📍", str(n_sectors), "Sectors Covered"),
        ("₹", f"{avg_price:.2f} Cr", "Average Price"),
        ("📊", f"{median_price:.2f} Cr", "Median Price"),
    ])

    # ─── 1. Geo Map (PyDeck + Native Streamlit Map) ───────────────────
    st.markdown("### 🗺️ Sector-wise Price Map (Gurugram Market View)")
    st.caption("Interactive geospatial view of average residential property prices across Gurugram sectors.")

    if os.path.exists(SECTOR_COORDS_CSV):
        coords = pd.read_csv(SECTOR_COORDS_CSV)
        sector_avg = price_df.groupby("sector")["price"].mean().round(2).reset_index()
        merged = sector_avg.merge(coords, on="sector", how="inner")

        if merged.empty:
            st.warning(f"{SECTOR_COORDS_CSV} was found but no sector names matched the price data.")
            _show_sector_bar(price_df)
        else:
            merged["price_str"] = merged["price"].apply(lambda p: f"₹ {p:.2f} Cr")
            merged["radius"] = merged["price"].apply(lambda p: max(180, min(1400, int(p * 180))))
            min_p, max_p = merged["price"].min(), merged["price"].max()

            def _get_rgba(p):
                ratio = (p - min_p) / (max_p - min_p) if max_p > min_p else 0.5
                r = int(0 + ratio * 200)
                g = int(120 - ratio * 40)
                b = int(219 - ratio * 100)
                return [r, g, b, 190]

            merged["color"] = merged["price"].apply(_get_rgba)

            map_tab1, map_tab2, map_tab3 = st.tabs([
                "📍 Interactive Price Map",
                "🗺️ Clean Map View",
                "📊 Top Sectors Ranking"
            ])

            with map_tab1:
                try:
                    view_state = pdk.ViewState(
                        latitude=float(merged["latitude"].mean()),
                        longitude=float(merged["longitude"].mean()),
                        zoom=10.3,
                        pitch=15,
                    )
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=merged,
                        get_position=["longitude", "latitude"],
                        get_color="color",
                        get_radius="radius",
                        pickable=True,
                        auto_highlight=True,
                    )
                    deck = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={
                            "html": "<div style='font-family:sans-serif; padding:6px; font-size:13px;'>"
                                    "<b>Sector:</b> {sector}<br/>"
                                    "<b>Avg Price:</b> <span style='color:#38bdf8;'>{price_str}</span></div>",
                            "style": {"backgroundColor": "#041533", "color": "#ffffff", "borderRadius": "6px"}
                        },
                        map_style=None,
                    )
                    st.pydeck_chart(deck, use_container_width=True)
                    st.caption("🔵 Circle size and color intensity reflect average property price. Hover over any sector circle for pricing details.")
                except Exception as map_err:
                    st.warning(f"Interactive PyDeck map couldn't load ({map_err}). Showing standard map below.")
                    st.map(merged, latitude="latitude", longitude="longitude", size="price", color="#0078db")

            with map_tab2:
                st.map(merged, latitude="latitude", longitude="longitude", size="price", color="#0078db")
                st.caption("Standard geospatial plot with dot size scaled by sector price.")

            with map_tab3:
                _show_sector_bar(price_df)
    else:
        st.info(
            f"Add a **{SECTOR_COORDS_CSV}** file (columns: `sector`, `latitude`, `longitude`) "
            "to enable the map. Showing top sectors as a bar chart instead."
        )
        _show_sector_bar(price_df)

    render_gradient_divider()

    # ─── 2. Scatter: area vs price ────────────────────────────────────
    st.markdown("### 📈 Built-up Area vs Price")
    color_col = "property_type" if "property_type" in price_df.columns else None
    fig = px.scatter(
        price_df, x="built_up_area", y="price", color=color_col, opacity=0.55,
        labels={"built_up_area": "Built-up Area (sq ft)", "price": "Price (Cr)"},
        color_discrete_sequence=["#2563eb", "#f59e0b"],
    )
    styled_plotly(fig, height=480)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

    render_gradient_divider()

    # ─── 3. Pie chart: BHK mix ────────────────────────────────────────
    st.markdown("### 🥧 BHK Mix by Sector")
    sector_choice = st.selectbox(
        "Select sector",
        ["All sectors"] + sorted(price_df["sector"].dropna().unique().tolist()),
    )
    subset = price_df if sector_choice == "All sectors" else price_df[price_df["sector"] == sector_choice]
    bhk_counts = subset["bedRoom"].value_counts().sort_index()
    if bhk_counts.empty:
        st.caption("No properties in this sector.")
    else:
        fig = px.pie(
            values=bhk_counts.values,
            names=[f"{int(b)} BHK" for b in bhk_counts.index],
            color_discrete_sequence=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
                                     "#f59e0b", "#8b5cf6", "#06b6d4"],
            hole=0.4,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        styled_plotly(fig, height=420, showlegend=False,
                      title=dict(text=f"BHK Distribution — {sector_choice}",
                                 font=dict(size=15, color="#64748b")))
        st.plotly_chart(fig, use_container_width=True)

    render_gradient_divider()

    # ─── 4. Box plot: bedroom count vs price (Plotly) ─────────────────
    st.markdown("### 📦 Price Spread by Bedroom Count")
    box_df = price_df.copy()
    box_df["bedRoom"] = box_df["bedRoom"].astype(int).astype(str) + " BHK"
    fig = px.box(
        box_df, x="bedRoom", y="price",
        labels={"bedRoom": "Bedrooms", "price": "Price (Cr)"},
        color="bedRoom",
        color_discrete_sequence=["#2563eb", "#f59e0b", "#ef4444", "#8b5cf6",
                                 "#06b6d4", "#3b82f6", "#f97316", "#ec4899"],
    )
    styled_plotly(fig, height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    render_gradient_divider()

    # ─── 5. Distribution: flat vs house (Plotly violin) ───────────────
    st.markdown("### 📉 Price Distribution — Flat vs House")
    if "property_type" in price_df.columns:
        fig = px.violin(
            price_df, x="property_type", y="price", color="property_type",
            box=True, points="outliers",
            labels={"property_type": "Property Type", "price": "Price (Cr)"},
            color_discrete_sequence=["#2563eb", "#f59e0b"],
        )
        styled_plotly(fig, height=440, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No `property_type` column found in the price data.")

    render_gradient_divider()

# ─── 6. Word cloud of amenities ──────────────────────────────────────
st.markdown("### ☁️ Most Common Amenities")
text, all_facilities = load_facilities_text()
if text is None:
    st.error(f"Couldn't find **{APPARTMENTS_CSV}** in this folder for the amenities word cloud.")
else:
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc = WordCloud(
            width=1200, height=400, background_color=None, mode="RGBA",
            colormap="Blues", max_words=80, prefer_horizontal=0.85,
            contour_width=0, contour_color="#2563eb",
        ).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_alpha(0.0)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except ImportError:
        st.warning(
            "Install the `wordcloud` package (`pip install wordcloud`) to see this chart. "
            "Showing a frequency table instead."
        )
        from collections import Counter

        top = Counter(all_facilities).most_common(20)
        freq_df = pd.DataFrame(top, columns=["Amenity", "Count"])
        fig = px.bar(
            freq_df, x="Count", y="Amenity", orientation="h",
            color="Count", color_continuous_scale=["#93c5fd", "#2563eb", "#1e40af"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        styled_plotly(fig, height=500)
    render_gradient_divider()

    # ─── 7. Head-to-Head Sector Comparison (99acres Style) ────────────
    st.markdown("### ⚖️ Head-to-Head Sector Comparison")
    st.caption("Compare two Gurugram sectors side-by-side on average valuation, price/sq ft, and BHK affordability.")

    all_sectors_list = sorted(price_df["sector"].dropna().unique().tolist())
    sec_col1, sec_col2 = st.columns(2)

    with sec_col1:
        default_a = "sector 54" if "sector 54" in all_sectors_list else all_sectors_list[0]
        sec_a = st.selectbox("📍 Select Sector A", all_sectors_list, index=all_sectors_list.index(default_a), key="cmp_sec_a")
    with sec_col2:
        default_b = "sector 102" if "sector 102" in all_sectors_list else (all_sectors_list[1] if len(all_sectors_list) > 1 else all_sectors_list[0])
        sec_b = st.selectbox("📍 Select Sector B", all_sectors_list, index=all_sectors_list.index(default_b), key="cmp_sec_b")

    df_a = price_df[price_df["sector"] == sec_a]
    df_b = price_df[price_df["sector"] == sec_b]

    if not df_a.empty and not df_b.empty:
        avg_a, avg_b = df_a["price"].mean(), df_b["price"].mean()
        med_a, med_b = df_a["price"].median(), df_b["price"].median()

        df_a_sqft = df_a[df_a["built_up_area"] > 0]
        df_b_sqft = df_b[df_b["built_up_area"] > 0]
        pps_a = ((df_a_sqft["price"] * 1e7) / df_a_sqft["built_up_area"]).mean() if not df_a_sqft.empty else 0
        pps_b = ((df_b_sqft["price"] * 1e7) / df_b_sqft["built_up_area"]).mean() if not df_b_sqft.empty else 0

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin:1rem 0;">
            <div class="portal-card" style="padding:1.2rem; text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Average Property Price</div>
                <div style="margin-top:0.4rem; font-size:1.1rem; font-weight:700; color:#38bdf8;">{sec_a.title()}: ₹ {avg_a:.2f} Cr</div>
                <div style="font-size:1.1rem; font-weight:700; color:#f59e0b;">{sec_b.title()}: ₹ {avg_b:.2f} Cr</div>
            </div>
            <div class="portal-card" style="padding:1.2rem; text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Median Rate / Sq Ft</div>
                <div style="margin-top:0.4rem; font-size:1.1rem; font-weight:700; color:#38bdf8;">{sec_a.title()}: ₹ {pps_a:,.0f}</div>
                <div style="font-size:1.1rem; font-weight:700; color:#f59e0b;">{sec_b.title()}: ₹ {pps_b:,.0f}</div>
            </div>
            <div class="portal-card" style="padding:1.2rem; text-align:center;">
                <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Verified Inventory</div>
                <div style="margin-top:0.4rem; font-size:1.1rem; font-weight:700; color:#38bdf8;">{sec_a.title()}: {len(df_a)} listings</div>
                <div style="font-size:1.1rem; font-weight:700; color:#f59e0b;">{sec_b.title()}: {len(df_b)} listings</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # BHK breakdown comparison chart
        bhk_a = df_a.groupby("bedRoom")["price"].mean().reset_index()
        bhk_a["Sector"] = sec_a.title()
        bhk_b = df_b.groupby("bedRoom")["price"].mean().reset_index()
        bhk_b["Sector"] = sec_b.title()

        bhk_comp = pd.concat([bhk_a, bhk_b])
        bhk_comp = bhk_comp[bhk_comp["bedRoom"].isin([1, 2, 3, 4, 5])]
        bhk_comp["BHK"] = bhk_comp["bedRoom"].astype(int).astype(str) + " BHK"

        fig_bhk = px.bar(
            bhk_comp, x="BHK", y="price", color="Sector", barmode="group",
            labels={"price": "Avg Price (₹ Cr)", "BHK": "Configuration"},
            color_discrete_map={sec_a.title(): "#38bdf8", sec_b.title(): "#f59e0b"},
            title=f"<b>BHK Price Comparison</b> — {sec_a.title()} vs {sec_b.title()}"
        )
        styled_plotly(fig_bhk, height=380)
        st.plotly_chart(fig_bhk, use_container_width=True)
    else:
        st.info("Insufficient data for the selected sectors.")

render_footer()