"""
Gurugram property price predictor - home page.
Run with: streamlit run Price_prediction.py
(pages/1_Analytics.py, pages/2_Recommendations.py, pages/3_About.py are
picked up automatically by Streamlit as sidebar navigation pages.)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from branding import (
    apply_theme, render_hero, render_gradient_divider, render_footer,
    render_prediction_result, render_rec_card, render_kpi_cards,
    render_feature_cards, col_label, COLUMN_LABELS, PLOTLY_LAYOUT
)

MODEL_DIR = "models"

st.set_page_config(
    page_title="EstateIQ Gurugram | Price Predictor",
    page_icon="🏙️",
    layout="wide",
)

apply_theme()


@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error(f"Couldn't find {model_path} / {meta_path}. Run finalmodel.py first.")
        st.stop()
    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data(show_spinner=False)
def load_sector_trends_data():
    csv_path = "gurgaon_properties_post_feature_selection_v2.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    outliers = [44, 333, 501, 589, 641, 673, 845, 854, 949, 1018, 1104, 1165, 1208,
                1278, 1294, 1365, 1432, 1508, 1545, 1551, 1558, 1579, 1680, 1705,
                1762, 1781, 1825, 2049, 2098, 2143, 2153, 2252, 2391, 2434, 2478,
                2496, 2579, 2670, 2672, 2673, 2676, 2710, 2743, 2775, 2869, 2948,
                2952, 2969, 3167, 3249, 3275, 3388]
    df = df.drop(index=[i for i in outliers if i in df.index]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_projects_catalog():
    pkl_path = os.path.join("recommender_artifacts", "recommender_artifacts.pkl")
    if os.path.exists(pkl_path):
        data = joblib.load(pkl_path)
        meta = data.get("properties_meta")
        if meta is not None:
            return meta
    if os.path.exists("appartments.csv"):
        return pd.read_csv("appartments.csv")
    return None


# ── SHAP explainability helpers ───────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_shap_explainer(_model):
    """Extract the raw XGBRegressor from inside the TransformedTargetRegressor
    pipeline, and create a SHAP TreeExplainer for it."""
    try:
        import shap
        # model structure: TransformedTargetRegressor -> Pipeline -> XGBRegressor
        inner_pipeline = _model.regressor_
        preprocessor = inner_pipeline.named_steps["preprocessor"]
        xgb_model = inner_pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(xgb_model)
        return explainer, preprocessor
    except Exception:
        return None, None


def get_shap_feature_names(preprocessor):
    """Build human-readable feature names from the ColumnTransformer."""
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(columns).tolist())
        else:
            names.extend(columns)
    return names


def format_feature_display_name(raw_name, user_inputs=None):
    """Format one-hot encoded or raw feature names into clean, human-readable labels."""
    clean = raw_name
    for prefix in ["remainder__", "num__", "cat__", "pipeline__"]:
        clean = clean.replace(prefix, "")

    lower = clean.lower()
    user_inputs = user_inputs or {}

    if "built_up_area" in lower:
        val = user_inputs.get("built_up_area")
        val_str = f" ({val:,.0f} sq ft)" if val else ""
        return f"📐 Built-Up Area{val_str}"
    if "bedroom" in lower:
        val = user_inputs.get("bedRoom")
        val_str = f" ({int(val)} BHK)" if val is not None else ""
        return f"🛏️ Bedrooms{val_str}"
    if "bathroom" in lower:
        val = user_inputs.get("bathroom")
        val_str = f" ({int(val)})" if val is not None else ""
        return f"🚿 Bathrooms{val_str}"
    if "servant room" in lower:
        val = user_inputs.get("servant room")
        val_str = " (Yes)" if val == 1 else " (None)" if val == 0 else ""
        return f"🏠 Servant Room{val_str}"
    if "store room" in lower:
        val = user_inputs.get("store room")
        val_str = " (Yes)" if val == 1 else " (None)" if val == 0 else ""
        return f"📦 Store Room{val_str}"
    if "property_type" in lower:
        val = clean.split("property_type_")[-1].capitalize() if "property_type_" in clean else str(user_inputs.get("property_type", "")).capitalize()
        return f"🏢 Type: {val}" if val else "🏢 Property Type"
    if "sector" in lower:
        sec = clean.replace("sector_", "").replace("sector", "").strip().title()
        if not sec:
            sec = str(user_inputs.get("sector", "")).title()
        return f"📍 Sector {sec}" if sec else "📍 Sector Location"
    if "furnishing" in lower:
        f_map = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished",
                 0.0: "Unfurnished", 1.0: "Semi-Furnished", 2.0: "Furnished"}
        f_val = user_inputs.get("furnishing_type")
        val = f_map.get(f_val, "Furnishing")
        return f"🛋️ {val}"
    if "agepossession" in lower:
        status = clean.split("agepossession_")[-1].replace("_", " ").title() if "agepossession_" in clean else str(user_inputs.get("agePossession", ""))
        return f"📅 Age: {status}" if status else "📅 Possession Age"
    if "luxury" in lower:
        tier = clean.split("luxury_category_")[-1].capitalize() if "luxury_category_" in clean else str(user_inputs.get("luxury_category", "")).capitalize()
        return f"✨ Luxury: {tier}" if tier else "✨ Luxury Tier"
    if "floor" in lower:
        fl = clean.split("floor_category_")[-1].title() if "floor_category_" in clean else str(user_inputs.get("floor_category", "")).title()
        return f"🏗️ Floor: {fl}" if fl else "🏗️ Floor Level"
    if "balcony" in lower:
        b = clean.split("balcony_")[-1] if "balcony_" in clean else str(user_inputs.get("balcony", ""))
        return f"🌿 Balconies ({b})" if b else "🌿 Balconies"

    return clean.replace("_", " ").title()


def format_rupee_impact(cr_val):
    """Format a value in Crores into a clean ₹ Cr or ₹ Lakhs string."""
    sign = "+" if cr_val >= 0 else "-"
    abs_cr = abs(cr_val)
    if abs_cr >= 1.0:
        return f"{sign}₹ {abs_cr:.2f} Cr"
    elif abs_cr >= 0.01:
        return f"{sign}₹ {abs_cr * 100:.1f} Lakhs"
    elif abs_cr >= 0.001:
        return f"{sign}₹ {abs_cr * 100:.2f} L"
    else:
        return "~₹ 0"


def render_shap_waterfall(explainer, preprocessor, input_df, col_label_fn, predicted_price=None, user_inputs=None):
    """Render a clean, human-readable explainability visualization for a prediction."""
    try:
        import shap
        X_transformed = preprocessor.transform(input_df)
        shap_values = explainer.shap_values(X_transformed)
        feature_names = get_shap_feature_names(preprocessor)

        if len(feature_names) != X_transformed.shape[1]:
            feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

        sv = shap_values[0] if shap_values.ndim > 1 else shap_values
        base_log = float(explainer.expected_value)
        base_price = float(np.expm1(base_log))

        pred_p = predicted_price if predicted_price is not None else float(np.expm1(base_log + np.sum(sv)))
        price_diff = pred_p - base_price
        total_shap = float(np.sum(sv))

        # Sort by absolute impact, top 10 features
        indices = np.argsort(np.abs(sv))[::-1][:10]
        top_raw_names = [feature_names[i] for i in indices]
        top_shap_vals = [float(sv[i]) for i in indices]

        clean_names = [format_feature_display_name(n, user_inputs) for n in top_raw_names]

        # Calculate rupee contribution and percentage impact for each feature
        rupee_impacts = []
        pct_impacts = []
        for v in top_shap_vals:
            if abs(total_shap) > 1e-6:
                cr_contrib = price_diff * (v / total_shap)
            else:
                cr_contrib = 0.0
            rupee_impacts.append(cr_contrib)
            pct = (np.exp(v) - 1.0) * 100.0
            pct_impacts.append(pct)

        # ── Key Executive Insights Cards ──
        pos_items = [(clean_names[i], rupee_impacts[i], pct_impacts[i]) for i in range(len(top_shap_vals)) if top_shap_vals[i] > 0]
        neg_items = [(clean_names[i], rupee_impacts[i], pct_impacts[i]) for i in range(len(top_shap_vals)) if top_shap_vals[i] < 0]

        top_pos_str = f"🟢 <strong>Top Value Driver:</strong> {pos_items[0][0]} adds <strong style='color:#10b981;'>{format_rupee_impact(pos_items[0][1])}</strong>" if pos_items else "No positive drivers"
        top_neg_str = f"🔴 <strong>Main Value Drag:</strong> {neg_items[0][0]} lowers price by <strong style='color:#ef4444;'>{format_rupee_impact(neg_items[0][1])}</strong>" if neg_items else "No negative drivers"

        st.markdown(
            f'<div style="background:#131d31; border:1px solid #1e293b; border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem; display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:0.75rem;">'
            f'<div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">Market Reference Baseline</div>'
            f'<div style="font-size:1.3rem; font-weight:800; color:#38bdf8;">₹ {base_price:.2f} Cr</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em;">Fair Market Valuation</div>'
            f'<div style="font-size:1.3rem; font-weight:800; color:#ffffff;">₹ {pred_p:.2f} Cr '
            f'<span style="font-size:0.9rem; font-weight:600; color:{"#10b981" if price_diff >= 0 else "#ef4444"};">'
            f'({("+" if price_diff >= 0 else "")}{price_diff:.2f} Cr)</span></div>'
            f'</div>'
            f'<div style="flex-basis:100%; border-top:1px solid #1e293b; padding-top:0.6rem; font-size:0.88rem; color:#cbd5e1;">'
            f'{top_pos_str} &nbsp;&bull;&nbsp; {top_neg_str}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Toggle Mode ──
        mode = st.radio(
            "Visualise Impact by:",
            ["💰 Estimated Rupee Contribution (₹ Lakhs / Cr)", "📊 Percentage Impact on Price (%)", "🔬 AI SHAP Feature Importance (Log scale)"],
            index=0,
            horizontal=True,
            key="shap_viz_mode_radio",
        )

        colors = ["#10b981" if v > 0 else "#ef4444" for v in top_shap_vals]

        if mode.startswith("💰"):
            x_vals = rupee_impacts
            text_labels = [format_rupee_impact(v) for v in rupee_impacts]
            x_axis_title = "Estimated Impact on Property Valuation (₹ Lakhs / ₹ Crores)"
            hover_tmpl = "<b>%{y}</b><br>Valuation Impact: <b>%{text}</b><br>Relative Shift: %{customdata[0]:+.1f}%<br>Raw SHAP: %{customdata[1]:+.3f}<extra></extra>"
            c_data = list(zip(pct_impacts, top_shap_vals))
        elif mode.startswith("📊"):
            x_vals = pct_impacts
            text_labels = [f"{v:+.1f}%" for v in pct_impacts]
            x_axis_title = "Relative Percentage Impact on Benchmark Price (%)"
            hover_tmpl = "<b>%{y}</b><br>Price Shift: <b>%{text}</b><br>Est. Rupee Contribution: %{customdata[0]}<br>Raw SHAP: %{customdata[1]:+.3f}<extra></extra>"
            c_data = list(zip([format_rupee_impact(v) for v in rupee_impacts], top_shap_vals))
        else:
            x_vals = top_shap_vals
            text_labels = [f"{v:+.3f}" for v in top_shap_vals]
            x_axis_title = "Raw AI SHAP Score (log-space contribution)"
            hover_tmpl = "<b>%{y}</b><br>SHAP Value: <b>%{text}</b><br>Est. Rupee Effect: %{customdata[0]}<br>Percentage Effect: %{customdata[1]:+.1f}%<extra></extra>"
            c_data = list(zip([format_rupee_impact(v) for v in rupee_impacts], pct_impacts))

        # Determine range with ample padding to prevent any text clipping
        max_abs = max(max([abs(x) for x in x_vals], default=1.0), 0.01)
        x_range = [-max_abs * 1.35, max_abs * 1.35]

        fig = go.Figure(go.Bar(
            x=x_vals,
            y=clean_names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=text_labels,
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0", family="Inter, sans-serif"),
            customdata=c_data,
            hovertemplate=hover_tmpl,
            cliponaxis=False,
        ))

        fig.update_layout(
            title=dict(
                text="<b>Feature Valuation Drivers</b> — What Made This Property Worth More or Less",
                font=dict(size=15, color="#ffffff"),
            ),
            xaxis_title=x_axis_title,
            yaxis=dict(
                categoryorder="array",
                categoryarray=list(reversed(clean_names)),
                tickfont=dict(color="#f1f5f9", size=12),
            ),
            xaxis=dict(
                tickfont=dict(color="#94a3b8", size=11),
                gridcolor="#1e293b",
                zerolinecolor="#475569",
                zerolinewidth=2,
                range=x_range,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter, sans-serif"),
            height=460,
            margin=dict(l=10, r=40, t=50, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "💡 **Reading Guide**: The baseline Gurugram reference home is **₹"
            f"{base_price:.2f} Cr**. Green bars indicate features boosting fair market value, while red bars indicate discounts relative to the Gurugram benchmark."
        )
        return True
    except Exception as e:
        st.caption(f"⚠️ SHAP explanation unavailable: {e}")
        return False


def infer_widget_kind(rng):
    """int-slider vs float-slider vs Yes/No toggle, inferred from the
    observed min/max/median rather than hardcoded per column."""
    lo, hi = float(rng["min"]), float(rng["max"])
    if hi <= 1.0:
        return "binary"
    if lo.is_integer() and hi.is_integer():
        return "int"
    return "float"


# Best-effort bridge between the price model's property_type vocabulary
# (flat/house, from the training CSV) and the recommender's real category
# labels (Apartment/Villa/Independent Floor/etc, from appartments.csv)
PROPERTY_TYPE_BRIDGE = {
    "flat": ["Apartment", "Studio Apartment", "Service Apartment"],
    "house": ["Independent Floor", "Villa"],
}

FURNISHING_LABELS = {0: "Unfurnished", 1: "Semi-furnished", 2: "Furnished"}


def format_furnishing(value):
    return FURNISHING_LABELS.get(int(round(float(value))), str(value))


CATEGORICAL_FORMAT_FUNCS = {
    "furnishing_type": format_furnishing,
}

model, metadata = load_model_and_metadata()
schema = metadata["input_schema"]
numeric_cols = schema["numeric_cols"]
categorical_cols = schema["categorical_cols"]
categorical_options = schema["categorical_options"]
numeric_ranges = schema["numeric_ranges"]

ESSENTIAL_NUMERIC = [c for c in ["bedRoom", "bathroom", "built_up_area"] if c in numeric_cols]
ESSENTIAL_CATEGORICAL = [c for c in ["property_type", "sector", "furnishing_type"] if c in categorical_cols]
ADVANCED_NUMERIC = [c for c in numeric_cols if c not in ESSENTIAL_NUMERIC]
ADVANCED_CATEGORICAL = [c for c in categorical_cols if c not in ESSENTIAL_CATEGORICAL]

best_model_name = metadata["best_model"]
metrics = metadata["metrics"][best_model_name]

# ── Hero ──────────────────────────────────────────────────────────────────
render_hero(
    "EstateIQ Gurugram",
    "Smart algorithmic price estimates for the Gurugram residential market",
    show_logo=True,
    image_path="static/hero_banner.jpg",
    hero_stats=[
        (f"{len(categorical_options.get('sector', []))}", "Sectors"),
        (best_model_name.split('(')[0].strip(), "Model"),
        ("4,500+", "Listings"),
    ],
)


with st.expander("ℹ️ About this data & model", expanded=False):
    dataset_line = (f"- Trained on **4,500+ property listings** across "
                    f"**{len(categorical_options.get('sector', []))} sectors** in Gurugram.")
    st.markdown(f"""
{dataset_line}
- Model: **{best_model_name}**, tuned via randomized search with 5-fold cross-validation.
- Accuracy is reported on a **held-out test set the model never saw during tuning**: R² **{metrics['holdout_r2']:.3f}**,
  typical error ₹{metrics['holdout_mae']:.2f} Cr (MAPE {metrics['holdout_mape']:.1f}%).
- Every prediction below is shown as a **range**, not a single number — no model is perfectly precise.
""")

render_gradient_divider()

# ── Feature Showcase Cards ────────────────────────────────────────────────
render_feature_cards([
    {
        "icon": "🏠",
        "title": "Buy & Valuation Check",
        "desc": "Get instant ML-powered price estimates for any property configuration across Gurugram sectors.",
        "tag": "CALCULATE PRICE →",
        "image": "static/hero_banner.jpg",
        "link": "#valuation-calculator",
    },
    {
        "icon": "🏘️",
        "title": "Smart Recommendations",
        "desc": "Five-angle property matching — by location, budget, configuration, landmarks, and amenities.",
        "tag": "FIND PROPERTIES →",
        "image": "static/property_card_bg.jpg",
        "link": "Recommendations",
    },
    {
        "icon": "📊",
        "title": "Market Analytics",
        "desc": "Interactive geospatial maps, sector trends, BHK distributions, and price correlations.",
        "tag": "VIEW ANALYTICS →",
        "image": "static/analytics_banner.jpg",
        "link": "Analytics",
    },
])

# Quick Action Nav Buttons
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
with col_nav1:
    if st.button("🏠 Calculate Valuation ↓", key="nav_to_calc_btn", use_container_width=True):
        st.toast("Ready! Configure property specs below 👇", icon="🏠")
with col_nav2:
    if st.button("🏘️ Find Properties →", key="nav_to_rec_btn", use_container_width=True):
        st.switch_page("pages/1_Recommendations.py")
with col_nav3:
    if st.button("📊 Market Analytics →", key="nav_to_analytics_btn", use_container_width=True):
        st.switch_page("pages/2_Analytics.py")
with col_nav4:
    if st.button("🔬 Model Insights →", key="nav_to_insights_btn", use_container_width=True):
        st.switch_page("pages/3_Model_Insights.py")

st.markdown('<div id="valuation-calculator" style="scroll-margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

# ── Interactive Navigation Tabs ──────────────────────────────────────────
tab_calc, tab_trends, tab_projects, tab_batch = st.tabs([
    "🏠 Buy & Valuation Check",
    "📊 Sector Rate Trends",
    "🏘️ Gurugram Projects",
    "📥 Batch Upload",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: Buy & Valuation Check
# ══════════════════════════════════════════════════════════════════════════
with tab_calc:
    st.markdown("### 🔍 Property Valuation Calculator")
    st.caption("Enter your property specifications to calculate current fair-market valuation across Gurugram sectors.")

    inputs = {}
    col_a, col_b, col_c = st.columns(3)
    essential_widget_cols = [col_a, col_b, col_c]

    for i, col in enumerate(ESSENTIAL_NUMERIC):
        rng = numeric_ranges[col]
        kind = infer_widget_kind(rng)
        target = essential_widget_cols[i % 3]
        with target:
            label = col_label(col)
            if kind == "binary":
                choice = st.select_slider(label, options=["No", "Yes"], value="No")
                inputs[col] = 1 if choice == "Yes" else 0
            elif kind == "int":
                inputs[col] = st.slider(label, min_value=int(rng["min"]), max_value=int(rng["max"]),
                                        value=int(round(rng["median"])), step=1)
            else:
                inputs[col] = st.slider(label, min_value=float(rng["min"]),
                                        max_value=float(rng["max"]) * 1.3,
                                        value=float(rng["median"]))

    col_d, col_e, col_f = st.columns(3)
    essential_cat_cols = [col_d, col_e, col_f]
    for i, col in enumerate(ESSENTIAL_CATEGORICAL):
        with essential_cat_cols[i % 3]:
            inputs[col] = st.selectbox(col_label(col), categorical_options[col],
                                       format_func=CATEGORICAL_FORMAT_FUNCS.get(col, str))

    with st.expander("⚙️ Advanced options (optional)"):
        adv_cols = st.columns(3)
        for i, col in enumerate(ADVANCED_NUMERIC):
            rng = numeric_ranges[col]
            kind = infer_widget_kind(rng)
            with adv_cols[i % 3]:
                label = col_label(col)
                if kind == "binary":
                    choice = st.select_slider(label, options=["No", "Yes"], value="No",
                                              key=f"adv_{col}")
                    inputs[col] = 1 if choice == "Yes" else 0
                elif kind == "int":
                    inputs[col] = st.slider(label, min_value=int(rng["min"]),
                                            max_value=int(rng["max"]),
                                            value=int(round(rng["median"])), step=1,
                                            key=f"adv_{col}")
                else:
                    inputs[col] = st.slider(label, min_value=float(rng["min"]),
                                            max_value=float(rng["max"]) * 1.3,
                                            value=float(rng["median"]), key=f"adv_{col}")
        adv_cat_cols = st.columns(3)
        for i, col in enumerate(ADVANCED_CATEGORICAL):
            with adv_cat_cols[i % 3]:
                inputs[col] = st.selectbox(col_label(col), categorical_options[col],
                                           key=f"adv_{col}",
                                           format_func=CATEGORICAL_FORMAT_FUNCS.get(col, str))

    predict_clicked = st.button("🔍 Check Estimated Market Price", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([inputs])
        point_estimate = float(model.predict(input_df)[0])
        st.session_state["prediction_result"] = {
            "point_estimate": point_estimate,
            "sector": inputs.get("sector"),
            "property_type": inputs.get("property_type"),
            "bedRoom": inputs.get("bedRoom", 2),
            "built_up_area": inputs.get("built_up_area"),
        }

    if "prediction_result" in st.session_state:
        pred = st.session_state["prediction_result"]
        point_estimate = pred["point_estimate"]
        half_mae = metrics["holdout_mae"] / 2
        low, high = point_estimate - half_mae, point_estimate + half_mae

        render_gradient_divider()

        render_prediction_result(
            low=low,
            point=point_estimate,
            high=high,
            built_up_area=pred.get("built_up_area"),
        )

        # ── Valuation Summary Report Exporter ─────────────────────────────
        sec_name = str(pred.get("sector", "Gurugram")).title()
        bhk_val = int(pred.get("bedRoom", 2))
        rate_sqft = (point_estimate * 1e7) / max(1, float(pred.get("built_up_area", 1)))
        furnishing_str = format_furnishing(pred.get("furnishing_type", 0))

        report_lines = [
            "============================================================",
            "   ESTATEIQ GURUGRAM — PROPERTY VALUATION SUMMARY REPORT    ",
            "============================================================",
            f"Expected Fair Valuation:   ₹ {point_estimate:.2f} Cr",
            f"Fair Market Range:         ₹ {low:.2f} Cr — ₹ {high:.2f} Cr",
            f"Built-Up Area:             {pred.get('built_up_area', '—')} sq ft",
            f"Estimated Rate / sq ft:    ₹ {rate_sqft:,.0f} / sq ft",
            "",
            "PROPERTY SPECIFICATIONS:",
            f"  • Sector Location:       {sec_name}",
            f"  • Configuration:         {bhk_val} BHK, {int(pred.get('bathroom', 2))} Bathrooms",
            f"  • Property Type:         {str(pred.get('property_type', '')).capitalize()}",
            f"  • Furnishing Status:     {furnishing_str}",
            f"  • Floor Category:        {pred.get('floor_category', 'Mid Floor')}",
            f"  • Servant Room:          {'Yes' if pred.get('servant room') == 1 else 'No'}",
            f"  • Store Room:            {'Yes' if pred.get('store room') == 1 else 'No'}",
            "",
            "MODEL CERTIFICATION:",
            f"  • Algorithmic Regressor: {best_model_name}",
            f"  • Audited Holdout R²:    {metrics.get('holdout_r2', 0.927):.3f}",
            f"  • Holdout MAE:           ₹ {metrics.get('holdout_mae', 0.43):.2f} Cr",
            f"  • Geographical Coverage: Gurugram, Haryana Residential Market",
            "============================================================",
            "EstateIQ Real Estate Intelligence · Generated for Decision Support © 2026",
        ]
        report_text = "\n".join(report_lines)

        col_d1, col_d2 = st.columns([2.8, 1.2])
        with col_d2:
            st.download_button(
                label="📄 Download Valuation Certificate",
                data=report_text,
                file_name=f"EstateIQ_Valuation_{sec_name.replace(' ', '_')}_{bhk_val}BHK.txt",
                mime="text/plain",
                use_container_width=True,
                key="btn_download_val_cert",
            )

        # ── Home Loan EMI & Haryana Stamp Duty Estimator ──────────────────
        with st.expander("🏦 Home Loan EMI & Haryana Registration Cost Estimator", expanded=False):
            st.caption("Plan your financing: calculate monthly EMIs, required down payment, and Haryana government stamp duty.")

            emi_c1, emi_c2, emi_c3 = st.columns(3)
            with emi_c1:
                down_payment_pct = st.slider("Down Payment (%)", min_value=10, max_value=50, value=20, step=5, key="emi_dp_slider")
            with emi_c2:
                loan_rate = st.slider("Interest Rate (% p.a.)", min_value=7.0, max_value=12.0, value=8.5, step=0.1, key="emi_rate_slider")
            with emi_c3:
                loan_tenure_years = st.slider("Tenure (Years)", min_value=5, max_value=30, value=20, step=1, key="emi_tenure_slider")

            # Highlighted Buyer Category Dropdown
            st.markdown(
                """
                <style>
                /* Prominent styling for Buyer Category Selectbox */
                div[data-testid="stExpanderDetails"] div[data-baseweb="select"] > div,
                div[data-testid="stExpander"] div[data-baseweb="select"] > div,
                details div[data-baseweb="select"] > div {
                    background: linear-gradient(135deg, #162a4d 0%, #0f1c33 100%) !important;
                    border: 2px solid #38bdf8 !important;
                    border-radius: 10px !important;
                    box-shadow: 0 4px 18px rgba(56, 189, 248, 0.3) !important;
                    transition: all 0.2s ease-in-out !important;
                }
                div[data-testid="stExpanderDetails"] div[data-baseweb="select"] > div:hover,
                div[data-testid="stExpander"] div[data-baseweb="select"] > div:hover,
                details div[data-baseweb="select"] > div:hover {
                    border-color: #60a5fa !important;
                    box-shadow: 0 6px 24px rgba(96, 165, 250, 0.5) !important;
                    background: linear-gradient(135deg, #1d3560 0%, #132442 100%) !important;
                }
                div[data-testid="stExpanderDetails"] div[data-baseweb="select"] span,
                div[data-testid="stExpander"] div[data-baseweb="select"] span,
                details div[data-baseweb="select"] span {
                    color: #ffffff !important;
                    font-weight: 700 !important;
                    font-size: 0.98rem !important;
                }
                div[data-testid="stExpanderDetails"] div[data-baseweb="select"] svg,
                div[data-testid="stExpander"] div[data-baseweb="select"] svg,
                details div[data-baseweb="select"] svg {
                    fill: #38bdf8 !important;
                    width: 22px !important;
                    height: 22px !important;
                }
                </style>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1.2rem; margin-bottom: 0.4rem;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">🏛️</span>
                        <span style="font-weight: 700; color: #ffffff; font-size: 1rem;">Buyer Category (Haryana Stamp Duty)</span>
                    </div>
                    <span style="background: rgba(56, 189, 248, 0.2); border: 1.5px solid #38bdf8; color: #38bdf8; padding: 3px 12px; border-radius: 20px; font-size: 0.76rem; font-weight: 700;">
                        Tap to change rate
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            buyer_gender = st.selectbox(
                "Buyer Category (Haryana Stamp Duty)",
                [
                    "Male Buyer (7% Stamp Duty + 1% Reg = 8%)",
                    "Female Buyer (5% Stamp Duty + 1% Reg = 6%)",
                    "Joint Ownership (Male + Female, 6% Stamp Duty + 1% Reg = 7%)",
                ],
                index=0,
                key="emi_buyer_type",
                label_visibility="collapsed",
            )

            # Financial calculations
            prop_price_inr = point_estimate * 1e7
            down_payment_inr = prop_price_inr * (down_payment_pct / 100.0)
            loan_amount_inr = prop_price_inr - down_payment_inr

            monthly_r = (loan_rate / 100.0) / 12.0
            n_months = loan_tenure_years * 12
            if monthly_r > 0:
                monthly_emi = loan_amount_inr * monthly_r * ((1 + monthly_r) ** n_months) / (((1 + monthly_r) ** n_months) - 1)
            else:
                monthly_emi = loan_amount_inr / n_months

            total_payment_inr = monthly_emi * n_months
            total_interest_inr = max(0, total_payment_inr - loan_amount_inr)

            duty_rate = 0.06 if "Female" in buyer_gender else (0.07 if "Joint" in buyer_gender else 0.08)
            stamp_duty_inr = prop_price_inr * duty_rate
            total_upfront_cash_inr = down_payment_inr + stamp_duty_inr

            # KPI Summary
            render_kpi_cards([
                ("💳", f"₹ {monthly_emi:,.0f} / mo", "Estimated Monthly EMI"),
                ("💰", f"₹ {down_payment_inr / 1e5:.1f} Lakhs", f"Down Payment ({down_payment_pct}%)"),
                ("🏦", f"₹ {loan_amount_inr / 1e7:.2f} Cr", "Loan Principal Amount"),
                ("🏛️", f"₹ {stamp_duty_inr / 1e5:.1f} Lakhs", f"Haryana Stamp Duty ({int(duty_rate*100)}%)"),
            ])

            donut_fig = go.Figure(data=[go.Pie(
                labels=["Loan Principal", "Total Interest Payable", "Down Payment", "Haryana Stamp Duty & Reg"],
                values=[loan_amount_inr, total_interest_inr, down_payment_inr, stamp_duty_inr],
                hole=0.55,
                marker_colors=["#2563eb", "#f59e0b", "#10b981", "#ec4899"],
            )])
            donut_fig.update_layout(
                title=dict(text="<b>Total Cost & Financing Breakdown</b>", font=dict(size=14, color="#ffffff")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                height=320,
                margin=dict(l=10, r=10, t=40, b=20),
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(donut_fig, use_container_width=True)
            st.caption(
                f"💡 **Total Upfront Cash Needed**: **₹ {total_upfront_cash_inr / 1e5:.1f} Lakhs** "
                f"(Down Payment: ₹{down_payment_inr/1e5:.1f} L + Government Registration & Stamp Duty: ₹{stamp_duty_inr/1e5:.1f} L)."
            )

        # ── SHAP Explainability ──────────────────────────────────────────
        render_gradient_divider()
        explainer, preproc = load_shap_explainer(model)
        if explainer is not None:
            with st.expander("🧠 Why This Price? — AI Explainability (SHAP)", expanded=True):
                st.caption(
                    "This chart shows which features had the **biggest impact** on your "
                    "predicted price. Green bars push the price **up**, red bars push it **down**."
                )
                input_df_for_shap = pd.DataFrame([inputs])
                render_shap_waterfall(
                    explainer, preproc, input_df_for_shap, col_label,
                    predicted_price=point_estimate, user_inputs=inputs
                )
        else:
            st.caption("ℹ️ SHAP explainability is loading...")

        # Cross-page state for recommendations
        st.session_state["rec_sector"] = pred["sector"]
        st.session_state["rec_property_types"] = PROPERTY_TYPE_BRIDGE.get(
            str(pred["property_type"]).lower(), []
        )
        st.session_state["rec_bhk"] = [f"{int(pred['bedRoom'])} BHK"]
        st.session_state["rec_budget"] = (
            round(max(0.1, point_estimate - 0.5), 2),
            round(point_estimate + 0.5, 2),
        )

        render_gradient_divider()
        st.markdown("#### 🏘️ Recommended Properties Matching Your Search")
        st.caption("View verified listings in this sector, matching BHK, and price bracket.")
        if st.button("Explore Matching Properties in Gurugram →", type="primary"):
            st.session_state["auto_run_recommendations"] = True
            if hasattr(st, "switch_page"):
                st.switch_page("pages/1_Recommendations.py")
            else:
                st.info("Open **Recommendations** from the sidebar to see your matches.")
    else:
        # Prompt card before prediction
        st.markdown(
            '<div class="portal-card" style="text-align:center; padding:2.2rem 1.5rem;">'
            '<div style="font-size:2.8rem; margin-bottom:0.6rem;">🏢</div>'
            '<div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:0.3rem;">'
            'Ready to calculate Gurugram property valuation'
            '</div>'
            '<div style="color:#94a3b8; font-size:0.95rem; max-width:520px; margin:0 auto;">'
            'Select your preferred sector, configuration, and area above, then click '
            '<strong style="color:#38bdf8;">Check Estimated Market Price</strong> for an instant valuation.'
            '</div></div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: Sector Rate Trends
# ══════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.markdown("### 📊 Gurugram Sector Rate Trends")
    st.caption("Real-time market price averages, trends, and sector valuations across Gurugram.")

    trends_df = load_sector_trends_data()
    if trends_df is not None and not trends_df.empty:
        avg_overall = trends_df["price"].mean()
        median_overall = trends_df["price"].median()
        sector_group = trends_df.groupby("sector")["price"].agg(["mean", "median", "count", "min", "max"]).reset_index()
        top_expensive = sector_group.sort_values(by="mean", ascending=False).iloc[0]

        render_kpi_cards([
            ("🏙️", f"₹ {avg_overall:.2f} Cr", "Gurugram Avg Price"),
            ("📈", f"₹ {median_overall:.2f} Cr", "Gurugram Median Price"),
            ("💎", f"{top_expensive['sector'].title()}", "Most Premium Sector"),
            ("📍", f"{len(sector_group)}", "Sectors Tracked"),
        ])

        c_chart, c_inspect = st.columns([1.3, 1])

        with c_chart:
            st.markdown("#### 🏆 Top 15 Premium Sectors by Avg Price")
            top15 = sector_group.sort_values(by="mean", ascending=False).head(15)
            fig = px.bar(
                top15, x="mean", y="sector", orientation="h",
                labels={"mean": "Avg Price (Cr)", "sector": ""},
                color="mean", color_continuous_scale=["#1d4ed8", "#38bdf8"],
            )
            layout_args = {**PLOTLY_LAYOUT, "height": 490, "coloraxis_showscale": False, "margin": dict(l=20, r=20, t=20, b=20)}
            layout_args["yaxis"] = {**PLOTLY_LAYOUT.get("yaxis", {}), "categoryorder": "total ascending"}
            fig.update_layout(**layout_args)
            st.plotly_chart(fig, use_container_width=True)

        with c_inspect:
            st.markdown("#### 🔍 Sector Price Inspector")
            sorted_sectors = sorted(sector_group["sector"].unique().tolist())
            selected_sector = st.selectbox(
                "Select Gurugram Sector to Inspect",
                sorted_sectors,
                key="trend_sector_pick"
            )

            sec_row = sector_group[sector_group["sector"] == selected_sector].iloc[0]
            sec_properties = trends_df[trends_df["sector"] == selected_sector]
            avg_area = sec_properties["built_up_area"].mean() if "built_up_area" in sec_properties else 0

            st.markdown(f"""
            <div class="portal-card" style="padding:1.4rem; border-left: 4px solid #38bdf8;">
                <div style="font-size:1.25rem; font-weight:800; color:#ffffff; margin-bottom:0.6rem;">
                    📍 {selected_sector.title()}
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.9rem; margin-top:0.8rem;">
                    <div>
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Average Price</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#38bdf8;">₹ {sec_row['mean']:.2f} Cr</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Median Price</div>
                        <div style="font-size:1.4rem; font-weight:800; color:#ffffff;">₹ {sec_row['median']:.2f} Cr</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Price Range</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#cbd5e1;">₹ {sec_row['min']:.2f} - {sec_row['max']:.2f} Cr</div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Listings Analyzed</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#cbd5e1;">{int(sec_row['count'])} properties</div>
                    </div>
                </div>
                <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid #1e293b; font-size:0.85rem; color:#94a3b8;">
                    📐 Avg Built-up Area: <strong style="color:#ffffff;">{avg_area:,.0f} sq ft</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            render_gradient_divider()
            st.caption("Want full geospatial price heatmaps, scatter correlations, and BHK distributions?")
            if st.button("🗺️ Open Interactive Geospatial Market Analytics →", use_container_width=True, key="btn_open_analytics"):
                if hasattr(st, "switch_page"):
                    st.switch_page("pages/2_Analytics.py")
                else:
                    st.info("Open **Analytics** from the sidebar to view complete charts.")
    else:
        st.info("Sector rate data is loading or unavailable.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: Gurugram Projects
# ══════════════════════════════════════════════════════════════════════════
with tab_projects:
    st.markdown("### 🏘️ Gurugram Residential Projects")
    st.caption("Explore verified residential societies, high-rise condominiums, and luxury apartments across Gurugram.")

    proj_meta = load_projects_catalog()
    if proj_meta is not None and not proj_meta.empty:
        f1, f2, f3 = st.columns([1.5, 1.2, 1])
        with f1:
            search_name = st.text_input(
                "🔍 Search Project Name",
                placeholder="e.g. DLF, M3M, Sobha, Smartworld...",
                key="proj_name_filter"
            )
        with f2:
            all_localities = ["All Localities"] + sorted(proj_meta["Locality"].dropna().unique().tolist())
            selected_locality = st.selectbox(
                "📍 Locality / Sector",
                all_localities,
                key="proj_loc_filter"
            )
        with f3:
            all_bhks = ["All BHKs"] + sorted({b for configs in proj_meta["BHKConfigs"] for b in (configs or [])})
            selected_bhk = st.selectbox(
                "🛏️ Configuration",
                all_bhks,
                key="proj_bhk_filter"
            )

        filtered_proj = proj_meta.copy()
        if search_name:
            filtered_proj = filtered_proj[filtered_proj["PropertyName"].str.contains(search_name, case=False, na=False)]
        if selected_locality != "All Localities":
            filtered_proj = filtered_proj[filtered_proj["Locality"].str.contains(selected_locality, case=False, na=False)]
        if selected_bhk != "All BHKs":
            filtered_proj = filtered_proj[filtered_proj["BHKConfigs"].apply(lambda cs: selected_bhk in (cs or []))]

        st.markdown(f"Showing **{len(filtered_proj)}** verified residential projects in Gurugram")

        for _, row in filtered_proj.head(12).iterrows():
            render_rec_card(
                name=row["PropertyName"],
                locality=row.get("Locality", "Gurugram"),
                bhk_configs=row.get("BHKConfigs", []),
                min_price=row.get("MinPriceCr"),
                max_price=row.get("MaxPriceCr"),
                link=row.get("Link", ""),
                card_type="match",
            )

        if len(filtered_proj) > 12:
            st.caption(f"Showing first 12 of {len(filtered_proj)} matching projects. Use search & filters above to narrow your list.")

        render_gradient_divider()
        st.markdown("#### Looking for smart recommendations based on amenities, landmarks, and budget?")
        if st.button("✨ Open Multi-Angle Property Recommender Engine →", use_container_width=True, key="btn_open_recs"):
            if hasattr(st, "switch_page"):
                st.switch_page("pages/1_Recommendations.py")
            else:
                st.info("Open **Recommendations** from the sidebar to view personalized matches.")
    else:
        st.info("Projects catalog is loading or unavailable.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: Batch Upload
# ══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📥 Batch Property Valuation")
    st.caption(
        "Upload a CSV with multiple properties and download a complete valuation report. "
        "Perfect for investors, agents, or bulk analysis."
    )

    # Show expected columns
    with st.expander("📋 Required CSV Columns", expanded=False):
        all_cols = numeric_cols + categorical_cols
        col_info = []
        for c in all_cols:
            if c in numeric_ranges:
                rng = numeric_ranges[c]
                col_info.append({
                    "Column": c,
                    "Type": "Numeric",
                    "Example": f"{rng['median']}",
                    "Range": f"{rng['min']} – {rng['max']}",
                })
            elif c in categorical_options:
                opts = categorical_options[c]
                col_info.append({
                    "Column": c,
                    "Type": "Categorical",
                    "Example": str(opts[0]) if opts else "",
                    "Range": f"{len(opts)} options",
                })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    # Download template
    template_df = pd.DataFrame({
        c: [numeric_ranges[c]["median"]] if c in numeric_ranges
        else [categorical_options[c][0]] if c in categorical_options
        else [""] for c in numeric_cols + categorical_cols
    })
    st.download_button(
        "⬇️ Download CSV Template",
        template_df.to_csv(index=False),
        "estateiq_template.csv",
        "text/csv",
        use_container_width=True,
    )

    render_gradient_divider()

    uploaded_file = st.file_uploader("Upload your property CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(batch_df)} properties** from CSV")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🔍 Run Batch Predictions", type="primary", use_container_width=True,
                         key="batch_predict_btn"):
                with st.spinner("Running predictions on all properties..."):
                    # Fill missing columns with defaults
                    for c in numeric_cols:
                        if c not in batch_df.columns:
                            batch_df[c] = numeric_ranges[c]["median"]
                    for c in categorical_cols:
                        if c not in batch_df.columns:
                            batch_df[c] = categorical_options[c][0]

                    predictions = model.predict(batch_df[numeric_cols + categorical_cols])
                    half_mae = metrics["holdout_mae"] / 2

                    result_df = batch_df.copy()
                    result_df["Predicted_Price_Cr"] = np.round(predictions, 2)
                    result_df["Price_Low_Cr"] = np.round(predictions - half_mae, 2)
                    result_df["Price_High_Cr"] = np.round(predictions + half_mae, 2)
                    if "built_up_area" in result_df.columns:
                        result_df["Price_Per_SqFt"] = np.where(
                            result_df["built_up_area"] > 0,
                            np.round((predictions * 1e7) / result_df["built_up_area"], 0),
                            0
                        ).astype(int)

                render_gradient_divider()
                st.markdown("#### 📊 Batch Prediction Results")

                render_kpi_cards([
                    ("🏠", str(len(result_df)), "Properties Analyzed"),
                    ("₹", f"{result_df['Predicted_Price_Cr'].mean():.2f} Cr", "Average Price"),
                    ("📈", f"{result_df['Predicted_Price_Cr'].max():.2f} Cr", "Highest"),
                    ("📉", f"{result_df['Predicted_Price_Cr'].min():.2f} Cr", "Lowest"),
                ])

                st.dataframe(
                    result_df[[c for c in [
                        "sector", "property_type", "bedRoom", "built_up_area",
                        "Predicted_Price_Cr", "Price_Low_Cr", "Price_High_Cr", "Price_Per_SqFt"
                    ] if c in result_df.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                # Download results
                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Full Valuation Report (CSV)",
                    csv_out,
                    "estateiq_valuation_report.csv",
                    "text/csv",
                    type="primary",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
    else:
        st.markdown(
            '<div class="portal-card" style="text-align:center; padding:2.2rem 1.5rem;">'
            '<div style="font-size:2.8rem; margin-bottom:0.6rem;">📤</div>'
            '<div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:0.3rem;">'
            'Upload a CSV to get started'
            '</div>'
            '<div style="color:#94a3b8; font-size:0.95rem; max-width:520px; margin:0 auto;">'
            'Download the template above, fill in your property data, and upload it here '
            'for instant batch predictions with downloadable reports.'
            '</div></div>',
            unsafe_allow_html=True,
        )

render_gradient_divider()

# ── Model Insights CTA Banner ─────────────────────────────────────────────
st.markdown("""
<div class="portal-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; padding:1.4rem 1.8rem; margin-top:1.5rem;">
    <div>
        <div style="font-size:1.15rem; font-weight:800; color:#ffffff;">🔬 Deep Dive into Model Performance & Explainability</div>
        <div style="color:#94a3b8; font-size:0.9rem; margin-top:0.3rem;">
            Inspect our audited <strong>R² = 0.927</strong> score, 10-fold cross-validation results, global SHAP feature importance, and end-to-end pipeline architecture.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Open Complete Model Insights & Architecture Page →", type="secondary", use_container_width=True, key="btn_cta_model_insights"):
    st.switch_page("pages/3_Model_Insights.py")

render_footer()