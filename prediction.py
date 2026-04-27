"""
views/prediction.py
--------------------
DemandLens — ML Demand Prediction (CO3).
Models: Linear Regression, Decision Tree, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import joblib

from utils.helper_functions import (
    apply_dark_chart_style,
    section_header,
    chart_label_html,
    kpi_html,
    format_number,
    COLORS,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SYLLABUS_MODELS = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
]


def _load_models():
    models, scaler, feat_cols = {}, None, None
    try:
        scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        feat_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        for n in SYLLABUS_MODELS:
            safe = n.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(MODELS_DIR, f"{safe}.pkl")
            if os.path.exists(path):
                models[n] = joblib.load(path)
    except Exception as e:
        st.error(f"Model load error: {e}. Please train models first.")
    return models, scaler, feat_cols


def _encode_col(df_ref, col, val):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df_ref[col].astype(str))
    try:
        return int(le.transform([val])[0])
    except Exception:
        return 0


# Models that require scaled input
SCALED_MODELS = {"Linear Regression"}


def show(df: pd.DataFrame):

    st.markdown(section_header(
        "ML DEMAND PREDICTION",
        "Predict units sold for any product configuration (CO3)"
    ), unsafe_allow_html=True)

    models, scaler, feat_cols = _load_models()
    if not models:
        st.warning("No trained models found. Go to the Performance tab and click Train Models.")
        return

    left, right = st.columns([1, 2.6], gap="medium")

    with left:
        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
PRODUCT PARAMETERS</div>
""", unsafe_allow_html=True)

        category    = st.selectbox("Product Category",    df["Category"].unique())
        price       = st.slider("Price (Rs)",             10.0, 5000.0, 1000.0, step=50.0)
        discount    = st.slider("Discount (%)",            0, 40, 10, step=5)
        comp_price  = st.slider("Competitor Price (Rs)",  10.0, 5000.0, 1000.0, step=50.0)
        inventory   = st.slider("Inventory Level",         50, 500, 250, step=10)
        demand_fc   = st.slider("Demand Forecast",         0.0, 500.0, 130.0, step=5.0)
        holiday     = st.selectbox("Holiday/Promotion",    [0, 1], format_func=lambda x: "Yes" if x else "No")
        month       = st.slider("Month",                   1, 12, 6)
        week        = st.slider("Week of Year",            1, 52, 26)
        quarter     = st.selectbox("Quarter",              [1, 2, 3, 4])
        day_of_week = st.slider("Day of Week (0=Mon)",     0, 6, 2)
        region      = st.selectbox("Region",               df["Region"].unique())
        weather     = st.selectbox("Weather Condition",    df["Weather Condition"].unique())
        seasonality = st.selectbox("Seasonality",          df["Seasonality"].unique())
        st.markdown("</div>", unsafe_allow_html=True)

        model_choice = st.selectbox("Model", [m for m in SYLLABUS_MODELS if m in models])
        run = st.button("Predict Demand", use_container_width=True)

    with right:
        if not run:
            st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:80px 20px;text-align:center;">
<div style="color:{COLORS['muted']};font-size:13px;font-family:JetBrains Mono,monospace;">
Configure parameters and click
<span style="color:{COLORS['primary']};font-weight:600;">Predict Demand</span>
to generate forecast.
</div>
</div>
""", unsafe_allow_html=True)
            return

        inp = {
            "Price":                 price,
            "Discount_frac":         discount / 100,
            "Competitor Pricing":    comp_price,
            "Inventory Level":       inventory,
            "Demand Forecast":       demand_fc,
            "Holiday/Promotion":     holiday,
            "Month":                 month,
            "Week":                  week,
            "Quarter":               quarter,
            "DayOfWeek":             day_of_week,
            "Category_enc":          _encode_col(df, "Category",          category),
            "Region_enc":            _encode_col(df, "Region",            region),
            "Weather Condition_enc": _encode_col(df, "Weather Condition", weather),
            "Seasonality_enc":       _encode_col(df, "Seasonality",       seasonality),
        }

        row   = pd.DataFrame([inp])[feat_cols]
        model = models[model_choice]

        if model_choice in SCALED_MODELS:
            pred = float(model.predict(scaler.transform(row))[0])
        else:
            pred = float(model.predict(row)[0])
        pred = max(0, pred)

        eff_price = price * (1 - discount / 100)
        revenue   = eff_price * pred
        profit    = (eff_price - price * 0.70) * pred
        margin    = (eff_price - price * 0.70) / price * 100 if price > 0 else 0

        st.markdown(section_header("PREDICTION RESULTS"), unsafe_allow_html=True)
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            st.markdown(kpi_html("Predicted Units", format_number(pred), color=COLORS["primary"]), unsafe_allow_html=True)
        with rk2:
            st.markdown(kpi_html("Est. Revenue",    f"Rs {revenue:,.0f}",  color=COLORS["teal"]),    unsafe_allow_html=True)
        with rk3:
            st.markdown(kpi_html("Est. Profit",     f"Rs {profit:,.0f}",   color=COLORS["purple"]),  unsafe_allow_html=True)
        with rk4:
            st.markdown(kpi_html("Profit Margin",   f"{margin:.1f}%",      color=COLORS["orange"]),  unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Gauge
        st.markdown(chart_label_html("PREDICTED DEMAND vs AVERAGE DEMAND"), unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            delta={"reference": df["Units Sold"].mean(), "valueformat": ".0f"},
            gauge={
                "axis":  {"range": [0, df["Units Sold"].max()], "tickcolor": COLORS["muted"]},
                "bar":   {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, df["Units Sold"].quantile(0.33)],
                     "color": "rgba(248,113,113,0.18)"},
                    {"range": [df["Units Sold"].quantile(0.33), df["Units Sold"].quantile(0.66)],
                     "color": "rgba(246,173,85,0.18)"},
                    {"range": [df["Units Sold"].quantile(0.66), df["Units Sold"].max()],
                     "color": "rgba(63,185,80,0.18)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["teal"], "width": 3},
                    "thickness": 0.8, "value": df["Units Sold"].mean(),
                },
            },
            title={"text": "", "font": {"color": COLORS["text"]}},
        ))
        apply_dark_chart_style(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Price sensitivity
        st.markdown(chart_label_html("PRICE SENSITIVITY ANALYSIS"), unsafe_allow_html=True)
        prices  = np.linspace(price * 0.7, price * 1.3, 15)
        demands = []
        for p in prices:
            r2_inp = inp.copy()
            r2_inp["Price"] = p
            row2 = pd.DataFrame([r2_inp])[feat_cols]
            if model_choice in SCALED_MODELS:
                d = float(model.predict(scaler.transform(row2))[0])
            else:
                d = float(model.predict(row2)[0])
            demands.append(max(0, d))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=prices, y=demands, mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=7, color=COLORS["primary"], line=dict(color="#fff", width=1)),
            fill="tozeroy", fillcolor="rgba(110,118,229,0.09)",
            name="Predicted Demand",
        ))
        fig2.add_vline(x=price, line_dash="dash", line_color=COLORS["teal"],
                       annotation_text="Current Price",
                       annotation_font_color=COLORS["teal"])
        apply_dark_chart_style(fig2, height=300)
        fig2.update_layout(xaxis_title="Price (Rs)", yaxis_title="Predicted Units Sold")
        st.plotly_chart(fig2, use_container_width=True)