"""
views/optimizer.py
------------------
DemandLens — Price Optimizer (CO2).
Uses best trained model from set (Linear Regression / Decision Tree / Random Forest).
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
    metric_card_html,
    impact_card_html,
    format_currency,
    format_number,
    COLORS,
    CATEGORY_COLORS,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SCALED_MODELS = {"Linear Regression"}


def _load():
    try:
        model     = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        feat_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        # Identify model name from class
        name = type(model).__name__
        return model, scaler, feat_cols, name
    except Exception:
        return None, None, None, None


def _encode_col(df, col, val):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    try:
        return int(le.transform([val])[0])
    except Exception:
        return 0


def _predict(model, scaler, feat_cols, inp_dict, model_name):
    row = pd.DataFrame([inp_dict])[feat_cols]
    if model_name in SCALED_MODELS:
        return max(0.0, float(model.predict(scaler.transform(row))[0]))
    return max(0.0, float(model.predict(row)[0]))


def _grid_search(model, scaler, feat_cols, base_inp, price_range, discount, model_name, n=100):
    prices  = np.linspace(price_range[0], price_range[1], n)
    results = []
    for p in prices:
        inp = base_inp.copy()
        inp["Price"] = p
        eff    = p * (1 - discount / 100)
        demand = _predict(model, scaler, feat_cols, inp, model_name)
        results.append({
            "Price":   p,
            "Demand":  demand,
            "Revenue": eff * demand,
            "Profit":  (eff - p * 0.70) * demand,
        })
    return pd.DataFrame(results)


def show(df: pd.DataFrame):

    st.markdown(section_header(
        "PRICE OPTIMIZER",
        "Find the optimal price to maximise revenue or profit (CO2)"
    ), unsafe_allow_html=True)

    model, scaler, feat_cols, model_name = _load()
    if model is None:
        st.warning("No trained model found. Go to the Performance tab and train models first.")
        return

    left, right = st.columns([1, 2.6], gap="medium")

    with left:
        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
PRODUCT CONFIGURATION</div>
""", unsafe_allow_html=True)

        category    = st.selectbox("Category",          df["Category"].unique(),          key="opt_cat")
        region      = st.selectbox("Region",            df["Region"].unique(),            key="opt_reg")
        weather     = st.selectbox("Weather Condition", df["Weather Condition"].unique(), key="opt_wea")
        seasonality = st.selectbox("Seasonality",       df["Seasonality"].unique(),       key="opt_sea")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
PRICE CONFIGURATION</div>
""", unsafe_allow_html=True)

        cur_price   = st.slider("Current Price (Rs)",      10.0, 5000.0, 1000.0, step=50.0, key="opt_price")
        discount    = st.slider("Discount (%)",             0,   40,     10,               key="opt_disc")
        comp_price  = st.slider("Competitor Price (Rs)",   10.0, 5000.0, 1000.0, step=50.0, key="opt_comp")
        inventory   = st.slider("Inventory Level",          50,  500,    250,              key="opt_inv")
        demand_fc   = st.slider("Demand Forecast",          0.0, 500.0,  130.0, step=5.0, key="opt_dfc")
        holiday     = st.selectbox("Promotion Active", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No",            key="opt_hol")
        month       = st.slider("Month", 1, 12, 6, key="opt_mon")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
SEARCH RANGE</div>
""", unsafe_allow_html=True)

        price_min = st.slider("Min Price (Rs)", 10.0, cur_price,   max(10.0, cur_price * 0.5))
        price_max = st.slider("Max Price (Rs)", cur_price, 10000.0, min(10000.0, cur_price * 1.5))
        objective = st.radio("Optimise for", ["Revenue", "Profit"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("Find Optimal Price", use_container_width=True)

    with right:
        if not run:
            st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:60px 20px;text-align:center;margin-bottom:14px;">
<div style="color:{COLORS['muted']};font-size:13px;font-family:JetBrains Mono,monospace;">
Configure parameters and click
<span style="color:{COLORS['primary']};font-weight:600;">Find Optimal Price</span>
to run the optimisation.
</div>
</div>
""", unsafe_allow_html=True)
            return

        base_inp = {
            "Price":                 cur_price,
            "Discount_frac":         discount / 100,
            "Competitor Pricing":    comp_price,
            "Inventory Level":       inventory,
            "Demand Forecast":       demand_fc,
            "Holiday/Promotion":     holiday,
            "Month":                 month,
            "Week":                  26,
            "Quarter":               (month - 1) // 3 + 1,
            "DayOfWeek":             2,
            "Category_enc":          _encode_col(df, "Category",          category),
            "Region_enc":            _encode_col(df, "Region",            region),
            "Weather Condition_enc": _encode_col(df, "Weather Condition", weather),
            "Seasonality_enc":       _encode_col(df, "Seasonality",       seasonality),
        }

        with st.spinner("Running optimisation ..."):
            grid = _grid_search(model, scaler, feat_cols, base_inp,
                                (price_min, price_max), discount, model_name)

        target_col = "Revenue" if objective == "Revenue" else "Profit"
        best_row   = grid.loc[grid[target_col].idxmax()]
        opt_price  = best_row["Price"]
        opt_demand = best_row["Demand"]
        opt_rev    = best_row["Revenue"]
        opt_profit = best_row["Profit"]

        cur_demand = _predict(model, scaler, feat_cols, base_inp, model_name)
        cur_eff    = cur_price * (1 - discount / 100)
        cur_rev    = cur_eff * cur_demand
        cur_profit = (cur_eff - cur_price * 0.70) * cur_demand
        rev_uplift = (opt_rev - cur_rev) / (cur_rev + 1e-9) * 100

        st.markdown(section_header("OPTIMAL PRICE FOUND"), unsafe_allow_html=True)
        rk1, rk2, rk3, rk4, rk5 = st.columns(5)
        with rk1:
            st.markdown(kpi_html("Optimal Price", f"Rs {opt_price:.0f}",    color=COLORS["primary"]), unsafe_allow_html=True)
        with rk2:
            st.markdown(kpi_html("Current Price", f"Rs {cur_price:.0f}",    color=COLORS["muted"]),   unsafe_allow_html=True)
        with rk3:
            st.markdown(kpi_html("Pred. Demand",  format_number(opt_demand), color=COLORS["teal"]),   unsafe_allow_html=True)
        with rk4:
            st.markdown(kpi_html("Est. Revenue",  format_currency(opt_rev),  color=COLORS["purple"]), unsafe_allow_html=True)
        with rk5:
            col = COLORS["accent"] if rev_uplift >= 0 else "#f87171"
            st.markdown(kpi_html(f"{objective} Uplift", f"{rev_uplift:+.1f}%", color=col), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(chart_label_html(f"PRICE vs {objective.upper()} OPTIMISATION CURVE"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grid["Price"], y=grid[target_col],
            fill="tozeroy", mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            fillcolor="rgba(110,118,229,0.09)",
        ))
        fig.add_vline(x=opt_price, line_dash="dash", line_color=COLORS["teal"],
                      annotation_text=f"Optimal Rs {opt_price:.0f}",
                      annotation_font_color=COLORS["teal"])
        fig.add_vline(x=cur_price, line_dash="dot", line_color=COLORS["secondary"],
                      annotation_text=f"Current Rs {cur_price:.0f}",
                      annotation_font_color=COLORS["secondary"])
        apply_dark_chart_style(fig, height=320)
        fig.update_layout(xaxis_title="Price (Rs)", yaxis_title=f"Predicted {objective} (Rs)")
        st.plotly_chart(fig, use_container_width=True)

        cr1, cr2 = st.columns(2)
        with cr1:
            st.markdown(chart_label_html("DEMAND CURVE"), unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=grid["Price"], y=grid["Demand"], mode="lines",
                                      line=dict(color=COLORS["purple"], width=2.5),
                                      fill="tozeroy", fillcolor="rgba(183,148,244,0.08)"))
            fig2.add_vline(x=opt_price, line_dash="dash", line_color=COLORS["teal"])
            apply_dark_chart_style(fig2, height=280)
            fig2.update_layout(xaxis_title="Price (Rs)", yaxis_title="Predicted Units")
            st.plotly_chart(fig2, use_container_width=True)

        with cr2:
            st.markdown(chart_label_html("REVENUE vs PROFIT CURVE"), unsafe_allow_html=True)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=grid["Price"], y=grid["Revenue"], mode="lines",
                                      line=dict(color=COLORS["teal"], width=2.5), name="Revenue"))
            fig3.add_trace(go.Scatter(x=grid["Price"], y=grid["Profit"], mode="lines",
                                      line=dict(color=COLORS["orange"], width=2, dash="dot"), name="Profit"))
            fig3.add_vline(x=opt_price, line_dash="dash", line_color=COLORS["secondary"])
            apply_dark_chart_style(fig3, height=280)
            fig3.update_layout(xaxis_title="Price (Rs)", yaxis_title="Rs Value")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(section_header("BEFORE vs AFTER COMPARISON"), unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Scenario":       ["Current Price",            "Optimal Price"],
            "Price (Rs)":     [f"Rs {cur_price:.0f}",      f"Rs {opt_price:.0f}"],
            "Demand (units)": [f"{cur_demand:.0f}",         f"{opt_demand:.0f}"],
            "Revenue (Rs)":   [f"Rs {cur_rev:,.0f}",        f"Rs {opt_rev:,.0f}"],
            "Profit (Rs)":    [f"Rs {cur_profit:,.0f}",     f"Rs {opt_profit:,.0f}"],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)