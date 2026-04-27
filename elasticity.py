"""
views/elasticity.py
--------------------
DemandLens — Price Elasticity Analysis.
Layout mirrors the screenshot: left control panel + right chart area.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.feature_engineering import compute_price_elasticity, compute_product_elasticity
from utils.helper_functions import (
    apply_dark_chart_style,
    section_header,
    chart_label_html,
    kpi_html,
    metric_card_html,
    impact_card_html,
    COLORS,
    CATEGORY_COLORS,
)

# PED values per product type (matching screenshot dropdown)
PED_PRESETS = {
    "Electronics (PED -1.3)":    -1.30,
    "Clothing (PED -0.8)":        -0.80,
    "Groceries (PED -0.4)":       -0.40,
    "Furniture (PED -1.1)":       -1.10,
    "Sports (PED -0.9)":          -0.90,
}


def _demand(base_price: float, base_qty: float, ped: float, new_price: float) -> float:
    """Simple PED demand formula: Q2 = Q1 * (P2/P1)^PED"""
    if base_price <= 0:
        return base_qty
    return base_qty * ((new_price / base_price) ** ped)


def show(df: pd.DataFrame):

    st.markdown(section_header(
        "PRICE ELASTICITY ANALYSIS",
        "How sensitive is demand to price changes?"
    ), unsafe_allow_html=True)

    # ── Main layout: left panel + right charts ────────────────────────────
    left, right = st.columns([1, 2.6], gap="medium")

    # ── LEFT: Control Panel ───────────────────────────────────────────────
    with left:
        # Product Intelligence panel
        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
PRODUCT INTELLIGENCE</div>
""", unsafe_allow_html=True)

        selected_product = st.selectbox(
            "Select Product Category",
            list(PED_PRESETS.keys()),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        ped_value = PED_PRESETS[selected_product]

        # Price Configuration panel
        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
PRICE CONFIGURATION</div>
""", unsafe_allow_html=True)

        base_price = st.slider(
            "Base Price (₹)   MARKET BASE",
            min_value=100, max_value=10000, value=1000, step=50,
            label_visibility="visible",
        )
        base_qty = st.slider(
            "Base Quantity (units)",
            min_value=100, max_value=5000, value=500, step=50,
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Strategy Shift panel
        st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:14px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:14px;">
STRATEGY SHIFT</div>
""", unsafe_allow_html=True)

        price_change_pct = st.slider(
            "Price Change %   PRICE CHANGE",
            min_value=-50, max_value=50, value=0, step=1,
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Compute derived values
        new_price   = base_price * (1 + price_change_pct / 100)
        new_demand  = _demand(base_price, base_qty, ped_value, new_price)
        demand_chg  = ((new_demand - base_qty) / base_qty * 100) if base_qty > 0 else 0
        base_rev    = base_price * base_qty
        new_rev     = new_price  * new_demand
        rev_impact  = new_rev - base_rev

        # PED INDEX + DEMAND Δ mini cards
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(metric_card_html(
                "PED INDEX", f"{ped_value:.2f}",
                value_color="#f87171" if abs(ped_value) > 1 else COLORS["teal"],
            ), unsafe_allow_html=True)
        with mc2:
            sign = "+" if demand_chg >= 0 else ""
            st.markdown(metric_card_html(
                "DEMAND Δ", f"{sign}{demand_chg:.1f}%",
                value_color=COLORS["accent"] if demand_chg >= 0 else "#f87171",
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Net Revenue Impact
        sign_r = "+" if rev_impact >= 0 else ""
        col = COLORS["accent"] if rev_impact >= 0 else "#f87171"
        if price_change_pct == 0:
            impact_val, impact_col = "—", COLORS["muted"]
        else:
            impact_val  = f"{sign_r}₹{abs(rev_impact):,.0f}"
            impact_col  = col
        st.markdown(impact_card_html(impact_val, impact_col), unsafe_allow_html=True)

    # ── RIGHT: Charts ─────────────────────────────────────────────────────
    with right:
        # Dynamic Demand Curve
        st.markdown(chart_label_html("DYNAMIC DEMAND CURVE (PRICE VS UNITS)"), unsafe_allow_html=True)
        prices = np.linspace(base_price * 0.1, base_price * 3, 300)
        units  = [_demand(base_price, base_qty, ped_value, p) for p in prices]

        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(
            x=prices, y=units,
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(110,118,229,0.07)",
            name="Demand",
            hovertemplate="Price: ₹%{x:,.0f}<br>Units: %{y:,.0f}<extra></extra>",
        ))
        # Mark current operating point
        fig_demand.add_trace(go.Scatter(
            x=[new_price], y=[new_demand],
            mode="markers",
            marker=dict(color=COLORS["teal"], size=9, symbol="circle",
                        line=dict(color="#fff", width=1.5)),
            name="Current",
            hovertemplate="New Price: ₹%{x:,.0f}<br>New Units: %{y:,.0f}<extra></extra>",
        ))
        apply_dark_chart_style(fig_demand, height=310)
        fig_demand.update_layout(showlegend=False)
        fig_demand.update_xaxes(tickprefix="₹", tickformat=",")
        fig_demand.update_yaxes(tickformat=",")
        st.plotly_chart(fig_demand, use_container_width=True)

        # Revenue Frontier
        st.markdown(chart_label_html("REVENUE FRONTIER (TOTAL EARNINGS)"), unsafe_allow_html=True)
        revenues = [p * _demand(base_price, base_qty, ped_value, p) for p in prices]

        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode="lines",
            line=dict(color=COLORS["teal"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(79,209,197,0.07)",
            name="Revenue",
            hovertemplate="Price: ₹%{x:,.0f}<br>Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_rev.add_trace(go.Scatter(
            x=[new_price], y=[new_price * new_demand],
            mode="markers",
            marker=dict(color=COLORS["accent"], size=9,
                        line=dict(color="#fff", width=1.5)),
            name="Current",
        ))
        apply_dark_chart_style(fig_rev, height=270)
        fig_rev.update_layout(showlegend=False)
        fig_rev.update_xaxes(tickprefix="₹", tickformat=",")
        fig_rev.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Category Elasticity section ───────────────────────────────────────
    st.markdown(section_header(
        "CATEGORY-LEVEL ELASTICITY",
        "Log-log regression slope per category"
    ), unsafe_allow_html=True)

    try:
        cat_elast = compute_price_elasticity(df)

        cols = st.columns(len(cat_elast))
        for col, (_, row) in zip(cols, cat_elast.iterrows()):
            color = "#f87171" if row["Sensitivity"] == "Elastic" else COLORS["teal"]
            col.markdown(kpi_html(
                row["Category"], f"{row['Elasticity']:.3f}",
                delta=f"{'🔴 Elastic' if row['Sensitivity'] == 'Elastic' else '🟢 Inelastic'}",
                icon="⚡", color=color,
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown(chart_label_html("ELASTICITY BY CATEGORY"), unsafe_allow_html=True)
            fig = px.bar(cat_elast, x="Category", y="Elasticity",
                         color="Sensitivity",
                         color_discrete_map={
                             "Elastic":   "#f87171",
                             "Inelastic": COLORS["teal"],
                         },
                         text="Elasticity")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.add_hline(y=-1, line_dash="dash", line_color=COLORS["orange"],
                          annotation_text="Elastic boundary (E=-1)")
            apply_dark_chart_style(fig, height=340)
            st.plotly_chart(fig, use_container_width=True)

        with cc2:
            st.markdown(chart_label_html("ELASTICITY RADAR"), unsafe_allow_html=True)
            cats        = cat_elast["Category"].tolist()
            vals        = np.abs(cat_elast["Elasticity"]).tolist()
            vals_closed = vals + vals[:1]
            cats_closed = cats + cats[:1]
            fig = go.Figure(go.Scatterpolar(
                r=vals_closed, theta=cats_closed, fill="toself",
                line=dict(color=COLORS["primary"], width=2),
                fillcolor="rgba(110,118,229,0.14)",
            ))
            fig.update_layout(polar=dict(
                radialaxis=dict(visible=True, color=COLORS["muted"]),
                bgcolor="rgba(19,25,39,0.8)",
            ))
            apply_dark_chart_style(fig, height=340)
            st.plotly_chart(fig, use_container_width=True)

        # Product level
        st.markdown(section_header(
            "PRODUCT-LEVEL ELASTICITY", "Top 20 products by sales volume"
        ), unsafe_allow_html=True)
        prod_elast = compute_product_elasticity(df)
        st.markdown(chart_label_html("PRODUCT ELASTICITY BUBBLE CHART"), unsafe_allow_html=True)
        fig = px.scatter(prod_elast, x="Avg Price", y="Elasticity",
                         color="Category", size="Avg Units Sold",
                         hover_data=["Product ID", "Sensitivity"],
                         color_discrete_sequence=CATEGORY_COLORS, text="Product ID")
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.add_hline(y=-1, line_dash="dash", line_color=COLORS["orange"],
                      annotation_text="Elastic boundary")
        apply_dark_chart_style(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            prod_elast.style.background_gradient(subset=["Elasticity"], cmap="RdYlGn_r"),
            use_container_width=True, height=320,
        )

        # Price-demand curves
        st.markdown(section_header(
            "PRICE–DEMAND CURVES", "Average units sold across price bins"
        ), unsafe_allow_html=True)
        df2 = df.copy()
        df2["Price Bin"] = pd.cut(df2["Price"], bins=10)
        curve = df2.groupby(["Category", "Price Bin"], observed=True)["Units Sold"].mean().reset_index()
        curve["Price Mid"] = curve["Price Bin"].apply(lambda x: x.mid).astype(float)
        st.markdown(chart_label_html("DEMAND CURVE BY CATEGORY"), unsafe_allow_html=True)
        fig = px.line(curve, x="Price Mid", y="Units Sold", color="Category",
                      color_discrete_sequence=CATEGORY_COLORS, markers=True)
        apply_dark_chart_style(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute category elasticity: {e}")