"""
views/dashboard.py
-------------------
DemandLens — Overview Dashboard.
Layout: KPI cards row → Revenue trend + mix → Demand + Price → Promo + Weather → Heatmap
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.helper_functions import (
    apply_dark_chart_style,
    kpi_html,
    chart_label_html,
    section_header,
    format_currency,
    format_number,
    COLORS,
    CATEGORY_COLORS,
)


def show(df: pd.DataFrame):

    # ── Ensure Date is datetime ───────────────────────────────────────────
    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ── Page heading ──────────────────────────────────────────────────────
    st.markdown(section_header(
        "RETAIL INTELLIGENCE OVERVIEW",
        "Demand analytics, pricing performance, and elasticity insights"
    ), unsafe_allow_html=True)

    # ── KPI Calculations ──────────────────────────────────────────────────
    total_revenue  = df["Revenue"].sum()    if "Revenue"    in df.columns else 0
    total_units    = df["Units Sold"].sum() if "Units Sold" in df.columns else 0
    avg_price      = df["Price"].mean()     if "Price"      in df.columns else 0
    total_profit   = df["Profit"].sum()     if "Profit"     in df.columns else 0
    avg_elasticity = -0.72

    # ── KPI Cards ─────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Revenue",    format_currency(total_revenue), "+12.4% this month", COLORS["primary"]),
        ("Units Sold", format_number(total_units),     "+5.1% demand growth", COLORS["teal"]),
        ("Avg Price",  f"₹{avg_price:.2f}",            "Market stable",       COLORS["purple"]),
        ("Profit",     format_currency(total_profit),  "+8.9% margin",        COLORS["orange"]),
        ("Elasticity", f"{avg_elasticity:.2f}",        "Moderately Elastic",  COLORS["secondary"]),
    ]
    for col, (label, value, delta, color) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(kpi_html(label, value, delta, color=color), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Revenue Trend + Category Mix ──────────────────────────────────────
    c1, c2 = st.columns([2.2, 1])

    with c1:
        st.markdown(chart_label_html("REVENUE TREND (MONTHLY PERFORMANCE)"), unsafe_allow_html=True)
        if "Date" in df.columns and "Revenue" in df.columns:
            monthly = (
                df.dropna(subset=["Date"])
                .groupby(df["Date"].dt.to_period("M").astype(str))["Revenue"]
                .sum().reset_index()
            )
            monthly.columns = ["Month", "Revenue"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["Month"], y=monthly["Revenue"],
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["primary"], width=2.5),
                fillcolor="rgba(110,118,229,0.10)",
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            ))
            apply_dark_chart_style(fig, height=340)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date or Revenue column not found.")

    with c2:
        st.markdown(chart_label_html("REVENUE MIX (CATEGORY %)"), unsafe_allow_html=True)
        if "Category" in df.columns and "Revenue" in df.columns:
            cat_rev = df.groupby("Category")["Revenue"].sum().reset_index()
            fig = px.pie(
                cat_rev, values="Revenue", names="Category",
                hole=0.70, color_discrete_sequence=CATEGORY_COLORS,
            )
            apply_dark_chart_style(fig, height=340)
            fig.update_traces(textinfo="percent", textfont_size=11)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or Revenue column not found.")

    # ── Demand by Category + Price Distribution ───────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(chart_label_html("DEMAND BY CATEGORY (UNITS SOLD)"), unsafe_allow_html=True)
        if all(c in df.columns for c in ["Category", "Seasonality", "Units Sold"]):
            cs = df.groupby(["Category", "Seasonality"])["Units Sold"].sum().reset_index()
            fig = px.bar(
                cs, x="Category", y="Units Sold",
                color="Seasonality", barmode="group",
                color_discrete_sequence=CATEGORY_COLORS,
            )
            apply_dark_chart_style(fig, height=320)
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Category, Seasonality, or Units Sold not found.")

    with c4:
        st.markdown(chart_label_html("PRICE DISTRIBUTION (BY CATEGORY)"), unsafe_allow_html=True)
        if all(c in df.columns for c in ["Category", "Price"]):
            fig = px.box(
                df, x="Category", y="Price",
                color="Category", color_discrete_sequence=CATEGORY_COLORS,
            )
            apply_dark_chart_style(fig, height=320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or Price column not found.")

    # ── Promotion Impact + Weather Impact ─────────────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        st.markdown(chart_label_html("PROMOTION IMPACT (AVG UNITS SOLD)"), unsafe_allow_html=True)
        if all(c in df.columns for c in ["Category", "Holiday/Promotion", "Units Sold"]):
            promo = df.groupby(["Category", "Holiday/Promotion"])["Units Sold"].mean().reset_index()
            promo["Promotion"] = promo["Holiday/Promotion"].map({0: "No Promo", 1: "Promotion"})
            fig = px.bar(
                promo, x="Category", y="Units Sold",
                color="Promotion", barmode="group",
                color_discrete_map={
                    "No Promo":  "rgba(255,255,255,0.12)",
                    "Promotion": COLORS["teal"],
                },
            )
            apply_dark_chart_style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Category, Holiday/Promotion, or Units Sold not found.")

    with c6:
        st.markdown(chart_label_html("WEATHER IMPACT (AVG UNITS BY CONDITION)"), unsafe_allow_html=True)
        if all(c in df.columns for c in ["Weather Condition", "Units Sold"]):
            weather = df.groupby("Weather Condition")["Units Sold"].mean().reset_index()
            fig = px.bar(
                weather, x="Weather Condition", y="Units Sold",
                color="Weather Condition", color_discrete_sequence=CATEGORY_COLORS,
            )
            apply_dark_chart_style(fig, height=300)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Weather Condition or Units Sold column not found.")

    # ── Correlation Heatmap ───────────────────────────────────────────────
    st.markdown(chart_label_html("CORRELATION MATRIX (RETAIL VARIABLES)"), unsafe_allow_html=True)
    numeric_cols = [
        "Price", "Discount", "Competitor Pricing", "Inventory Level",
        "Units Sold", "Demand Forecast", "Revenue", "Profit",
        "Profit Margin %", "Stock Turnover",
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]
    if len(available_cols) >= 2:
        corr = df[available_cols].corr()
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        apply_dark_chart_style(fig, height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to render the correlation matrix.")