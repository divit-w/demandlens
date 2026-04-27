"""
views/data_explorer.py
-----------------------
DemandLens — Interactive Data Explorer.
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from utils.helper_functions import (
    apply_dark_chart_style,
    section_header,
    chart_label_html,
    metric_card_html,
    COLORS,
    CATEGORY_COLORS,
)


def show(df: pd.DataFrame):

    st.markdown(section_header(
        "DATA EXPLORER",
        "Slice, filter, and explore the retail dataset"
    ), unsafe_allow_html=True)

    # ── Filters row ───────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:{COLORS['card']};border:1px solid {COLORS['border']};
border-radius:10px;padding:20px;margin-bottom:20px;">
<div style="font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.18em;
color:{COLORS['muted']};font-family:JetBrains Mono,monospace;margin-bottom:16px;">
FILTERS</div>
""", unsafe_allow_html=True)

    fl1, fl2, fl3 = st.columns(3)
    with fl1:
        cats    = st.multiselect("Category",    df["Category"].unique(),    default=list(df["Category"].unique()))
        regions = st.multiselect("Region",      df["Region"].unique(),      default=list(df["Region"].unique()))
    with fl2:
        seasons = st.multiselect("Seasonality", df["Seasonality"].unique(), default=list(df["Seasonality"].unique()))
        price_r = st.slider("Price Range (₹)",
                            float(df["Price"].min()), float(df["Price"].max()),
                            (float(df["Price"].min()), float(df["Price"].max())))
    with fl3:
        disc_r = st.slider("Discount (%)",
                           int(df["Discount"].min()), int(df["Discount"].max()),
                           (int(df["Discount"].min()), int(df["Discount"].max())))

    st.markdown("</div>", unsafe_allow_html=True)

    mask = (
        df["Category"].isin(cats) &
        df["Region"].isin(regions) &
        df["Seasonality"].isin(seasons) &
        df["Price"].between(*price_r) &
        df["Discount"].between(*disc_r)
    )
    fd = df[mask].copy()

    st.info(f"**{len(fd):,}** rows matching current filters  ({len(fd)/len(df)*100:.1f}% of dataset)")

    # ── Quick metrics ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card_html("Avg Price",      f"₹{fd['Price'].mean():.2f}",    COLORS["primary"]), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card_html("Avg Units Sold", f"{fd['Units Sold'].mean():.0f}", COLORS["teal"]),   unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card_html("Avg Discount",   f"{fd['Discount'].mean():.1f}%",  COLORS["purple"]), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card_html("Avg Revenue",    f"₹{fd['Revenue'].mean():.0f}",   COLORS["orange"]), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Price vs Demand",
        "🏷️  Discount Effect",
        "🌍  Regional",
        "🗓️  Seasonal",
    ])

    with tab1:
        st.markdown(chart_label_html("PRICE VS UNITS SOLD (SCATTER BY CATEGORY)"), unsafe_allow_html=True)
        samp = fd.sample(min(3000, len(fd)), random_state=1)
        fig  = px.scatter(
            samp, x="Price", y="Units Sold", color="Category",
            size="Revenue", hover_data=["Product ID", "Store ID", "Discount"],
            color_discrete_sequence=CATEGORY_COLORS, opacity=0.7, trendline="ols",
        )
        apply_dark_chart_style(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(chart_label_html("DISCOUNT IMPACT ON DEMAND (AVG UNITS PER DISCOUNT LEVEL)"), unsafe_allow_html=True)
        disc_grp = fd.groupby("Discount").agg(
            Avg_Units=("Units Sold", "mean"),
            Avg_Revenue=("Revenue", "mean"),
        ).reset_index()
        fig = px.bar(disc_grp, x="Discount", y="Avg_Units",
                     color="Avg_Revenue", color_continuous_scale="Blues",
                     labels={"Avg_Units": "Avg Units Sold", "Avg_Revenue": "Avg Revenue (₹)"},
                     text_auto=".0f")
        apply_dark_chart_style(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        r1, r2 = st.columns(2)
        reg_rev = fd.groupby("Region")["Revenue"].sum().reset_index()
        with r1:
            st.markdown(chart_label_html("REVENUE BY REGION"), unsafe_allow_html=True)
            fig1 = px.bar(reg_rev, x="Region", y="Revenue", color="Region",
                          color_discrete_sequence=CATEGORY_COLORS, text_auto=".2s")
            apply_dark_chart_style(fig1, height=320)
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        reg_units = fd.groupby(["Region", "Category"])["Units Sold"].sum().reset_index()
        with r2:
            st.markdown(chart_label_html("UNITS SOLD — REGION × CATEGORY"), unsafe_allow_html=True)
            fig2 = px.sunburst(reg_units, path=["Region", "Category"], values="Units Sold",
                               color_discrete_sequence=CATEGORY_COLORS)
            apply_dark_chart_style(fig2, height=320)
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        s1, s2 = st.columns(2)
        sea = fd.groupby("Seasonality")["Units Sold"].mean().reset_index()
        with s1:
            st.markdown(chart_label_html("DEMAND BY SEASON"), unsafe_allow_html=True)
            fig1 = px.bar(sea, x="Seasonality", y="Units Sold", color="Seasonality",
                          color_discrete_sequence=CATEGORY_COLORS, text_auto=".0f")
            apply_dark_chart_style(fig1, height=300)
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        monthly = fd.groupby(["Month", "Category"])["Units Sold"].mean().reset_index()
        with s2:
            st.markdown(chart_label_html("MONTHLY DEMAND BY CATEGORY"), unsafe_allow_html=True)
            fig2 = px.line(monthly, x="Month", y="Units Sold", color="Category",
                           color_discrete_sequence=CATEGORY_COLORS, markers=True)
            apply_dark_chart_style(fig2, height=300)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Dataset preview ───────────────────────────────────────────────────
    st.markdown(section_header(
        "DATASET PREVIEW",
        f"Showing first 200 of {len(fd):,} filtered rows"
    ), unsafe_allow_html=True)
    display_cols = [
        "Date", "Store ID", "Product ID", "Category", "Region",
        "Price", "Discount", "Units Sold", "Revenue", "Profit",
        "Profit Margin %", "Competitor Pricing", "Competitor Diff",
        "Seasonality", "Weather Condition",
    ]
    st.dataframe(
        fd[[c for c in display_cols if c in fd.columns]].head(200),
        use_container_width=True, height=360,
    )