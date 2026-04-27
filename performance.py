"""
views/performance.py
---------------------
DemandLens  — Model Performance & Comparison (CO4).
Models: Linear Regression, Decision Tree, Random Forest
Libraries: scikit-learn, NumPy, Pandas, Plotly
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

from utils.preprocessing  import get_feature_matrix
from utils.model_training import train_all_models
from utils.helper_functions import (
    apply_dark_chart_style,
    section_header,
    chart_label_html,
    kpi_html,
    COLORS,
    CATEGORY_COLORS,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SYLLABUS_MODELS = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
]


def show(df: pd.DataFrame):

    st.markdown(section_header(
        "MODEL PERFORMANCE & COMPARISON",
        "Train, evaluate, and compare syllabus ML models (CO3, CO4)"
    ), unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        train_clicked = st.button("Train All Models", use_container_width=True)
    with col_info:
        st.info(
            "Click **Train All Models** to run **Linear Regression, Decision Tree, "
            "Random Forest** on the full dataset."
        )

    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None
        st.session_state["best_name"]     = None

    if train_clicked:
        with st.spinner("Training models — this may take 30–90 seconds ..."):
            X, y, feat_cols = get_feature_matrix(df)
            results, best_name = train_all_models(X, y)
            st.session_state["model_results"] = results
            st.session_state["best_name"]     = best_name
        st.success(f"Training complete. Best model: **{best_name}**")

    results   = st.session_state["model_results"]
    best_name = st.session_state["best_name"]

    if results is None:
        st.markdown(f"""
<div style="text-align:center;padding:80px 20px;background:{COLORS['card']};
border:1px solid {COLORS['border']};border-radius:10px;">
<div style="color:{COLORS['muted']};font-size:13px;font-family:JetBrains Mono,monospace;margin-top:8px;">
Click <span style="color:{COLORS['primary']};font-weight:600;">Train All Models</span> above to begin.
</div>
</div>
""", unsafe_allow_html=True)
        return

    # ── Metrics ───────────────────────────────────────────────────────────
    st.markdown(section_header(
        "EVALUATION METRICS",
        "R², MAE, MSE, RMSE on 20% held-out test set"
    ), unsafe_allow_html=True)

    rows = []
    for name in SYLLABUS_MODELS:
        if name not in results:
            continue
        m = results[name]
        rows.append({
            "Model": name,
            "R²":    round(m["R²"],   4),
            "MAE":   round(m["MAE"],  2),
            "MSE":   round(m["MSE"],  2),
            "RMSE":  round(m["RMSE"], 2),
            "Best":  "Best" if name == best_name else "",
        })
    metrics_df = pd.DataFrame(rows)

    best = results[best_name]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_html("Best R²",   f"{best['R²']:.4f}",  color=COLORS["primary"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html("Best MAE",  f"{best['MAE']:.2f}", color=COLORS["teal"]),    unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_html("Best RMSE", f"{best['RMSE']:.2f}",color=COLORS["purple"]),  unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_html("Best MSE",  f"{best['MSE']:.2f}", color=COLORS["orange"]),  unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=["R²"],                  color="#163b20")
            .highlight_min(subset=["MAE", "RMSE", "MSE"],  color="#163b20"),
        use_container_width=True, hide_index=True,
    )

    # ── Visual Comparison ─────────────────────────────────────────────────
    st.markdown(section_header("VISUAL MODEL COMPARISON"), unsafe_allow_html=True)
    t1, t2 = st.tabs(["Bar Charts", "Actual vs Predicted"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(chart_label_html("R² SCORE (HIGHER IS BETTER)"), unsafe_allow_html=True)
            fig1 = px.bar(metrics_df, x="Model", y="R²",
                          color="Model", color_discrete_sequence=CATEGORY_COLORS, text="R²")
            fig1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            apply_dark_chart_style(fig1, height=320)
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown(chart_label_html("RMSE (LOWER IS BETTER)"), unsafe_allow_html=True)
            fig2 = px.bar(metrics_df, x="Model", y="RMSE",
                          color="Model", color_discrete_sequence=CATEGORY_COLORS, text="RMSE")
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            apply_dark_chart_style(fig2, height=320)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(chart_label_html("MAE (LOWER IS BETTER)"), unsafe_allow_html=True)
            fig3 = px.bar(metrics_df, x="Model", y="MAE",
                          color="Model", color_discrete_sequence=CATEGORY_COLORS, text="MAE")
            fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            apply_dark_chart_style(fig3, height=280)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            st.markdown(chart_label_html("NORMALISED RADAR (ALL METRICS)"), unsafe_allow_html=True)
            from sklearn.preprocessing import MinMaxScaler
            mc    = metrics_df[["R²", "MAE", "RMSE", "MSE"]].copy()
            mc_sc = MinMaxScaler().fit_transform(mc)
            mc_sc[:, 1] = 1 - mc_sc[:, 1]
            mc_sc[:, 2] = 1 - mc_sc[:, 2]
            mc_sc[:, 3] = 1 - mc_sc[:, 3]
            cats  = ["R²", "1-MAE", "1-RMSE", "1-MSE"]
            fig4  = go.Figure()
            for i, name in enumerate(metrics_df["Model"]):
                vals = mc_sc[i].tolist() + [mc_sc[i][0]]
                fig4.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]], fill="toself", name=name,
                    line=dict(color=CATEGORY_COLORS[i % len(CATEGORY_COLORS)], width=2),
                ))
            fig4.update_layout(polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color=COLORS["muted"]),
                bgcolor="rgba(19,25,39,0.8)",
            ))
            apply_dark_chart_style(fig4, height=280)
            st.plotly_chart(fig4, use_container_width=True)

    with t2:
        available = [n for n in SYLLABUS_MODELS if n in results]
        model_sel = st.selectbox("Select Model", available, key="avp_sel")
        sel    = results[model_sel]
        y_test = sel["y_test"]
        y_pred = sel["y_pred"]
        idx    = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)

        st.markdown(chart_label_html(f"ACTUAL vs PREDICTED — {model_sel.upper()}"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test[idx], y=y_pred[idx], mode="markers",
            marker=dict(color=COLORS["primary"], opacity=0.45, size=5),
            name="Predictions",
        ))
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=COLORS["secondary"], dash="dash", width=2),
            name="Perfect Fit",
        ))
        apply_dark_chart_style(fig, height=400)
        fig.update_layout(xaxis_title="Actual Units Sold", yaxis_title="Predicted Units Sold")
        st.plotly_chart(fig, use_container_width=True)

        residuals = y_test[idx] - y_pred[idx]
        st.markdown(chart_label_html("RESIDUALS DISTRIBUTION"), unsafe_allow_html=True)
        fig_r = px.histogram(x=residuals, nbins=40, color_discrete_sequence=[COLORS["purple"]])
        apply_dark_chart_style(fig_r, height=260)
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────
    tree_models = {n: results[n] for n in ["Random Forest", "Decision Tree"] if n in results}
    if tree_models:
        st.markdown(section_header(
            "FEATURE IMPORTANCE",
            "Which features drive demand predictions? (tree-based models)"
        ), unsafe_allow_html=True)
        best_tree = max(tree_models, key=lambda k: tree_models[k]["R²"])
        model_obj = tree_models[best_tree]["model"]
        feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
        if os.path.exists(feat_path):
            import joblib
            feat_cols  = joblib.load(feat_path)
            importance = pd.DataFrame({
                "Feature":    feat_cols,
                "Importance": model_obj.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(14)
            st.markdown(chart_label_html(f"FEATURE IMPORTANCE — {best_tree.upper()}"), unsafe_allow_html=True)
            fig_fi = px.bar(importance, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Blues")
            apply_dark_chart_style(fig_fi, height=360)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("feature_cols.pkl not found. Train models first.")