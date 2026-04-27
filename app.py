"""
app.py
------
DemandLens 
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import re
import sys
import importlib.util

ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="DemandLens ",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
CSS_PATH = os.path.join(ROOT, "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, encoding="utf-8") as f:
        raw_css = f.read()
    clean_css = re.sub(r'/\*.*?\*/', '', raw_css, flags=re.DOTALL)
    st.markdown(f"<style>{clean_css}</style>", unsafe_allow_html=True)

# ── Dynamic page loader ───────────────────────────────────────────────────────
def _load_page(module_name: str):
    file_path = os.path.join(ROOT, "views", f"{module_name}.py")
    if not os.path.exists(file_path):
        st.error(f"Page file not found: {file_path}")
        st.stop()
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

dashboard     = _load_page("dashboard")
data_explorer = _load_page("data_explorer")
elasticity    = _load_page("elasticity")
prediction    = _load_page("prediction")
optimizer     = _load_page("optimizer")
performance   = _load_page("performance")

from utils.preprocessing       import load_raw_data, preprocess_data
from utils.feature_engineering import add_business_features
from utils.helper_functions    import render_app_header


@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    raw      = load_raw_data()
    clean    = preprocess_data(raw)
    enriched = add_business_features(clean)
    return enriched

df = load_data()

# ── Top Header ────────────────────────────────────────────────────────────────
st.markdown(render_app_header(), unsafe_allow_html=True)

# ── Top-level Tab Navigation (no emojis) ─────────────────────────────────────
tab_overview, tab_explorer, tab_elasticity, tab_prediction, tab_optimizer, tab_performance = st.tabs([
    "Overview",
    "Explorer",
    "Elasticity",
    "Prediction",
    "Optimizer",
    "Performance",
])

with tab_overview:
    dashboard.show(df)

with tab_explorer:
    data_explorer.show(df)

with tab_elasticity:
    elasticity.show(df)

with tab_prediction:
    prediction.show(df)

with tab_optimizer:
    optimizer.show(df)

with tab_performance:
    performance.show(df)