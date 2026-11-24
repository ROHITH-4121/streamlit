"""
E-Commerce Data Preprocessing & ML Readiness Platform
Main Application Entry Point & Navigation Router
Author: DWDM Capstone Project
"""

import os
# Suppress Joblib physical cores subprocess call on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import streamlit as st
import pandas as pd

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="E-Commerce Preprocessing & ML Readiness Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Custom CSS
def load_css(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
load_css(css_path)

# 3. Session State Initialization
from utils.session_manager import init_session_state, reset_pipeline, get_current_working_df, mark_stage_complete, log_action
from modules.data_collection import load_sample_dataset

init_session_state()

# Auto-load sample dataset on first launch if raw_df is empty so user immediately sees live data
if st.session_state.raw_df is None:
    sample_df, _ = load_sample_dataset("ecommerce_main.csv")
    if sample_df is not None:
        st.session_state.raw_df = sample_df.copy()
        st.session_state.dataset_name = "ecommerce_main.csv"
        mark_stage_complete("Raw Data")
        log_action("Collection", "Preloaded default sample e-commerce transactions.")

# 4. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2 class="sidebar-title">⚡ DATA PREPROCESSING & ML PLATFORM</h2>
        <div class="sidebar-sub">E-Commerce Capstone Framework</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    nav_options = [
        "Dashboard",
        "Module 1 – Cleaning & Integration",
        "Module 2 – Transformation & Reduction",
        "Module 3 – Validation & ML Readiness",
        "Classification Analysis",
        "Customer Segmentation",
        "Reports & Downloads",
        "About Project"
    ]

    # Map current active page or set default
    if st.session_state.active_page not in nav_options:
        st.session_state.active_page = "Dashboard"

    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.active_page),
        label_visibility="collapsed"
    )
    # Sync if user clicked sidebar radio
    st.session_state.active_page = selected_page

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Session Dataset Inspector Card in Sidebar
    working_df = get_current_working_df()
    if working_df is not None:
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px; margin-bottom: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Active Dataset</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; word-break: break-all;">{st.session_state.dataset_name}</div>
            <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">
                <b>{len(working_df):,}</b> rows × <b>{len(working_df.columns)}</b> cols
            </div>
            <div style="font-size: 0.75rem; color: #06b6d4; margin-top: 4px;">
                Stage: <b>{st.session_state.current_stage}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Reset Pipeline Button
    st.markdown("---")
    if st.button("🔄 Reset Preprocessing Pipeline", use_container_width=True, help="Reverts all transformed stages back to initial raw dataset"):
        reset_pipeline()
        st.success("Pipeline reset to raw dataset!")
        st.rerun()

    st.markdown("""
    <div style='font-size: 0.72rem; color: #64748b; text-align: center; margin-top: 20px;'>
        Google Deepmind Capstone Edition<br/>Python 3.13 | Streamlit | Scikit-Learn
    </div>
    """, unsafe_allow_html=True)

# 5. Route to Page Views
from pages import (
    dashboard,
    module1_cleaning,
    module2_transformation,
    module3_validation,
    classification,
    clustering,
    reports,
    about
)

if st.session_state.active_page == "Dashboard":
    dashboard.render()
elif st.session_state.active_page == "Module 1 – Cleaning & Integration":
    module1_cleaning.render()
elif st.session_state.active_page == "Module 2 – Transformation & Reduction":
    module2_transformation.render()
elif st.session_state.active_page == "Module 3 – Validation & ML Readiness":
    module3_validation.render()
elif st.session_state.active_page == "Classification Analysis":
    classification.render()
elif st.session_state.active_page == "Customer Segmentation":
    clustering.render()
elif st.session_state.active_page == "Reports & Downloads":
    reports.render()
elif st.session_state.active_page == "About Project":
    about.render()
