"""
Page: Dashboard
Main executive view featuring high-level dataset metrics, workflow progress tracker,
data quality indicator, recent pipeline activities, and quick-action navigation.
"""

import streamlit as st
import pandas as pd
from utils.helpers import render_page_header, render_metric_card, render_workflow_progress
from utils.session_manager import get_current_working_df
from modules.data_collection import get_dataset_summary
from modules.data_validation import validate_dataset


def render():
    render_page_header(
        title="E-Commerce Preprocessing & ML Readiness Dashboard",
        subtitle="Real-time data quality monitoring, pipeline progress tracking, and statistical overview.",
        icon="🚀",
        stage_badge=st.session_state.current_stage
    )

    # Workflow Breadcrumbs
    render_workflow_progress(
        current_stage=st.session_state.current_stage,
        completed_dict=st.session_state.pipeline_stages_completed
    )

    # Current working dataset
    current_df = get_current_working_df()

    if current_df is None:
        st.info("👋 Welcome! No dataset is loaded yet. Head over to **Module 1 (Data Collection & Cleaning)** or load our realistic sample e-commerce dataset below to get started.")
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if st.button("📥 Load Built-in Sample Dataset", use_container_width=True):
                from modules.data_collection import load_sample_dataset
                from utils.session_manager import mark_stage_complete, log_action
                sample_df, err = load_sample_dataset("ecommerce_main.csv")
                if sample_df is not None:
                    st.session_state.raw_df = sample_df.copy()
                    st.session_state.dataset_name = "ecommerce_main.csv"
                    mark_stage_complete("Raw Data")
                    log_action("Collection", "Loaded built-in sample e-commerce dataset (1,235 records).")
                    st.success("Sample dataset successfully loaded!")
                    st.rerun()
        return

    # Dataset Summary Statistics
    summary = get_dataset_summary(current_df)
    
    # Calculate Data Quality Score
    val_report = validate_dataset(current_df)
    st.session_state.validation_report = val_report

    # Top Metric Cards (6 cards in 3 columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            title="Total Records",
            value=f"{summary['total_rows']:,}",
            subtitle=f"{summary['memory_usage_mb']} MB in memory",
            icon="📄",
            badge="Rows",
            badge_color="#38bdf8"
        )
    with col2:
        render_metric_card(
            title="Total Features",
            value=f"{summary['total_cols']}",
            subtitle=f"Source: {st.session_state.dataset_name}",
            icon="🏷️",
            badge="Columns",
            badge_color="#6366f1"
        )
    with col3:
        qual_score = val_report["score"]
        badge_col = val_report["badge_color"]
        render_metric_card(
            title="Data Quality Score",
            value=f"{qual_score}%",
            subtitle=val_report["grade"],
            icon="🛡️",
            badge=val_report["grade"].split()[0],
            badge_color=badge_col
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        miss_cells = summary["missing_cells"]
        miss_pct = summary["missing_percentage"]
        miss_color = "#10b981" if miss_cells == 0 else "#f59e0b"
        render_metric_card(
            title="Missing Values",
            value=f"{miss_cells:,}",
            subtitle=f"{miss_pct}% of total data matrix",
            icon="🔍",
            badge="Cells" if miss_cells > 0 else "Clean",
            badge_color=miss_color
        )
    with col5:
        dup_rows = summary["duplicate_rows"]
        dup_pct = summary["duplicate_percentage"]
        dup_color = "#10b981" if dup_rows == 0 else "#ef4444"
        render_metric_card(
            title="Duplicate Records",
            value=f"{dup_rows:,}",
            subtitle=f"{dup_pct}% redundancy rate",
            icon="📑",
            badge="Duplicates" if dup_rows > 0 else "Unique",
            badge_color=dup_color
        )
    with col6:
        stages_done = sum(1 for v in st.session_state.pipeline_stages_completed.values() if v)
        render_metric_card(
            title="Pipeline Status",
            value=f"{stages_done} / 7",
            subtitle=f"Active Stage: {st.session_state.current_stage}",
            icon="⚙️",
            badge="Progress",
            badge_color="#06b6d4"
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Interactive Dataset Preview & Column Explorer
    st.subheader("📋 Dataset Preview & Column Inventory")
    tab_preview, tab_cols, tab_history = st.tabs(["👁️ Data Table", "📊 Column Diagnostics", "📜 Processing History"])

    with tab_preview:
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            preview_rows = st.selectbox("Rows to preview:", [5, 10, 25, 50], index=1)
        with col_ctrl2:
            cols_to_show = st.multiselect("Filter columns (optional):", options=list(current_df.columns), default=list(current_df.columns)[:8])

        view_cols = cols_to_show if cols_to_show else list(current_df.columns)
        st.dataframe(current_df[view_cols].head(preview_rows), use_container_width=True)

    with tab_cols:
        st.dataframe(summary["column_details"], use_container_width=True)

    with tab_history:
        history = st.session_state.get("processing_history", [])
        if history:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.write("No preprocessing transformations executed yet in this session.")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Quick Navigation Cards
    st.subheader("⚡ Quick Navigation to Modules")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧹 Module 1</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Ingest data, purge duplicates, impute missing values, cap outliers, and merge relational tables.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Module 1 →", key="btn_open_m1", use_container_width=True):
            st.session_state.active_page = "Module 1 – Cleaning & Integration"
            st.rerun()

    with nav_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔄 Module 2</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Min-Max / Z-Score scaling, Categorical One-Hot/Label Encoding, PCA, Discretization & Sampling.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Module 2 →", key="btn_open_m2", use_container_width=True):
            st.session_state.active_page = "Module 2 – Transformation & Reduction"
            st.rerun()

    with nav_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Module 3</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>8-point validation check, Data Quality certification, interactive EDA, and train/test ML preparation.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Module 3 →", key="btn_open_m3", use_container_width=True):
            st.session_state.active_page = "Module 3 – Validation & ML Readiness"
            st.rerun()
