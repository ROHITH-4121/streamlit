"""
Page: Module 3 – Data Validation, Visualization & ML Readiness
Executes 8-point data validation, issues Data Quality Score certification,
provides interactive domain & custom EDA, and prepares leak-free train/test sets for ML.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import render_page_header, render_metric_card, render_info_box, render_workflow_progress
from utils.session_manager import get_current_working_df, mark_stage_complete, log_action
from utils.file_handler import dataframe_to_csv_bytes, dataframe_to_excel_bytes

from modules.data_validation import validate_dataset
from modules.visualization import (
    plot_sales_trends, plot_revenue_by_category, plot_revenue_by_payment_method,
    plot_customer_age_distribution, plot_gender_distribution, plot_top_customers,
    plot_delivery_status, plot_best_selling_products, build_custom_chart
)
from modules.classification_models import prepare_ml_dataset


def render():
    render_page_header(
        title="Module 3: Data Validation, Visualization & ML Readiness",
        subtitle="Validate data integrity across 8 quality dimensions, explore interactive visual insights, and package leak-free ML train/test sets.",
        icon="🛡️",
        stage_badge=st.session_state.current_stage
    )

    render_workflow_progress(st.session_state.current_stage, st.session_state.pipeline_stages_completed)

    working_df = get_current_working_df()
    if working_df is None:
        st.warning("⚠️ No dataset loaded. Please ingest or prepare a dataset first.")
        return

    # 3 Main Tabs: Validation, EDA & Visuals, ML Readiness
    tab_val, tab_eda, tab_ml = st.tabs([
        "🛡️ 1. Data Validation & Quality Score",
        "📊 2. Exploratory Data Analytics (EDA)",
        "🤖 3. ML Readiness & Partitioning"
    ])

    # =========================================================================
    # TAB 1: DATA VALIDATION & QUALITY SCORING
    # =========================================================================
    with tab_val:
        st.subheader("1.1 Automated 8-Point Data Quality Audit")
        
        val_report = validate_dataset(working_df)
        st.session_state.validation_report = val_report
        st.session_state.validated_df = working_df

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            render_metric_card("Quality Score", f"{val_report['score']}%", val_report["grade"], "🛡️", val_report["grade"].split()[0], val_report["badge_color"])
        with col_q2:
            render_metric_card("Passed Checks", f"{val_report['passed_count']} / 8", "Passed all rules", "✅", "Optimal", "#10b981")
        with col_q3:
            render_metric_card("Warnings", f"{val_report['warning_count']} / 8", "Non-critical issues", "⚠️", "Notice", "#f59e0b")
        with col_q4:
            render_metric_card("Failed Checks", f"{val_report['failed_count']} / 8", "Critical anomalies", "❌", "Alert", "#ef4444")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        render_info_box(
            title="Data Quality Scoring Formula & Methodology",
            content="""
            The Data Quality Score is deterministically derived using weighted dimension penalties:
            $$\\text{Score} = 0.30 \\times \\text{Completeness} + 0.20 \\times \\text{Uniqueness} + 0.25 \\times \\text{Validity} + 0.15 \\times \\text{Consistency} + 0.10 \\times \\text{Integrity}$$
            • **Grade A (≥90%):** ML Ready — Exceptional hygiene and integrity.<br/>
            • **Grade B (75-89%):** Minor warnings detected — Suitable for exploratory modeling.<br/>
            • **Grade C (60-74%):** Moderate liabilities — Preprocessing required before model training.<br/>
            • **Grade D (<60%):** Critical failures — Unsuitable for production algorithms.
            """,
            icon="📐",
            box_type="info"
        )

        st.markdown("#### Validation Check Registry")
        st.dataframe(val_report["checks_df"][["check_id", "name", "status", "score", "details"]], use_container_width=True)

        if st.button("Mark Dataset as Validated", type="primary", use_container_width=True):
            mark_stage_complete("Validation")
            log_action("Validation", f"Validated dataset with Quality Score {val_report['score']}% ({val_report['grade']}).")
            st.success("Dataset successfully validated and recorded in pipeline lineage!")

    # =========================================================================
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    with tab_eda:
        st.subheader("2.1 Domain-Specific E-Commerce Analytics")
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "💰 Sales Analytics",
            "👥 Customer Demographics",
            "📦 Product Intelligence",
            "🚚 Orders & Logistics",
            "🛠️ Custom Dynamic Chart Builder"
        ])

        with eda_tab1:
            fig_sales = plot_sales_trends(working_df)
            if fig_sales:
                st.plotly_chart(fig_sales, use_container_width=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_cat = plot_revenue_by_category(working_df)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("Product Category or spend column missing.")
            with col_s2:
                fig_pay = plot_revenue_by_payment_method(working_df)
                if fig_pay:
                    st.plotly_chart(fig_pay, use_container_width=True)
                else:
                    st.info("Payment Method column missing.")

        with eda_tab2:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_age = plot_customer_age_distribution(working_df)
                if fig_age:
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("Age column missing or non-numeric.")
            with col_c2:
                fig_gen = plot_gender_distribution(working_df)
                if fig_gen:
                    st.plotly_chart(fig_gen, use_container_width=True)
                else:
                    st.info("Gender column missing.")

            fig_top = plot_top_customers(working_df, top_n=10)
            if fig_top:
                st.plotly_chart(fig_top, use_container_width=True)

        with eda_tab3:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_best = plot_best_selling_products(working_df)
                if fig_best:
                    st.plotly_chart(fig_best, use_container_width=True)
                else:
                    st.info("Product Name / Quantity column missing.")
            with col_p2:
                fig_cat_share = plot_revenue_by_category(working_df)
                if fig_cat_share:
                    st.plotly_chart(fig_cat_share, use_container_width=True)

        with eda_tab4:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                fig_deliv = plot_delivery_status(working_df)
                if fig_deliv:
                    st.plotly_chart(fig_deliv, use_container_width=True)
                else:
                    st.info("Delivery Status column missing.")
            with col_o2:
                fig_sales_vol = plot_sales_trends(working_df)
                if fig_sales_vol:
                    st.plotly_chart(fig_sales_vol, use_container_width=True)

        with eda_tab5:
            st.markdown("#### Dynamic Interactive Chart Builder")
            st.caption("Plot any columns dynamically across various statistical chart types.")
            
            all_cols = list(working_df.columns)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Violin Plot", "Pie Chart"])
            with col_b2:
                x_axis = st.selectbox("X-Axis Feature:", options=all_cols)
            with col_b3:
                y_opts = ["None"] + all_cols
                y_axis = st.selectbox("Y-Axis Feature:", options=y_opts, index=min(2, len(y_opts)-1))
            with col_b4:
                color_opts = ["None"] + all_cols
                hue_col = st.selectbox("Color / Hue:", options=color_opts, index=0)

            actual_y = None if y_axis == "None" else y_axis
            actual_color = None if hue_col == "None" else hue_col

            fig_custom = build_custom_chart(working_df, chart_type, x_axis, y_col=actual_y, color_col=actual_color)
            if fig_custom:
                st.plotly_chart(fig_custom, use_container_width=True)
            else:
                st.warning("Could not render the requested chart configuration. Please verify feature types.")

    # =========================================================================
    # TAB 3: ML READINESS & PARTITIONING
    # =========================================================================
    with tab_ml:
        st.subheader("3.1 Machine Learning Partitioning & Leakage Prevention")
        render_info_box(
            title="Zero Data Leakage Architecture",
            content="To maintain strict statistical separation between training and test sets, all feature scaling (StandardScaler) and imputation parameters are fitted strictly on the training partition (X_train) and applied out-of-sample to the testing partition (X_test).",
            icon="🔒",
            box_type="success"
        )

        all_cols = list(working_df.columns)
        candidate_targets = [c for c in all_cols if not any(x in c.lower() for x in ["id", "address", "name"])]

        col_ml1, col_ml2, col_ml3 = st.columns(3)
        with col_ml1:
            target_col = st.selectbox(
                "Select Categorical Target (y):",
                options=candidate_targets,
                index=candidate_targets.index("Delivery Status") if "Delivery Status" in candidate_targets else 0
            )
        with col_ml2:
            default_feats = [c for c in all_cols if c != target_col and not any(x in c.lower() for x in ["id", "address", "name"])]
            feature_cols = st.multiselect(
                "Select Input Features (X):",
                options=[c for c in all_cols if c != target_col],
                default=default_feats[:min(6, len(default_feats))]
            )
        with col_ml3:
            split_ratio = st.selectbox(
                "Train-Test Split Ratio:",
                options=["80:20 (Standard)", "75:25 (Balanced)", "70:30 (Thorough Test)"],
                index=0
            )

        test_fraction = 0.20 if "80:20" in split_ratio else (0.25 if "75:25" in split_ratio else 0.30)

        if st.button("🚀 Prepare & Finalize ML-Ready Dataset", type="primary", use_container_width=True):
            ml_data, err = prepare_ml_dataset(
                df=working_df,
                target_col=target_col,
                feature_cols=feature_cols,
                test_size=test_fraction,
                random_state=42
            )
            if err:
                st.error(err)
            else:
                st.session_state.ml_data = ml_data
                mark_stage_complete("ML Ready")
                log_action("ML Readiness", f"Partitioned data for target '{target_col}' ({ml_data['train_size']} train, {ml_data['test_size']} test).")
                st.success(f"ML Dataset finalized! Train rows: {ml_data['train_size']} | Test rows: {ml_data['test_size']} | Target classes: {ml_data['class_names']}.")

        # ML Readiness Checklist
        st.markdown("#### ✅ ML Readiness Pre-Flight Checklist")
        ml_data = st.session_state.get("ml_data")
        is_ready = ml_data is not None

        c_item1 = "✅" if is_ready else "⬜"
        c_item2 = "✅" if (working_df.isna().sum().sum() == 0 or is_ready) else "⬜"
        c_item3 = "✅" if (working_df.duplicated().sum() == 0 or is_ready) else "⬜"
        c_item4 = "✅" if is_ready else "⬜"
        c_item5 = "✅" if is_ready else "⬜"

        st.markdown(f"""
        - {c_item1} **Target variable selected and encoded.**
        - {c_item2} **Missing value imputation completed.**
        - {c_item3} **Duplicate entries checked and resolved.**
        - {c_item4} **Numerical features scaled with out-of-sample parameter isolation.**
        - {c_item5} **Leakage-free Train/Test split formed ({split_ratio if is_ready else 'Pending'}).**
        """)

        if is_ready:
            st.markdown("#### Preview Scaled Training Features (X_train)")
            st.dataframe(ml_data["X_train_df"].head(6), use_container_width=True)

    # =========================================================================
    # MODULE 3 EXPORTS
    # =========================================================================
    st.divider()
    st.subheader("📥 Export Module 3 Deliverables")
    ml_data = st.session_state.get("ml_data")
    if ml_data is not None:
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.download_button(
                label="📄 Download X_train (CSV)",
                data=dataframe_to_csv_bytes(ml_data["X_train_df"]),
                file_name="X_train_features.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_ex2:
            st.download_button(
                label="📄 Download X_test (CSV)",
                data=dataframe_to_csv_bytes(ml_data["X_test_df"]),
                file_name="X_test_features.csv",
                mime="text/csv",
                use_container_width=True
            )
