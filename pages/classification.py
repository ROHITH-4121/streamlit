"""
Page: Classification Analysis
Trains, compares, and evaluates Logistic Regression, Decision Tree, and Random Forest
on preprocessed, leakage-free e-commerce data with real metrics and confusion heatmaps.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import render_page_header, render_metric_card, render_info_box
from utils.session_manager import get_current_working_df, log_action
from utils.file_handler import dataframe_to_csv_bytes, dataframe_to_excel_bytes

from modules.classification_models import (
    prepare_ml_dataset, train_and_evaluate_classifiers,
    plot_confusion_matrix, plot_model_comparison
)


def render():
    render_page_header(
        title="Supervised Classification Analysis",
        subtitle="Benchmark Logistic Regression, Decision Tree, and Random Forest classifiers on leakage-free preprocessed data.",
        icon="🧠",
        stage_badge="Classification"
    )

    working_df = get_current_working_df()
    if working_df is None:
        st.warning("⚠️ No dataset loaded. Please ingest and clean a dataset in Module 1 first.")
        return

    # Check if ML dataset is already prepared in Module 3 or prepare directly here
    ml_data = st.session_state.get("ml_data")

    with st.expander("⚙️ Dataset Configuration & Target Selection", expanded=(ml_data is None)):
        all_cols = list(working_df.columns)
        candidate_targets = [c for c in all_cols if not any(x in c.lower() for x in ["id", "address", "name"])]
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            target_col = st.selectbox(
                "Target Feature (y):",
                options=candidate_targets,
                index=candidate_targets.index("Delivery Status") if "Delivery Status" in candidate_targets else 0
            )
        with col_c2:
            default_feats = [c for c in all_cols if c != target_col and not any(x in c.lower() for x in ["id", "address", "name"])]
            feature_cols = st.multiselect(
                "Input Features (X):",
                options=[c for c in all_cols if c != target_col],
                default=default_feats[:min(6, len(default_feats))]
            )
        with col_c3:
            split_ratio = st.selectbox("Split Ratio:", ["80:20 (Default)", "75:25", "70:30"])
            test_fraction = 0.20 if "80:20" in split_ratio else (0.25 if "75:25" in split_ratio else 0.30)

        if st.button("Configure & Partition ML Features", use_container_width=True):
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
                st.success(f"Configured dataset with target '{target_col}'!")

    if ml_data is None:
        st.info("Please select a target and input features above, then click 'Configure & Partition ML Features'.")
        return

    # Training trigger
    st.markdown(f"**Target:** `{ml_data['target_col']}` | **Classes ({len(ml_data['class_names'])}):** `{', '.join(ml_data['class_names'])}` | **Train:** `{ml_data['train_size']}` rows | **Test:** `{ml_data['test_size']}` rows")
    
    if st.button("🚀 Train & Evaluate All Classifiers", type="primary", use_container_width=True):
        with st.spinner("Training Logistic Regression, Decision Tree, and Random Forest..."):
            eval_results = train_and_evaluate_classifiers(ml_data)
            st.session_state.classification_results = eval_results
            log_action("Classification", f"Trained 3 classifiers on target '{ml_data['target_col']}'. Best model: {eval_results['best_model']}.")
            st.success("Models trained and evaluated successfully!")

    # Display results if available
    eval_results = st.session_state.get("classification_results")
    if eval_results is not None:
        st.divider()
        st.subheader("1. Algorithm Performance Benchmark")

        comp_df = eval_results["comparison_table"]
        best_model = eval_results["best_model"]
        best_row = comp_df[comp_df["Algorithm"] == best_model].iloc[0]

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            render_metric_card("Top Model", best_model, "Highest F1-Score", "🏆", "Best", "#10b981")
        with col_b2:
            render_metric_card("Accuracy", f"{best_row['Accuracy']:.4f}", f"{best_model}", "🎯", "Score", "#06b6d4")
        with col_b3:
            render_metric_card("Precision (W)", f"{best_row['Precision (Weighted)']:.4f}", f"{best_model}", "⚖️", "Score", "#38bdf8")
        with col_b4:
            render_metric_card("F1-Score (W)", f"{best_row['F1-Score (Weighted)']:.4f}", f"{best_model}", "📈", "Score", "#6366f1")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.dataframe(comp_df, use_container_width=True)

        # Performance comparison chart
        fig_comp = plot_model_comparison(comp_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()

        # Detailed Model Diagnostics & Confusion Matrices
        st.subheader("2. Model Diagnostics & Confusion Matrices")
        model_names_list = list(eval_results["models"].keys())
        model_tabs = st.tabs(model_names_list)

        for tab, m_name in zip(model_tabs, model_names_list):
            m_data = eval_results["models"][m_name]
            with tab:
                col_d1, col_d2 = st.columns([1, 1])
                with col_d1:
                    fig_cm = plot_confusion_matrix(m_data["confusion_matrix"], eval_results["class_names"], m_name)
                    st.plotly_chart(fig_cm, use_container_width=True)

                with col_d2:
                    st.markdown(f"#### Classification Report: {m_name}")
                    cr_dict = m_data["classification_report"]
                    # Render as styled dataframe
                    cr_df = pd.DataFrame(cr_dict).transpose()
                    st.dataframe(cr_df.round(3), use_container_width=True)

        # Download results
        st.divider()
        st.subheader("📥 Export Classification Deliverables")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📄 Download Benchmark Table (CSV)",
                data=dataframe_to_csv_bytes(comp_df),
                file_name="classification_benchmark.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            st.download_button(
                label="📊 Download Benchmark Table (Excel)",
                data=dataframe_to_excel_bytes(comp_df),
                file_name="classification_benchmark.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
