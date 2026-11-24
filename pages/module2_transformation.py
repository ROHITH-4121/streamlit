"""
Page: Module 2 – Data Transformation & Data Reduction
Implements feature normalization, categorical encoding, domain aggregations,
e-commerce feature engineering, discretization, correlation filtering, PCA, and sampling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.helpers import render_page_header, render_info_box, render_workflow_progress
from utils.session_manager import get_current_working_df, mark_stage_complete, log_action
from utils.file_handler import dataframe_to_csv_bytes, dataframe_to_excel_bytes

from modules.data_transformation import (
    scale_numerical_features, encode_categorical_features,
    aggregate_dataset, engineer_ecommerce_features, discretize_feature
)
from modules.data_reduction import (
    compute_correlation_matrix, drop_selected_features,
    apply_pca_reduction, sample_dataset
)


def render():
    render_page_header(
        title="Module 2: Data Transformation & Data Reduction",
        subtitle="Scale numerical ranges, encode categorical dimensions, engineer domain features, and reduce redundancy via PCA & sampling.",
        icon="🔄",
        stage_badge=st.session_state.current_stage
    )

    render_workflow_progress(st.session_state.current_stage, st.session_state.pipeline_stages_completed)

    working_df = get_current_working_df()
    if working_df is None:
        st.warning("⚠️ No dataset loaded. Please ingest and clean a dataset in Module 1 first.")
        return

    # Two Main Tabs: Transformation & Reduction
    tab_trans, tab_reduc = st.tabs([
        "⚡ 1. Data Transformation",
        "📉 2. Data Reduction & Dimensionality"
    ])

    # =========================================================================
    # TAB 1: DATA TRANSFORMATION
    # =========================================================================
    with tab_trans:
        st.subheader("1.1 Feature Normalization & Scaling")
        num_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

        if num_cols:
            col_scale1, col_scale2 = st.columns([2, 1])
            with col_scale1:
                sel_scale_cols = st.multiselect("Select numerical features to scale:", options=num_cols, default=num_cols[:min(3, len(num_cols))])
            with col_scale2:
                scale_method = st.radio("Scaling Algorithm:", ["Min-Max Normalization [0, 1]", "Z-score Standardization (μ=0, σ=1)"])

            method_param = "minmax" if "Min-Max" in scale_method else "standard"
            if st.button("Scale Selected Features", use_container_width=True):
                scaled_df, scale_stats = scale_numerical_features(working_df, sel_scale_cols, method=method_param)
                st.session_state.transformed_df = scaled_df
                mark_stage_complete("Transformation")
                log_action("Transformation", f"Scaled features {sel_scale_cols} using {scale_method}.")
                st.success(f"Applied {scale_method} to {len(sel_scale_cols)} columns!")
                
                # Show comparison table of original vs transformed
                comp_cols = []
                for c in sel_scale_cols:
                    comp_cols.append(c)
                    new_c = f"{c}_minmax" if method_param == "minmax" else f"{c}_standardized"
                    if new_c in scaled_df.columns:
                        comp_cols.append(new_c)
                st.dataframe(scaled_df[comp_cols].head(6), use_container_width=True)

        st.divider()

        # 1.2 Categorical Encoding
        st.subheader("1.2 Categorical Encoding")
        cat_cols = working_df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Exclude IDs and addresses
        clean_cat_cols = [c for c in cat_cols if not any(x in c.lower() for x in ["id", "address", "name"])]
        
        if clean_cat_cols:
            col_enc1, col_enc2, col_enc3 = st.columns(3)
            with col_enc1:
                sel_enc_cols = st.multiselect("Select categorical columns:", options=clean_cat_cols, default=clean_cat_cols[:min(2, len(clean_cat_cols))])
            with col_enc2:
                enc_method = st.selectbox("Encoding Technique:", ["One-Hot Encoding", "Label Encoding"])
            with col_enc3:
                drop_first_opt = st.checkbox("Drop First Category (One-Hot Dummy Trap)", value=False)

            enc_key = "one_hot" if "One-Hot" in enc_method else "label"
            if st.button("Apply Categorical Encoding", use_container_width=True):
                target_base = st.session_state.get("transformed_df", working_df)
                encoded_df, enc_info = encode_categorical_features(target_base, sel_enc_cols, method=enc_key, drop_first=drop_first_opt)
                st.session_state.transformed_df = encoded_df
                mark_stage_complete("Transformation")
                log_action("Transformation", f"Encoded {sel_enc_cols} using {enc_method}.")
                st.success(f"Encoding applied! Current columns count: {len(encoded_df.columns)}.")
                st.dataframe(encoded_df.head(6), use_container_width=True)

        st.divider()

        # 1.3 Feature Engineering
        st.subheader("1.3 Automated E-Commerce Feature Engineering")
        st.markdown("""
        Automatically derives domain-specific high-value machine learning features:
        - **Total Amount:** Price × Quantity
        - **Temporal Decomposition:** Order Year, Month, Day, Day of Week, Is Weekend
        - **Customer Behavioral Analytics:** Purchase Frequency, Total Lifetime Spending, Average Purchase Value
        """)

        if st.button("🚀 Synthesize E-Commerce Features", type="primary", use_container_width=True):
            target_base = st.session_state.get("transformed_df", working_df)
            feat_df, added = engineer_ecommerce_features(target_base)
            st.session_state.transformed_df = feat_df
            mark_stage_complete("Transformation")
            log_action("Transformation", f"Synthesized {len(added)} features: {', '.join(added)}")
            st.success(f"Generated {len(added)} new features: {', '.join(added)}!")
            st.dataframe(feat_df.head(6), use_container_width=True)

        st.divider()

        # 1.4 Discretization / Binning
        st.subheader("1.4 Feature Discretization & Binning")
        curr_num_cols = st.session_state.get("transformed_df", working_df).select_dtypes(include=[np.number]).columns.tolist()
        
        if curr_num_cols:
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                bin_col = st.selectbox("Column to Discretize:", options=curr_num_cols)
            with col_b2:
                bin_method = st.selectbox("Binning Strategy:", ["Equal-Width Binning", "Equal-Frequency (Quantile)"])
            with col_b3:
                num_bins = st.slider("Number of Bins:", min_value=2, max_value=6, value=3)

            bin_key = "equal_width" if "Width" in bin_method else "equal_frequency"
            if st.button(f"Discretize '{bin_col}'", use_container_width=True):
                target_base = st.session_state.get("transformed_df", working_df)
                binned_df, bin_res = discretize_feature(target_base, bin_col, method=bin_key, n_bins=num_bins)
                if "error" not in bin_res:
                    st.session_state.transformed_df = binned_df
                    mark_stage_complete("Transformation")
                    log_action("Transformation", f"Discretized '{bin_col}' into {num_bins} bins.")
                    st.success(f"Discretized '{bin_col}'! Value distribution: {bin_res['distribution']}")
                    st.dataframe(binned_df[[bin_col, f"{bin_col}_binned"]].head(8), use_container_width=True)
                else:
                    st.error(bin_res["error"])

    # =========================================================================
    # TAB 2: DATA REDUCTION
    # =========================================================================
    with tab_reduc:
        target_reduc_base = st.session_state.get("transformed_df", working_df)

        st.subheader("2.1 Feature Selection & Multicollinearity Filtering")
        corr_mat, high_pairs = compute_correlation_matrix(target_reduc_base)

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            if corr_mat is not None:
                st.markdown("#### Pearson Correlation Matrix")
                fig_corr = px.imshow(
                    corr_mat,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )
                fig_corr.update_layout(
                    paper_bgcolor="#1e293b",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#cbd5e1"),
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        with col_c2:
            st.markdown("#### Collinearity Diagnostics (|r| ≥ 0.75)")
            if high_pairs:
                st.warning(f"Detected {len(high_pairs)} highly correlated feature pairs!")
                for p in high_pairs:
                    st.markdown(f"• **{p['feature_1']}** ↔ **{p['feature_2']}**: `r = {p['correlation']:.2f}` ({p['interpretation']})")
            else:
                st.success("No critical multicollinear pairs detected above threshold.")

        # Manual Feature Drop Selector
        st.markdown("#### Drop Irrelevant or Redundant Columns")
        cols_to_drop = st.multiselect("Select columns to drop:", options=list(target_reduc_base.columns))
        if st.button("Drop Selected Columns", use_container_width=True):
            if cols_to_drop:
                dropped_df, drop_summary = drop_selected_features(target_reduc_base, cols_to_drop)
                st.session_state.reduced_df = dropped_df
                mark_stage_complete("Reduction")
                log_action("Reduction", f"Dropped columns: {cols_to_drop}")
                st.success(f"Dropped {len(cols_to_drop)} columns! Remaining columns: {drop_summary['cols_after']}.")
                st.rerun()

        st.divider()

        # 2.2 Dimensionality Reduction (PCA)
        st.subheader("2.2 Dimensionality Reduction: Principal Component Analysis (PCA)")
        curr_active = st.session_state.get("reduced_df", target_reduc_base)
        num_pca_cols = curr_active.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_pca_cols) >= 2:
            col_pca1, col_pca2 = st.columns([2, 1])
            with col_pca1:
                pca_features = st.multiselect("Select numerical features for PCA:", options=num_pca_cols, default=num_pca_cols[:min(5, len(num_pca_cols))])
            with col_pca2:
                if len(pca_features) >= 2:
                    max_k = min(len(pca_features), 5)
                    n_comp = st.slider("Number of Principal Components:", min_value=2, max_value=max_k, value=min(2, max_k))
                else:
                    st.info("Select at least 2 features for PCA.")
                    n_comp = 2

            if len(pca_features) >= 2 and st.button("Execute PCA Decomposition", use_container_width=True):
                pca_result_df, pca_meta = apply_pca_reduction(curr_active, pca_features, n_components=n_comp)
                if pca_result_df is not None:
                    st.session_state.reduced_df = pca_result_df
                    mark_stage_complete("Reduction")
                    log_action("Reduction", f"Applied PCA: reduced {pca_meta['original_features_count']} features to {n_comp} components ({pca_meta['total_variance_explained_pct']}% variance).")
                    
                    st.success(f"PCA completed! Total variance explained by {n_comp} components: **{pca_meta['total_variance_explained_pct']}%**.")

                    # Scree plot & 2D Projection
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        # Scree Plot
                        fig_scree = px.bar(
                            x=[f"PC{i+1}" for i in range(n_comp)],
                            y=pca_meta["explained_variance_ratio"],
                            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
                            title="Scree Plot: Variance per Component",
                            color_discrete_sequence=["#06b6d4"]
                        )
                        fig_scree.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), height=300)
                        st.plotly_chart(fig_scree, use_container_width=True)

                    with col_p2:
                        # 2D PCA Scatter
                        fig_scatter = px.scatter(
                            pca_result_df,
                            x="PC1",
                            y="PC2",
                            title="2D Projection: PC1 vs PC2",
                            color_discrete_sequence=["#38bdf8"]
                        )
                        fig_scatter.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), height=300)
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    st.markdown("#### PCA Component Loadings")
                    st.dataframe(pca_meta["loadings"], use_container_width=True)
                else:
                    st.error(pca_meta.get("error", "PCA failed."))

        st.divider()

        # 2.3 Dataset Sampling
        st.subheader("2.3 Dataset Sampling")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            samp_method = st.selectbox("Sampling Method:", ["Simple Random Sampling", "Stratified Sampling", "Systematic Sampling"])
        with col_s2:
            samp_frac = st.slider("Sample Fraction:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        with col_s3:
            strata_column = None
            if "Stratified" in samp_method:
                cat_choices = curr_active.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_choices:
                    strata_column = st.selectbox("Stratification Column:", options=cat_choices)
                else:
                    st.caption("No categorical columns found for stratification.")

        samp_key = "random" if "Random" in samp_method else ("stratified" if "Stratified" in samp_method else "systematic")
        if st.button("Apply Sampling", use_container_width=True):
            sampled_df, samp_summary = sample_dataset(curr_active, method=samp_key, fraction=samp_frac, strata_col=strata_column)
            if "error" not in samp_summary:
                st.session_state.reduced_df = sampled_df
                mark_stage_complete("Reduction")
                log_action("Reduction", f"Sampled dataset via {samp_method}: {samp_summary['original_rows']} → {samp_summary['sampled_rows']} rows.")
                st.success(f"Dataset sampled! {samp_summary['original_rows']} → {samp_summary['sampled_rows']} rows ({samp_summary['reduction_pct']}% volume reduction).")
                st.rerun()
            else:
                st.error(samp_summary["error"])

    # =========================================================================
    # MODULE 2 EXPORTS
    # =========================================================================
    st.divider()
    st.subheader("📥 Export Module 2 Deliverables")
    final_m2_df = st.session_state.get("reduced_df", st.session_state.get("transformed_df", working_df))
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            label="📄 Download Transformed/Reduced Data (CSV)",
            data=dataframe_to_csv_bytes(final_m2_df),
            file_name="transformed_reduced_ecommerce.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_e2:
        st.download_button(
            label="📊 Download Transformed/Reduced Data (Excel)",
            data=dataframe_to_excel_bytes(final_m2_df),
            file_name="transformed_reduced_ecommerce.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
