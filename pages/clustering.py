"""
Page: Customer Segmentation via Clustering
Implements unsupervised K-Means customer segmentation:
- Feature scaling
- Elbow Method & Silhouette Score curve
- Dynamic cluster persona generation
- 2D / 3D scatter plots
- Segment profiling and customer lifetime value insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.helpers import render_page_header, render_metric_card, render_info_box
from utils.session_manager import get_current_working_df, log_action
from utils.file_handler import dataframe_to_csv_bytes, dataframe_to_excel_bytes

from modules.clustering_models import (
    compute_clustering_metrics, perform_kmeans_segmentation,
    plot_elbow_and_silhouette, plot_cluster_scatter
)


def render():
    render_page_header(
        title="Unsupervised Customer Segmentation",
        subtitle="Segment customer cohorts using K-Means clustering, optimize K via Elbow & Silhouette metrics, and synthesize behavioral personas.",
        icon="👥",
        stage_badge="Clustering"
    )

    working_df = get_current_working_df()
    if working_df is None:
        st.warning("⚠️ No dataset loaded. Please ingest and clean a dataset in Module 1 first.")
        return

    num_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out IDs
    clean_num_cols = [c for c in num_cols if not any(x in c.lower() for x in ["id", "postal", "zip"])]

    if len(clean_num_cols) < 2:
        st.error("Clustering requires at least 2 numerical features. Current dataset does not contain sufficient numeric columns.")
        return

    # Configuration Section
    st.subheader("1. Feature Selection & Cluster Optimization")
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        default_cluster_features = [c for c in clean_num_cols if any(k in c.lower() for k in ["spend", "price", "amount", "age", "freq", "rating", "quantity"])]
        if len(default_cluster_features) < 2:
            default_cluster_features = clean_num_cols[:min(4, len(clean_num_cols))]

        selected_features = st.multiselect(
            "Select Numerical Features for Customer Profiling:",
            options=clean_num_cols,
            default=default_cluster_features
        )

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 numerical features to proceed with clustering.")
            return

    with col_f2:
        st.markdown("#### Optimal K Exploration")
        run_opt = st.button("🔍 Evaluate Elbow & Silhouette (k=2..8)", use_container_width=True)

    # Calculate optimal k metrics
    if run_opt or "clustering_metrics" not in st.session_state:
        if len(selected_features) >= 2:
            metrics, err = compute_clustering_metrics(working_df, selected_features)
            if not err:
                st.session_state.clustering_metrics = metrics
            else:
                st.error(err)

    if "clustering_metrics" in st.session_state:
        metrics = st.session_state.clustering_metrics
        st.markdown(f"**Recommended Cluster Count (Max Silhouette Score):** `k = {metrics['recommended_k']}`")
        fig_opt = plot_elbow_and_silhouette(metrics)
        st.plotly_chart(fig_opt, use_container_width=True)

    st.divider()

    # K-Means Execution
    st.subheader("2. Execute K-Means Clustering")
    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        rec_k = st.session_state.get("clustering_metrics", {}).get("recommended_k", 3)
        chosen_k = st.slider("Select Cluster Count (K):", min_value=2, max_value=8, value=int(rec_k))
        exec_cluster = st.button("🚀 Cluster Customers", type="primary", use_container_width=True)

    with col_k2:
        render_info_box(
            title="Dynamic Persona Synthesis",
            content="Customer personas are generated strictly from the relative deviation of cluster centroids against overall population means, establishing accurate data-driven customer archetypes.",
            icon="💡",
            box_type="info"
        )

    if exec_cluster or st.session_state.get("clustering_results") is not None:
        if exec_cluster:
            clustered_df, cl_summary, cl_err = perform_kmeans_segmentation(
                df=working_df,
                feature_cols=selected_features,
                n_clusters=chosen_k
            )
            if cl_err:
                st.error(cl_err)
                return
            st.session_state.clustering_results = cl_summary
            st.session_state.clustering_df = clustered_df
            log_action("Clustering", f"Partitioned data into {chosen_k} customer clusters (Silhouette: {cl_summary['silhouette_score']}).")
            st.success(f"Customer segmentation complete! Partitioned into {chosen_k} cohorts.")

        cl_summary = st.session_state.clustering_results
        clustered_df = st.session_state.clustering_df

        st.divider()
        st.subheader("3. Cluster Profiles & Synthesized Archetypes")

        # Metric cards for segments
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            render_metric_card("Active Clusters", f"{cl_summary['n_clusters']}", "K-Means Cohorts", "👥", "K", "#06b6d4")
        with col_m2:
            render_metric_card("Silhouette Score", f"{cl_summary['silhouette_score']:.4f}", "Separation Quality", "📐", "Validation", "#10b981")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Behavioral Centroids & Segment Profile")
        st.dataframe(cl_summary["profile_table"], use_container_width=True)

        st.divider()

        # Scatter Visualizations
        st.subheader("4. Cluster Scatter Visualizations")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            x_feat = st.selectbox("X-Axis Feature:", options=selected_features, index=0)
        with col_sc2:
            y_feat = st.selectbox("Y-Axis Feature:", options=selected_features, index=min(1, len(selected_features)-1))

        fig_cluster = plot_cluster_scatter(clustered_df, x_feat, y_feat)
        st.plotly_chart(fig_cluster, use_container_width=True)

        # PCA 2D Scatter if available
        if "PCA1" in clustered_df.columns and "PCA2" in clustered_df.columns:
            st.markdown("#### Principal Component Projection of Segments (PCA1 vs PCA2)")
            fig_pca_cl = px.scatter(
                clustered_df,
                x="PCA1",
                y="PCA2",
                color="Segment_Persona",
                hover_data=["Cluster"] + ([ "Customer ID" ] if "Customer ID" in clustered_df.columns else []),
                color_discrete_sequence=["#06b6d4", "#f59e0b", "#10b981", "#a855f7", "#ec4899", "#3b82f6"]
            )
            fig_pca_cl.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font=dict(color="#cbd5e1"), height=380)
            st.plotly_chart(fig_pca_cl, use_container_width=True)

        # Download Segmented Data
        st.divider()
        st.subheader("📥 Export Customer Segmentation Deliverables")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📄 Download Segmented Customers (CSV)",
                data=dataframe_to_csv_bytes(clustered_df),
                file_name="customer_segmentation_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="📊 Download Segment Profile Table (Excel)",
                data=dataframe_to_excel_bytes(cl_summary["profile_table"]),
                file_name="segment_profile_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
