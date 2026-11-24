"""
Module: Customer Segmentation via K-Means Clustering
Performs unsupervised customer segmentation:
- Feature scaling via StandardScaler
- Elbow Method (Inertia analysis for k=2..8)
- Silhouette Score evaluation for optimal cluster validation
- K-Means model fitting and 2D/3D scatter visualization
- Automated, data-driven customer persona profiling
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional


def compute_clustering_metrics(
    df: pd.DataFrame,
    feature_cols: List[str],
    k_range: range = range(2, 9)
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Computes Inertia (Elbow Method) and Silhouette Scores for a range of k values.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return None, "Clustering requires at least 2 numerical features."

    clean_df = df[valid_cols].dropna().copy()
    if len(clean_df) < 15:
        return None, "Insufficient data points (< 15) for meaningful clustering."

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df)

    inertias = []
    silhouette_scores_list = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        score = silhouette_score(X_scaled, km.labels_)
        silhouette_scores_list.append(round(float(score), 4))

    # Identify best k by silhouette score
    best_k_idx = int(np.argmax(silhouette_scores_list))
    recommended_k = k_values[best_k_idx]

    return {
        "k_values": k_values,
        "inertias": [round(x, 2) for x in inertias],
        "silhouette_scores": silhouette_scores_list,
        "recommended_k": recommended_k,
        "features_used": valid_cols,
        "records_count": len(clean_df)
    }, None


def perform_kmeans_segmentation(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Fits K-Means on scaled numerical features and appends 'Cluster' and 'Segment_Persona'
    to the dataframe, along with comprehensive profile metrics.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return None, None, "Clustering requires at least 2 numerical features."

    clean_df = df.copy()
    # Fill missing with median for clustering
    X_imputed = clean_df[valid_cols].fillna(clean_df[valid_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    clean_df["Cluster"] = [f"Cluster {c}" for c in clusters]
    clean_df["Cluster_ID"] = clusters

    # 2D PCA projection for visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    clean_df["PCA1"] = pca_coords[:, 0].round(3)
    clean_df["PCA2"] = pca_coords[:, 1].round(3)

    # Compute overall and cluster-specific means
    overall_means = X_imputed.mean()
    cluster_means = clean_df.groupby("Cluster")[valid_cols].mean().round(2)
    cluster_sizes = clean_df["Cluster"].value_counts().to_dict()

    # Dynamic data-driven persona naming
    personas = {}
    for c_name in cluster_means.index:
        c_series = cluster_means.loc[c_name]
        diff_ratios = (c_series - overall_means) / (overall_means.abs() + 1e-6)
        
        # Determine highest and lowest dominant attributes
        top_positive = diff_ratios.idxmax()
        top_negative = diff_ratios.idxmin()
        
        pos_val = diff_ratios[top_positive]
        neg_val = diff_ratios[top_negative]
        
        if pos_val > 0.25 and any(k in top_positive.lower() for k in ["spend", "price", "amount"]):
            persona = "High-Value Spenders"
        elif pos_val > 0.25 and any(k in top_positive.lower() for k in ["freq", "qty", "count"]):
            persona = "High-Volume Loyalists"
        elif neg_val < -0.25 and any(k in top_negative.lower() for k in ["spend", "price", "amount"]):
            persona = "Budget-Conscious Shoppers"
        elif "age" in top_positive.lower() and pos_val > 0.20:
            persona = "Mature Demographic Buyers"
        elif "age" in top_negative.lower() and neg_val < -0.20:
            persona = "Young Adult Shoppers"
        else:
            persona = f"Moderate Segment ({c_name})"

        personas[c_name] = persona

    clean_df["Segment_Persona"] = clean_df["Cluster"].map(personas)

    # Compile cluster profile statistics
    profile_df = cluster_means.copy()
    profile_df.insert(0, "Persona", [personas.get(c, "Standard") for c in profile_df.index])
    profile_df.insert(1, "Customer Count", [cluster_sizes.get(c, 0) for c in profile_df.index])
    profile_df.insert(2, "Share (%)", [round(cluster_sizes.get(c, 0) / len(clean_df) * 100.0, 1) for c in profile_df.index])

    summary = {
        "n_clusters": n_clusters,
        "silhouette_score": round(float(silhouette_score(X_scaled, clusters)), 4),
        "cluster_sizes": cluster_sizes,
        "personas": personas,
        "profile_table": profile_df.reset_index(),
        "features_used": valid_cols
    }

    return clean_df, summary, None


def plot_elbow_and_silhouette(metrics: Dict[str, Any]) -> go.Figure:
    """Displays side-by-side plots for the Elbow Method and Silhouette Analysis."""
    fig = go.Figure()

    # Inertia line (Elbow)
    fig.add_trace(go.Scatter(
        x=metrics["k_values"],
        y=metrics["inertias"],
        name="Inertia (Elbow Method)",
        mode="lines+markers",
        line=dict(color="#06b6d4", width=3),
        marker=dict(size=8, symbol="diamond")
    ))

    # Silhouette Score on secondary y-axis
    fig.add_trace(go.Scatter(
        x=metrics["k_values"],
        y=metrics["silhouette_scores"],
        name="Silhouette Score",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#ec4899", width=3, dash="dot"),
        marker=dict(size=8, symbol="circle")
    ))

    fig.update_layout(
        title={"text": "<b>Optimal Cluster Count (K) Evaluation: Elbow vs Silhouette</b>", "font": {"color": "#f8fafc", "size": 15}},
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        xaxis=dict(title="Number of Clusters (k)", gridcolor="#334155", tickmode="linear", tick0=2, dtick=1),
        yaxis=dict(title="Inertia (Sum of Squared Distances)", gridcolor="#334155", title_font=dict(color="#06b6d4")),
        yaxis2=dict(
            title="Silhouette Score",
            overlaying="y",
            side="right",
            gridcolor="#334155",
            title_font=dict(color="#ec4899"),
            range=[0, 1]
        ),
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
    )
    return fig


def plot_cluster_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str
) -> go.Figure:
    """Generates an interactive scatter plot highlighting cluster boundaries."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="Segment_Persona" if "Segment_Persona" in df.columns else "Cluster",
        hover_data=["Cluster", "Customer ID"] if "Customer ID" in df.columns else ["Cluster"],
        color_discrete_sequence=["#06b6d4", "#f59e0b", "#10b981", "#a855f7", "#ec4899", "#3b82f6"]
    )
    fig.update_layout(
        title={"text": f"<b>Customer Segmentation: {x_col} vs {y_col}</b>", "font": {"color": "#f8fafc", "size": 15}},
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        height=420,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
