"""
Module 2: Data Reduction
Provides dimensionality reduction and dataset volume compression:
- Feature Selection & Multicollinearity filtering via Correlation Matrix
- Principal Component Analysis (PCA) with Scree & Cumulative Variance analysis
- Sampling: Simple Random, Stratified, and Systematic Sampling
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional


def compute_correlation_matrix(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
    """
    Computes Pearson correlation matrix for numeric columns and detects highly correlated pairs (|r| > 0.80).
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None, []

    corr_matrix = numeric_df.corr().round(3)
    
    # Identify high correlation pairs
    high_corr_pairs = []
    cols = numeric_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = corr_matrix.loc[col1, col2]
            if abs(val) >= 0.75:
                high_corr_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": float(val),
                    "interpretation": "Strong Positive Collinearity" if val > 0 else "Strong Negative Collinearity"
                })

    return corr_matrix, high_corr_pairs


def drop_selected_features(df: pd.DataFrame, columns_to_drop: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drops irrelevant or redundant columns from dataset.
    """
    df_result = df.copy()
    valid_drops = [c for c in columns_to_drop if c in df_result.columns]
    
    if valid_drops:
        df_result.drop(columns=valid_drops, inplace=True)

    summary = {
        "dropped_columns": valid_drops,
        "cols_before": len(df.columns),
        "cols_after": len(df_result.columns)
    }
    return df_result, summary


def apply_pca_reduction(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_components: int = 2
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Performs Principal Component Analysis (PCA) on specified numerical features.
    Standardizes data prior to decomposition to prevent high-scale dominance.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return None, {"error": "PCA requires at least 2 numeric features."}

    n_comp = min(n_components, len(valid_cols), len(df))
    if n_comp < 1:
        return None, {"error": "Invalid component count."}

    X = df[valid_cols].fillna(df[valid_cols].mean()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    comp_cols = [f"PC{i+1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(X_pca, columns=comp_cols, index=df.index)

    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cum_var = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=comp_cols,
        index=valid_cols
    ).round(4)

    summary = {
        "original_features_count": len(valid_cols),
        "reduced_components_count": n_comp,
        "explained_variance_ratio": explained_var,
        "cumulative_explained_variance": cum_var,
        "total_variance_explained_pct": round(cum_var[-1] * 100.0, 2),
        "loadings": loadings,
        "features_used": valid_cols
    }

    # Combined dataframe preserving other columns
    combined_df = pd.concat([df.drop(columns=valid_cols, errors="ignore"), pca_df], axis=1)

    return combined_df, summary


def sample_dataset(
    df: pd.DataFrame,
    method: str = "random",  # "random", "stratified", "systematic"
    sample_size: Optional[int] = None,
    fraction: Optional[float] = None,
    strata_col: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs dataset sampling:
    - Random Sampling: by row count or fraction
    - Stratified Sampling: preserves class frequencies of strata_col
    - Systematic Sampling: selects every k-th row
    """
    total_rows = len(df)
    if total_rows == 0:
        return df, {"error": "Empty dataset."}

    df_sample = df.copy()

    if method == "random":
        if fraction is not None and 0.0 < fraction <= 1.0:
            df_sample = df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
        elif sample_size is not None and 0 < sample_size <= total_rows:
            df_sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        else:
            df_sample = df.sample(frac=0.5, random_state=random_state).reset_index(drop=True)

    elif method == "stratified":
        if not strata_col or strata_col not in df.columns:
            return df, {"error": "Valid stratification column required."}
        
        frac = fraction if (fraction is not None and 0 < fraction <= 1.0) else (sample_size / total_rows if sample_size else 0.5)
        # Drop rows with NaN in strata column for safe grouping
        valid_idx = df[strata_col].dropna().index
        df_valid = df.loc[valid_idx]
        
        try:
            sample_indices = []
            for _, group in df_valid.groupby(strata_col):
                n = max(1, int(np.round(len(group) * frac)))
                sample_indices.extend(group.sample(min(n, len(group)), random_state=random_state).index)
            df_sample = df.loc[sample_indices].reset_index(drop=True)
        except Exception:
            # Fallback if a stratum has too few records
            df_sample = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    elif method == "systematic":
        n_target = sample_size if sample_size and sample_size <= total_rows else int(total_rows * (fraction or 0.5))
        n_target = max(1, n_target)
        k = max(1, total_rows // n_target)
        start_idx = np.random.RandomState(random_state).randint(0, k)
        sample_indices = list(range(start_idx, total_rows, k))[:n_target]
        df_sample = df.iloc[sample_indices].reset_index(drop=True)

    summary = {
        "method": method.capitalize(),
        "original_rows": total_rows,
        "sampled_rows": len(df_sample),
        "reduction_pct": round((1.0 - len(df_sample) / total_rows) * 100.0, 2)
    }

    return df_sample, summary
