"""
Module 1: Data Collection & Ingestion
Handles reading CSV, Excel, and JSON files, loading sample datasets,
and providing comprehensive structural summary statistics.
"""

import io
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Reads an uploaded file (Streamlit UploadedFile) into a Pandas DataFrame.
    Supports CSV, Excel (.xlsx, .xls), and JSON.
    """
    if uploaded_file is None:
        return None, "No file uploaded."
    
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            return None, f"Unsupported file format: {uploaded_file.name}. Please upload CSV, Excel, or JSON."
        return df, None
    except Exception as e:
        return None, f"Error reading file '{uploaded_file.name}': {str(e)}"


def load_sample_dataset(dataset_name: str = "ecommerce_main.csv") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads a built-in sample e-commerce dataset from sample_data/ directory.
    """
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
    file_path = os.path.join(sample_dir, dataset_name)
    
    if not os.path.exists(file_path):
        return None, f"Sample file not found at: {file_path}"
    
    try:
        if dataset_name.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif dataset_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        elif dataset_name.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            return None, "Unsupported sample file format."
        return df, None
    except Exception as e:
        return None, f"Error loading sample dataset '{dataset_name}': {str(e)}"


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns high-level structural metrics and column metadata.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "missing_cells": 0,
            "missing_percentage": 0.0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
            "memory_usage_mb": 0.0,
            "column_types": {},
            "columns": []
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_cells = int(df.isna().sum().sum())
    missing_percentage = (missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage = (duplicate_rows / total_rows * 100.0) if total_rows > 0 else 0.0
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    col_details = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round(null_count / total_rows * 100.0, 2)
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        col_details.append({
            "Column": col,
            "Data Type": dtype_str,
            "Null Count": null_count,
            "Null %": null_pct,
            "Unique Values": n_unique,
            "Sample Value": sample_val
        })

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(duplicate_percentage, 2),
        "memory_usage_mb": memory_usage_mb,
        "column_types": col_types,
        "columns": list(df.columns),
        "column_details": pd.DataFrame(col_details)
    }
