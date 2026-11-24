"""
Module 1: Data Cleaning
Provides robust data cleaning methods:
- Missing value analysis & imputation (drop, mean, median, mode, constant)
- Duplicate detection & removal
- Outlier detection (IQR, Z-Score) and treatment (remove, cap, keep)
- Data type conversions
- Inconsistent categorical text standardization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe detailing missing values per column.
    """
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / total_rows * 100.0).round(2)
    dtypes = df.dtypes

    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(d) for d in dtypes],
        "Missing Count": missing_counts.values,
        "Missing Percentage (%)": missing_pct.values
    })
    return report.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    strategy_dict: Dict[str, Dict[str, Any]],
    drop_threshold_pct: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles missing values column-by-column based on user strategy.
    strategy_dict example:
      {
        "Age": {"action": "median"},
        "Gender": {"action": "mode"},
        "Price": {"action": "mean"},
        "Review Rating": {"action": "constant", "value": 3.0},
        "Product Category": {"action": "drop_rows"}
      }
    """
    df_cleaned = df.copy()
    actions_taken = []
    
    # 1. Drop columns exceeding drop_threshold_pct if specified
    if drop_threshold_pct is not None and drop_threshold_pct > 0:
        total_rows = len(df_cleaned)
        dropped_cols = []
        for col in df_cleaned.columns:
            pct = (df_cleaned[col].isna().sum() / total_rows) * 100.0
            if pct > drop_threshold_pct:
                dropped_cols.append(col)
        if dropped_cols:
            df_cleaned.drop(columns=dropped_cols, inplace=True)
            actions_taken.append(f"Dropped columns with >{drop_threshold_pct}% missing values: {', '.join(dropped_cols)}")

    # 2. Column-specific treatments
    for col, strat in strategy_dict.items():
        if col not in df_cleaned.columns:
            continue
            
        action = strat.get("action", "keep")
        val = strat.get("value", None)
        null_count_before = int(df_cleaned[col].isna().sum())
        
        if null_count_before == 0:
            continue
            
        if action == "drop_rows":
            rows_before = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=[col]).reset_index(drop=True)
            dropped = rows_before - len(df_cleaned)
            actions_taken.append(f"Dropped {dropped} row(s) with missing '{col}'")
            
        elif action == "mean":
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                fill_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                actions_taken.append(f"Imputed '{col}' ({null_count_before} nulls) with Mean ({round(fill_val, 2)})")
                
        elif action == "median":
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                fill_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                actions_taken.append(f"Imputed '{col}' ({null_count_before} nulls) with Median ({round(fill_val, 2)})")
                
        elif action == "mode":
            mode_vals = df_cleaned[col].mode(dropna=True)
            if not mode_vals.empty:
                fill_val = mode_vals.iloc[0]
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                actions_taken.append(f"Imputed '{col}' ({null_count_before} nulls) with Mode ('{fill_val}')")
                
        elif action == "constant":
            if val is not None:
                # Cast to numeric if target column is numeric
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                df_cleaned[col] = df_cleaned[col].fillna(val)
                actions_taken.append(f"Imputed '{col}' ({null_count_before} nulls) with Constant ('{val}')")

    summary = {
        "rows_before": len(df),
        "rows_after": len(df_cleaned),
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": int(df_cleaned.isna().sum().sum()),
        "actions": actions_taken
    }
    return df_cleaned, summary


def detect_and_remove_duplicates(
    df: pd.DataFrame,
    subset_cols: Optional[List[str]] = None,
    keep: str = "first",
    remove: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies and optionally removes duplicate records.
    keep: 'first', 'last', or False
    """
    cols = subset_cols if subset_cols and len(subset_cols) > 0 else None
    dup_mask = df.duplicated(subset=cols, keep=False)
    dup_count = int(df.duplicated(subset=cols, keep=keep).sum())
    
    duplicated_rows_df = df[dup_mask].copy()

    df_result = df.copy()
    if remove and dup_count > 0:
        df_result = df_result.drop_duplicates(subset=cols, keep=keep).reset_index(drop=True)

    stats = {
        "initial_rows": len(df),
        "duplicate_rows": dup_count,
        "final_rows": len(df_result),
        "removed": (len(df) - len(df_result)) if remove else 0,
        "duplicated_samples": duplicated_rows_df.head(20)
    }
    return df_result, stats


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5
) -> Dict[str, Any]:
    """
    Computes IQR-based bounds and identifies outlier rows for a numeric column.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df[outlier_mask].index.tolist()
    outlier_count = len(outlier_indices)

    return {
        "column": column,
        "method": "IQR",
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / len(df) * 100.0, 2) if len(df) > 0 else 0.0,
        "outlier_indices": outlier_indices
    }


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Computes Z-Score bounds and identifies outlier rows for a numeric column.
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    series = df[column].dropna()
    mean = series.mean()
    std = series.std()
    
    if std == 0 or np.isnan(std):
        return {"column": column, "method": "Z-Score", "outlier_count": 0, "outlier_indices": []}

    z_scores = (df[column] - mean) / std
    outlier_mask = z_scores.abs() > threshold
    outlier_indices = df[outlier_mask].index.tolist()
    outlier_count = len(outlier_indices)
    
    lower_bound = mean - (threshold * std)
    upper_bound = mean + (threshold * std)

    return {
        "column": column,
        "method": "Z-Score",
        "mean": round(float(mean), 2),
        "std": round(float(std), 2),
        "threshold": threshold,
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_count / len(df) * 100.0, 2) if len(df) > 0 else 0.0,
        "outlier_indices": outlier_indices
    }


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    action: str,  # "remove", "cap", "keep"
    method: str = "IQR",  # "IQR" or "Z-Score"
    factor_or_threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies the chosen outlier treatment on the target numerical column.
    """
    df_result = df.copy()
    if method == "IQR":
        res = detect_outliers_iqr(df_result, column, factor=factor_or_threshold)
    else:
        res = detect_outliers_zscore(df_result, column, threshold=factor_or_threshold)

    if "error" in res or res.get("outlier_count", 0) == 0:
        return df_result, {"status": "no_outliers", "details": res}

    lower_bound = res["lower_bound"]
    upper_bound = res["upper_bound"]
    indices = res["outlier_indices"]

    if action == "remove":
        df_result = df_result.drop(index=indices).reset_index(drop=True)
        message = f"Removed {len(indices)} outlier rows from '{column}'."
    elif action == "cap":
        df_result[column] = df_result[column].clip(lower=lower_bound, upper=upper_bound)
        message = f"Capped {len(indices)} outlier values in '{column}' between [{lower_bound}, {upper_bound}]."
    else:
        message = f"Kept original values for {len(indices)} detected outliers in '{column}'."

    summary = {
        "column": column,
        "method": method,
        "action": action,
        "outliers_found": len(indices),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "rows_before": len(df),
        "rows_after": len(df_result),
        "message": message
    }
    return df_result, summary


def convert_data_types(
    df: pd.DataFrame,
    conversions: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts column data types:
    conversions example: {"Age": "Integer", "Price": "Float", "Order Date": "DateTime"}
    """
    df_result = df.copy()
    messages = []

    for col, target_type in conversions.items():
        if col not in df_result.columns:
            continue
        try:
            if target_type == "Integer":
                df_result[col] = pd.to_numeric(df_result[col], errors="coerce").round().astype("Int64")
                messages.append(f"Converted '{col}' to Nullable Integer (Int64).")
            elif target_type == "Float":
                df_result[col] = pd.to_numeric(df_result[col], errors="coerce").astype(float)
                messages.append(f"Converted '{col}' to Float.")
            elif target_type == "String":
                df_result[col] = df_result[col].astype(str).replace({"nan": np.nan, "None": np.nan})
                messages.append(f"Converted '{col}' to String.")
            elif target_type == "DateTime":
                df_result[col] = pd.to_datetime(df_result[col], errors="coerce")
                messages.append(f"Converted '{col}' to DateTime.")
            elif target_type == "Category":
                df_result[col] = df_result[col].astype("category")
                messages.append(f"Converted '{col}' to Categorical.")
        except Exception as e:
            messages.append(f"Failed to convert '{col}' to {target_type}: {str(e)}")

    return df_result, messages


def standardize_categorical_data(
    df: pd.DataFrame,
    column: str,
    standardization_type: str = "gender",
    custom_mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes inconsistent categorical values:
    - Gender standardization ('M', 'male', 'Male' -> 'Male'; 'F', 'female', 'Female' -> 'Female')
    - Trim whitespace and Capitalize / Title Case
    - Custom dictionary replacement mapping
    """
    df_result = df.copy()
    if column not in df_result.columns:
        return df_result, {"error": f"Column '{column}' not found."}

    original_counts = df_result[column].value_counts(dropna=False).to_dict()

    if standardization_type == "gender":
        gender_map = {
            "m": "Male", "male": "Male", "M": "Male", "MALE": "Male", "Man": "Male",
            "f": "Female", "female": "Female", "F": "Female", "FEMALE": "Female", "Woman": "Female"
        }
        df_result[column] = df_result[column].astype(str).str.strip().map(
            lambda x: gender_map.get(x, gender_map.get(x.lower(), x)) if x not in ["nan", "None", "<NA>"] else np.nan
        )
    elif standardization_type == "trim_title":
        df_result[column] = df_result[column].astype(str).map(
            lambda x: x.strip().title() if x not in ["nan", "None", "<NA>"] else np.nan
        )
    elif standardization_type == "uppercase":
        df_result[column] = df_result[column].astype(str).map(
            lambda x: x.strip().upper() if x not in ["nan", "None", "<NA>"] else np.nan
        )
    elif standardization_type == "lowercase":
        df_result[column] = df_result[column].astype(str).map(
            lambda x: x.strip().lower() if x not in ["nan", "None", "<NA>"] else np.nan
        )
    elif standardization_type == "custom" and custom_mapping:
        df_result[column] = df_result[column].replace(custom_mapping)

    new_counts = df_result[column].value_counts(dropna=False).to_dict()
    summary = {
        "column": column,
        "type": standardization_type,
        "original_distribution": original_counts,
        "new_distribution": new_counts
    }
    return df_result, summary
