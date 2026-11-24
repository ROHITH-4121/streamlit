"""
Module 2: Data Transformation
Provides functions for:
- Normalization (Min-Max, Z-score Standardization)
- Categorical Encoding (Label, One-Hot, Ordinal)
- Data Aggregation (Group-by operations and pre-built e-commerce metrics)
- Feature Engineering (Total Amount, Date features, RFM & Customer behavioral stats)
- Discretization / Binning (Equal-width, Equal-frequency, Custom rules)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from typing import Dict, Any, List, Tuple, Optional


def scale_numerical_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies Min-Max Normalization or Z-score Standardization to selected columns.
    """
    df_result = df.copy()
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if not valid_cols:
        return df_result, {"error": "No valid numeric columns selected for scaling."}

    stats = {}
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_vals = scaler.fit_transform(df_result[valid_cols].fillna(df_result[valid_cols].mean()))
        for i, col in enumerate(valid_cols):
            orig_min = float(df_result[col].min())
            orig_max = float(df_result[col].max())
            new_col = f"{col}_minmax"
            df_result[new_col] = scaled_vals[:, i].round(4)
            stats[col] = {
                "method": "Min-Max Normalization",
                "transformed_column": new_col,
                "orig_min": round(orig_min, 2),
                "orig_max": round(orig_max, 2),
                "scaled_min": round(float(scaled_vals[:, i].min()), 4),
                "scaled_max": round(float(scaled_vals[:, i].max()), 4)
            }
    else:  # Z-Score Standardization
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df_result[valid_cols].fillna(df_result[valid_cols].mean()))
        for i, col in enumerate(valid_cols):
            orig_mean = float(df_result[col].mean())
            orig_std = float(df_result[col].std())
            new_col = f"{col}_standardized"
            df_result[new_col] = scaled_vals[:, i].round(4)
            stats[col] = {
                "method": "Z-score Standardization",
                "transformed_column": new_col,
                "orig_mean": round(orig_mean, 2),
                "orig_std": round(orig_std, 2),
                "scaled_mean": round(float(scaled_vals[:, i].mean()), 4),
                "scaled_std": round(float(scaled_vals[:, i].std()), 4)
            }

    return df_result, stats


def encode_categorical_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "one_hot",  # "label", "one_hot", "ordinal"
    drop_first: bool = False,
    ordinal_mappings: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encodes categorical features via Label Encoding, One-Hot Encoding, or Ordinal Encoding.
    """
    df_result = df.copy()
    valid_cols = [c for c in columns if c in df.columns]
    
    if not valid_cols:
        return df_result, {"error": "No valid columns selected for encoding."}

    encoding_info = {}

    if method == "label":
        for col in valid_cols:
            le = LabelEncoder()
            # Convert NaN to string placeholder or handle cleanly
            series_str = df_result[col].astype(str)
            new_col = f"{col}_encoded"
            df_result[new_col] = le.fit_transform(series_str)
            mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            encoding_info[col] = {
                "method": "Label Encoding",
                "encoded_column": new_col,
                "classes_count": len(le.classes_),
                "mapping_sample": dict(list(mapping.items())[:10])
            }
            
    elif method == "one_hot":
        for col in valid_cols:
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first, dtype=int)
            new_cols = list(dummies.columns)
            df_result = pd.concat([df_result, dummies], axis=1)
            encoding_info[col] = {
                "method": "One-Hot Encoding",
                "new_columns_created": new_cols,
                "columns_count": len(new_cols)
            }
            
    elif method == "ordinal" and ordinal_mappings:
        for col in valid_cols:
            if col in ordinal_mappings:
                order_list = ordinal_mappings[col]
                mapping = {val: idx for idx, val in enumerate(order_list)}
                new_col = f"{col}_ordinal"
                df_result[new_col] = df_result[col].map(mapping).fillna(-1).astype(int)
                encoding_info[col] = {
                    "method": "Ordinal Encoding",
                    "encoded_column": new_col,
                    "order_mapping": mapping
                }

    return df_result, encoding_info


def aggregate_dataset(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_dict: Dict[str, List[str]]
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Performs flexible group-by aggregation.
    agg_dict example: {"Price": ["sum", "mean"], "Quantity": ["sum"]}
    """
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        return None, {"error": f"Group-by columns missing: {missing_group_cols}"}

    try:
        grouped = df.groupby(group_cols).agg(agg_dict)
        # Flatten MultiIndex columns
        grouped.columns = [f"{col}_{func}" if isinstance(func, str) else f"{col}_{func.__name__}" for col, func in grouped.columns]
        agg_df = grouped.reset_index()

        summary = {
            "group_by": group_cols,
            "agg_rules": {k: v for k, v in agg_dict.items()},
            "result_rows": len(agg_df),
            "result_cols": len(agg_df.columns)
        }
        return agg_df, summary
    except Exception as e:
        return None, {"error": f"Aggregation error: {str(e)}"}


def engineer_ecommerce_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Automates extraction of domain-specific e-commerce features:
    - Total Amount = Price * Quantity
    - Order Date parsing: Year, Month, Day, DayOfWeek, Is_Weekend
    - Customer Level RFM / Behavioral features (Total Spending, Purchase Frequency, Average Purchase Value)
    """
    df_feat = df.copy()
    added_features = []

    # 1. Total Amount = Price * Quantity
    if "Price" in df_feat.columns and "Quantity" in df_feat.columns:
        if pd.api.types.is_numeric_dtype(df_feat["Price"]) and pd.api.types.is_numeric_dtype(df_feat["Quantity"]):
            df_feat["Total Amount"] = (df_feat["Price"] * df_feat["Quantity"]).round(2)
            added_features.append("Total Amount (Price × Quantity)")

    # 2. Date features from Order Date
    date_col = next((c for c in df_feat.columns if "date" in c.lower()), None)
    if date_col:
        try:
            parsed_date = pd.to_datetime(df_feat[date_col], errors="coerce")
            if parsed_date.notna().sum() > 0:
                df_feat["Order Year"] = parsed_date.dt.year.astype("Int64")
                df_feat["Order Month"] = parsed_date.dt.month.astype("Int64")
                df_feat["Order Day"] = parsed_date.dt.day.astype("Int64")
                df_feat["Order DayOfWeek"] = parsed_date.dt.day_name()
                df_feat["Is_Weekend"] = parsed_date.dt.dayofweek.isin([5, 6]).astype(int)
                added_features.extend(["Order Year", "Order Month", "Order Day", "Order DayOfWeek", "Is_Weekend"])
        except Exception:
            pass

    # 3. Customer behavioral aggregations if Customer ID exists
    cust_col = next((c for c in df_feat.columns if "customer" in c.lower() and "id" in c.lower()), None)
    spend_col = "Total Amount" if "Total Amount" in df_feat.columns else ("Price" if "Price" in df_feat.columns else None)
    
    if cust_col and spend_col:
        cust_stats = df_feat.groupby(cust_col).agg(
            Customer_Purchase_Frequency=(spend_col, "count"),
            Total_Customer_Spending=(spend_col, "sum"),
            Average_Purchase_Value=(spend_col, "mean")
        ).round(2).reset_index()

        df_feat = pd.merge(df_feat, cust_stats, on=cust_col, how="left")
        added_features.extend(["Customer_Purchase_Frequency", "Total_Customer_Spending", "Average_Purchase_Value"])

    return df_feat, added_features


def discretize_feature(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",  # "equal_width", "equal_frequency", "custom"
    n_bins: int = 3,
    labels: Optional[List[str]] = None,
    custom_bins: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Converts a continuous numerical column into discrete categorical bins.
    """
    df_result = df.copy()
    if column not in df_result.columns or not pd.api.types.is_numeric_dtype(df_result[column]):
        return df_result, {"error": f"Column '{column}' is not numeric."}

    series = df_result[column].dropna()
    new_col = f"{column}_binned"

    try:
        if method == "equal_width":
            df_result[new_col] = pd.cut(df_result[column], bins=n_bins, labels=labels)
        elif method == "equal_frequency":
            df_result[new_col] = pd.qcut(df_result[column], q=n_bins, labels=labels, duplicates="drop")
        elif method == "custom" and custom_bins:
            df_result[new_col] = pd.cut(df_result[column], bins=custom_bins, labels=labels)
        else:
            return df_result, {"error": "Invalid discretization parameters."}

        counts = df_result[new_col].value_counts(dropna=False).to_dict()
        summary = {
            "column": column,
            "binned_column": new_col,
            "method": method,
            "distribution": {str(k): int(v) for k, v in counts.items()}
        }
        return df_result, summary
    except Exception as e:
        return df_result, {"error": f"Binning failed: {str(e)}"}
