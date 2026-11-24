"""
Module 1: Data Integration
Enables combining multiple datasets through relational joins (Inner, Left, Right, Outer)
or horizontal/vertical concatenations, calculating alignment metrics and previewing results.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


def join_datasets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    how: str = "inner",
    suffixes: Tuple[str, str] = ("_left", "_right")
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Performs relational merge of two datasets with audit statistics.
    how: 'inner', 'left', 'right', 'outer'
    """
    if left_on not in left_df.columns:
        return None, {"error": f"Join key '{left_on}' not found in left dataset."}
    if right_on not in right_df.columns:
        return None, {"error": f"Join key '{right_on}' not found in right dataset."}

    # Cast keys to string to avoid mismatch between int and str representations of IDs
    left_copy = left_df.copy()
    right_copy = right_df.copy()
    left_copy[left_on] = left_copy[left_on].astype(str).str.strip()
    right_copy[right_on] = right_copy[right_on].astype(str).str.strip()

    merged_df = pd.merge(
        left_copy,
        right_copy,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=suffixes
    )

    # Compute key intersection stats
    left_keys = set(left_copy[left_on].dropna())
    right_keys = set(right_copy[right_on].dropna())
    intersection_keys = left_keys.intersection(right_keys)

    summary = {
        "operation": f"{how.upper()} JOIN",
        "left_rows": len(left_df),
        "right_rows": len(right_df),
        "merged_rows": len(merged_df),
        "left_cols": len(left_df.columns),
        "right_cols": len(right_df.columns),
        "merged_cols": len(merged_df.columns),
        "left_unique_keys": len(left_keys),
        "right_unique_keys": len(right_keys),
        "matched_unique_keys": len(intersection_keys),
        "match_rate_pct": round(len(intersection_keys) / max(len(left_keys), 1) * 100.0, 2)
    }

    return merged_df, summary


def concatenate_datasets(
    dataframes: List[pd.DataFrame],
    axis: int = 0,
    ignore_index: bool = True
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Concatenates multiple dataframes vertically (axis=0) or horizontally (axis=1).
    """
    if not dataframes:
        return None, {"error": "No dataframes provided for concatenation."}

    initial_row_counts = [len(df) for df in dataframes]
    initial_col_counts = [len(df.columns) for df in dataframes]

    concat_df = pd.concat(dataframes, axis=axis, ignore_index=ignore_index)

    summary = {
        "operation": "Vertical Concatenation" if axis == 0 else "Horizontal Concatenation",
        "input_dataset_count": len(dataframes),
        "input_rows": initial_row_counts,
        "input_cols": initial_col_counts,
        "resulting_rows": len(concat_df),
        "resulting_cols": len(concat_df.columns)
    }

    return concat_df, summary
