"""
Module 3: Data Validation & Quality Scoring
Performs 8 rigorous data quality checks and computes an explainable Data Quality Score (0-100%).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes 8 comprehensive data validation checks:
    1. Missing Value Validation
    2. Duplicate Record Validation
    3. Data Type Validation
    4. Numerical Range Validation
    5. Categorical Value Validation
    6. Date Format Validation
    7. Unique Identifier Validation
    8. Infinite & Invalid Value Detection
    """
    if df is None or df.empty:
        return {
            "score": 0.0,
            "grade": "F",
            "passed_count": 0,
            "failed_count": 8,
            "warning_count": 0,
            "checks": [],
            "breakdown": {}
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    checks = []

    # Check 1: Missing Values
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells > 0 else 0.0
    if missing_cells == 0:
        c1_status = "PASSED"
        c1_msg = "Zero missing values detected across all columns."
        c1_score = 100.0
    elif missing_pct < 5.0:
        c1_status = "WARNING"
        c1_msg = f"{missing_cells} missing cell(s) ({missing_pct}%) detected. Minor imputation recommended."
        c1_score = max(0.0, 100.0 - missing_pct * 4)
    else:
        c1_status = "FAILED"
        c1_msg = f"{missing_cells} missing cell(s) ({missing_pct}%) detected. Requires imputation or row removal."
        c1_score = max(0.0, 100.0 - missing_pct * 2)

    checks.append({
        "check_id": "CHK-01",
        "name": "Missing Value Validation",
        "status": c1_status,
        "score": round(c1_score, 1),
        "details": c1_msg
    })

    # Check 2: Duplicate Records
    duplicate_rows = int(df.duplicated().sum())
    dup_pct = round((duplicate_rows / total_rows * 100.0), 2) if total_rows > 0 else 0.0
    if duplicate_rows == 0:
        c2_status = "PASSED"
        c2_msg = "No duplicate rows found. Record uniqueness is preserved."
        c2_score = 100.0
    elif dup_pct < 3.0:
        c2_status = "WARNING"
        c2_msg = f"{duplicate_rows} duplicate row(s) ({dup_pct}%) found. Consider deduplication."
        c2_score = max(0.0, 100.0 - dup_pct * 10)
    else:
        c2_status = "FAILED"
        c2_msg = f"{duplicate_rows} duplicate row(s) ({dup_pct}%) found. High redundancy."
        c2_score = max(0.0, 100.0 - dup_pct * 5)

    checks.append({
        "check_id": "CHK-02",
        "name": "Duplicate Record Validation",
        "status": c2_status,
        "score": round(c2_score, 1),
        "details": c2_msg
    })

    # Check 3: Data Type Consistency
    # Check if object columns contain mixed types or should be numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    mixed_cols = []
    for col in non_numeric_cols:
        types = set(type(x) for x in df[col].dropna())
        if len(types) > 1:
            mixed_cols.append(col)

    if not mixed_cols:
        c3_status = "PASSED"
        c3_msg = f"All {total_cols} columns have consistent single data types."
        c3_score = 100.0
    else:
        c3_status = "WARNING"
        c3_msg = f"Mixed python types detected in columns: {', '.join(mixed_cols[:3])}."
        c3_score = max(50.0, 100.0 - len(mixed_cols) * 15)

    checks.append({
        "check_id": "CHK-03",
        "name": "Data Type Validation",
        "status": c3_status,
        "score": round(c3_score, 1),
        "details": c3_msg
    })

    # Check 4: Numerical Range Validation
    range_violations = []
    for col in numeric_cols:
        col_lower = col.lower()
        if "age" in col_lower:
            invalid_age = df[(df[col] < 0) | (df[col] > 120)][col].count()
            if invalid_age > 0:
                range_violations.append(f"{col}: {invalid_age} values outside [0, 120]")
        elif "price" in col_lower or "amount" in col_lower:
            invalid_price = df[df[col] < 0][col].count()
            if invalid_price > 0:
                range_violations.append(f"{col}: {invalid_price} negative prices")
        elif "quantity" in col_lower:
            invalid_qty = df[df[col] <= 0][col].count()
            if invalid_qty > 0:
                range_violations.append(f"{col}: {invalid_qty} non-positive quantities")
        elif "rating" in col_lower:
            invalid_rating = df[(df[col] < 1.0) | (df[col] > 5.0)][col].count()
            if invalid_rating > 0:
                range_violations.append(f"{col}: {invalid_rating} ratings outside [1.0, 5.0]")

    if not range_violations:
        c4_status = "PASSED"
        c4_msg = "All numerical fields comply with domain range constraints."
        c4_score = 100.0
    else:
        c4_status = "FAILED"
        c4_msg = f"Numerical range violations detected: {'; '.join(range_violations)}."
        c4_score = max(40.0, 100.0 - len(range_violations) * 20)

    checks.append({
        "check_id": "CHK-04",
        "name": "Numerical Range Validation",
        "status": c4_status,
        "score": round(c4_score, 1),
        "details": c4_msg
    })

    # Check 5: Categorical Value Validation
    empty_strings_detected = 0
    for col in non_numeric_cols:
        empty_mask = df[col].astype(str).str.strip().isin(["", "nan", "none", "null", "undefined"])
        empty_strings_detected += int(empty_mask.sum())

    if empty_strings_detected == 0:
        c5_status = "PASSED"
        c5_msg = "No empty strings or unhandled null pseudovalues in categorical fields."
        c5_score = 100.0
    elif empty_strings_detected < 20:
        c5_status = "WARNING"
        c5_msg = f"{empty_strings_detected} empty string/unhandled null categorical representations found."
        c5_score = 80.0
    else:
        c5_status = "FAILED"
        c5_msg = f"{empty_strings_detected} unhandled pseudonulls/blanks in categorical columns."
        c5_score = 60.0

    checks.append({
        "check_id": "CHK-05",
        "name": "Categorical Value Validation",
        "status": c5_status,
        "score": round(c5_score, 1),
        "details": c5_msg
    })

    # Check 6: Date Format Validation
    date_cols = [c for c in df.columns if "date" in c.lower()]
    date_issues = []
    for col in date_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            unparseable = int(parsed.isna().sum()) - int(df[col].isna().sum())
            if unparseable > 0:
                date_issues.append(f"{col}: {unparseable} unparseable date strings")
        except Exception:
            date_issues.append(f"{col}: Could not parse column as dates")

    if not date_cols or not date_issues:
        c6_status = "PASSED"
        c6_msg = "All date columns conform to standard timestamp representations." if date_cols else "No date columns present (N/A)."
        c6_score = 100.0
    else:
        c6_status = "WARNING"
        c6_msg = f"Date parsing warnings: {'; '.join(date_issues)}."
        c6_score = 70.0

    checks.append({
        "check_id": "CHK-06",
        "name": "Date Format Validation",
        "status": c6_status,
        "score": round(c6_score, 1),
        "details": c6_msg
    })

    # Check 7: Unique Identifier Validation
    id_cols = [c for c in df.columns if "id" in c.lower() and not "product" in c.lower()]
    id_violations = []
    for col in id_cols:
        if "order" in col.lower():
            dup_ids = df[col].dropna().duplicated().sum()
            if dup_ids > 0:
                id_violations.append(f"{col}: {dup_ids} duplicate IDs")

    if not id_violations:
        c7_status = "PASSED"
        c7_msg = "Primary/Order identifiers demonstrate unique referential integrity."
        c7_score = 100.0
    else:
        c7_status = "FAILED"
        c7_msg = f"Identifier collision detected: {'; '.join(id_violations)}."
        c7_score = 65.0

    checks.append({
        "check_id": "CHK-07",
        "name": "Unique Identifier Validation",
        "status": c7_status,
        "score": round(c7_score, 1),
        "details": c7_msg
    })

    # Check 8: Infinite & Invalid Value Detection
    inf_count = 0
    for col in numeric_cols:
        inf_count += int(np.isinf(df[col].dropna()).sum())

    if inf_count == 0:
        c8_status = "PASSED"
        c8_msg = "Zero infinite (+inf / -inf) floating point values detected."
        c8_score = 100.0
    else:
        c8_status = "FAILED"
        c8_msg = f"{inf_count} infinite numerical value(s) detected. Will crash ML models."
        c8_score = 30.0

    checks.append({
        "check_id": "CHK-08",
        "name": "Infinite & Invalid Value Detection",
        "status": c8_status,
        "score": round(c8_score, 1),
        "details": c8_msg
    })

    # Weighted Data Quality Score Calculation
    # Completeness (30%), Uniqueness (20%), Validity (25%), Consistency (15%), Integrity (10%)
    weights = [0.30, 0.20, 0.10, 0.15, 0.10, 0.05, 0.05, 0.05]
    final_score = sum(check["score"] * w for check, w in zip(checks, weights))
    final_score = round(max(0.0, min(100.0, final_score)), 1)

    if final_score >= 90.0:
        grade = "A (ML Ready / High Quality)"
        badge_color = "#10b981"
    elif final_score >= 75.0:
        grade = "B (Good / Minor Cleaning Needed)"
        badge_color = "#06b6d4"
    elif final_score >= 60.0:
        grade = "C (Moderate / Issues Present)"
        badge_color = "#f59e0b"
    else:
        grade = "D (Poor / Critical Failures)"
        badge_color = "#ef4444"

    passed_count = sum(1 for c in checks if c["status"] == "PASSED")
    warning_count = sum(1 for c in checks if c["status"] == "WARNING")
    failed_count = sum(1 for c in checks if c["status"] == "FAILED")

    breakdown = {
        "completeness_score": round(c1_score, 1),
        "uniqueness_score": round(c2_score, 1),
        "data_type_score": round(c3_score, 1),
        "range_validity_score": round(c4_score, 1),
        "categorical_score": round(c5_score, 1),
        "date_score": round(c6_score, 1),
        "identifier_score": round(c7_score, 1),
        "numerical_integrity_score": round(c8_score, 1)
    }

    return {
        "score": final_score,
        "grade": grade,
        "badge_color": badge_color,
        "passed_count": passed_count,
        "warning_count": warning_count,
        "failed_count": failed_count,
        "checks": checks,
        "breakdown": breakdown,
        "checks_df": pd.DataFrame(checks)
    }
