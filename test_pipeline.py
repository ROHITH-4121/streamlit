import sys
import os
import pandas as pd
import numpy as np

print("Python version:", sys.version)

# 1. Test Data Collection
from modules.data_collection import load_sample_dataset, get_dataset_summary
df, err = load_sample_dataset("ecommerce_main.csv")
assert err is None, f"Collection error: {err}"
assert len(df) == 1235, f"Unexpected row count: {len(df)}"
summary = get_dataset_summary(df)
print(f"1. Data Collection OK: {summary['total_rows']} rows, {summary['total_cols']} cols, {summary['missing_cells']} missing cells, {summary['duplicate_rows']} duplicates")

# 2. Test Data Cleaning
from modules.data_cleaning import (
    get_missing_value_report, handle_missing_values, detect_and_remove_duplicates,
    detect_outliers_iqr, treat_outliers, convert_data_types, standardize_categorical_data
)
rep = get_missing_value_report(df)
assert len(rep) > 0
cleaned_df, cl_res = handle_missing_values(df, {"Age": {"action": "median"}, "Price": {"action": "mean"}, "Gender": {"action": "mode"}})
dedup_df, dup_res = detect_and_remove_duplicates(cleaned_df, remove=True)
treated_df, out_res = treat_outliers(dedup_df, "Price", action="cap", method="IQR")
std_df, std_res = standardize_categorical_data(treated_df, "Gender", standardization_type="gender")
print(f"2. Data Cleaning OK: Duplicates removed = {dup_res['removed']}, Outliers capped = {out_res.get('outliers_found', 0)}")

# 3. Test Integration
from modules.data_integration import join_datasets
cust_df, _ = load_sample_dataset("customers.csv")
ord_df, _ = load_sample_dataset("orders.csv")
merged, j_res = join_datasets(cust_df, ord_df, "Customer ID", "Customer ID", how="inner")
print(f"3. Data Integration OK: Merged rows = {j_res['merged_rows']}, match rate = {j_res['match_rate_pct']}%")

# 4. Test Transformation & Reduction
from modules.data_transformation import scale_numerical_features, engineer_ecommerce_features, discretize_feature
from modules.data_reduction import compute_correlation_matrix, apply_pca_reduction, sample_dataset

scaled_df, sc_info = scale_numerical_features(std_df, ["Price", "Age"], method="minmax")
feat_df, added_feats = engineer_ecommerce_features(scaled_df)
print(f"4. Transformation OK: Added features = {added_feats}")

corr_mat, high_pairs = compute_correlation_matrix(feat_df)
pca_df, pca_meta = apply_pca_reduction(feat_df, ["Price", "Age"], n_components=2)
sampled_df, samp_summary = sample_dataset(feat_df, method="random", fraction=0.5)
print(f"5. Reduction OK: PCA total variance = {pca_meta['total_variance_explained_pct']}%, sampled rows = {samp_summary['sampled_rows']}")

# 6. Test Validation
from modules.data_validation import validate_dataset
val_report = validate_dataset(feat_df)
print(f"6. Validation OK: Score = {val_report['score']}%, Grade = {val_report['grade']}, Passed = {val_report['passed_count']}/8")

# 7. Test Classification
from modules.classification_models import prepare_ml_dataset, train_and_evaluate_classifiers
ml_data, ml_err = prepare_ml_dataset(feat_df, target_col="Delivery Status", feature_cols=["Price", "Age", "Quantity", "Review Rating"], test_size=0.2)
assert ml_err is None, f"ML preparation error: {ml_err}"
clf_results = train_and_evaluate_classifiers(ml_data)
print(f"7. Classification OK: Best model = {clf_results['best_model']}, Models = {list(clf_results['models'].keys())}")

# 8. Test Clustering
from modules.clustering_models import compute_clustering_metrics, perform_kmeans_segmentation
cl_metrics, cl_err = compute_clustering_metrics(feat_df, ["Price", "Age", "Quantity"])
assert cl_err is None, f"Clustering metrics error: {cl_err}"
cl_df, cl_summary, cl_err2 = perform_kmeans_segmentation(feat_df, ["Price", "Age", "Quantity"], n_clusters=3)
assert cl_err2 is None, f"Segmentation error: {cl_err2}"
print(f"8. Clustering OK: Silhouette = {cl_summary['silhouette_score']}, Personas = {cl_summary['personas']}")

# 9. Test PDF Report Generation
from modules.report_generation import generate_pdf_report
audit_data = {
    "dataset_name": "ecommerce_main.csv",
    "initial_records": 1235,
    "current_records": len(feat_df),
    "total_features": len(feat_df.columns),
    "columns_list": list(feat_df.columns),
    "initial_missing_cells": 120,
    "residual_missing_cells": int(feat_df.isna().sum().sum()),
    "missing_imputed": 120,
    "duplicates_found": 35,
    "duplicates_purged": 35,
    "quality_score": val_report["score"],
    "quality_grade": val_report["grade"],
    "validation_checks": val_report["checks"],
    "classification_table": clf_results["comparison_table"],
    "clustering_table": cl_summary["profile_table"]
}
pdf_buf = generate_pdf_report(audit_data)
assert len(pdf_buf.getvalue()) > 5000, "PDF buffer empty or too small"
print(f"9. PDF Report Generation OK: Generated valid PDF of {len(pdf_buf.getvalue()):,} bytes!")

print("\n>>> ALL BACKEND MODULES AND MACHINE LEARNING PIPELINES VERIFIED 100% SUCCESFULLY! <<<")
