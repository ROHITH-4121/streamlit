"""
Module: Classification Analysis
Trains and evaluates 3 core machine learning classifiers:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
Guarantees data leakage prevention: all scaling & imputation fitted strictly on X_train.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional


def prepare_ml_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Splits features and target into train/test sets, encodes categorical columns,
    and scales numerical features strictly using training data parameters to prevent leakage.
    """
    if target_col not in df.columns:
        return None, f"Target column '{target_col}' not found in dataset."

    valid_features = [c for c in feature_cols if c in df.columns and c != target_col]
    if len(valid_features) < 1:
        return None, "Please select at least one input feature."

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 20:
        return None, "Dataset has too few records (< 20) with valid target labels."

    # Check target classes
    y_raw = clean_df[target_col].astype(str)
    unique_classes = y_raw.unique()
    if len(unique_classes) < 2:
        return None, f"Target '{target_col}' must contain at least 2 distinct classes. Found: {list(unique_classes)}"
    if len(unique_classes) > 15:
        return None, f"Target '{target_col}' has {len(unique_classes)} classes. Classification requires <= 15 discrete classes."

    # Encode Target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # Partition numeric and categorical feature columns
    num_cols = [c for c in valid_features if pd.api.types.is_numeric_dtype(clean_df[c])]
    cat_cols = [c for c in valid_features if c not in num_cols]

    X_raw = clean_df[valid_features].copy()

    # Encode categorical features
    for col in cat_cols:
        cat_le = LabelEncoder()
        X_raw[col] = cat_le.fit_transform(X_raw[col].astype(str))

    # Check stratification possibility
    min_class_count = pd.Series(y).value_counts().min()
    stratify = y if min_class_count >= 2 else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Impute missing values (fit on train only!)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scale features (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Form clean DataFrames
    X_train_df = pd.DataFrame(X_train_scaled, columns=valid_features)
    X_test_df = pd.DataFrame(X_test_scaled, columns=valid_features)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "class_names": class_names,
        "feature_names": valid_features,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "test_ratio": test_size,
        "target_col": target_col
    }, None


def train_and_evaluate_classifiers(ml_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains Logistic Regression, Decision Tree, and Random Forest on X_train
    and evaluates on X_test with real metrics.
    """
    X_train = ml_data["X_train"]
    X_test = ml_data["X_test"]
    y_train = ml_data["y_train"]
    y_test = ml_data["y_test"]
    class_names = ml_data["class_names"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    results = {}
    comparison_records = []

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics with explicit labels to prevent missing-class dimension mismatch
        all_labels = list(range(len(class_names)))
        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        rec = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        cr = classification_report(y_test, y_pred, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": cr,
            "y_pred": y_pred
        }

        comparison_records.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision (Weighted)": prec,
            "Recall (Weighted)": rec,
            "F1-Score (Weighted)": f1
        })

    comparison_df = pd.DataFrame(comparison_records).sort_values(by="F1-Score (Weighted)", ascending=False).reset_index(drop=True)

    return {
        "models": results,
        "comparison_table": comparison_df,
        "class_names": class_names,
        "best_model": comparison_df.iloc[0]["Algorithm"]
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_name: str) -> go.Figure:
    """Plots a dark SaaS-themed heatmap for the confusion matrix."""
    z = cm.tolist()
    x = class_names
    y = class_names

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(
        title={"text": f"<b>Confusion Matrix Heatmap ({model_name})</b>", "font": {"color": "#f8fafc", "size": 15}},
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    df_melt = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        df_melt,
        x="Algorithm",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#06b6d4", "#38bdf8", "#6366f1", "#10b981"],
        text_auto=".3f"
    )
    fig.update_layout(
        title={"text": "<b>Classifier Performance Benchmark</b>", "font": {"color": "#f8fafc", "size": 15}},
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        yaxis=dict(range=[0, 1.05], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
    )
    return fig
