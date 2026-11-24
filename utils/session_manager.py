"""
Session State Manager
Maintains data lineage across all preprocessing modules without mutating original raw data.
Tracks processing history, active stage, and intermediate representations.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List


STAGES = [
    "Raw Data",
    "Cleaning",
    "Integration",
    "Transformation",
    "Reduction",
    "Validation",
    "ML Ready"
]


def init_session_state():
    """Initializes all required session state keys if not already present."""
    defaults = {
        # Dataset storage
        "raw_df": None,
        "dataset_name": "ecommerce_main.csv",
        "cleaned_df": None,
        "integrated_df": None,
        "transformed_df": None,
        "reduced_df": None,
        "validated_df": None,
        
        # Relational tables for integration
        "customer_df": None,
        "product_df": None,
        "order_df": None,
        "transaction_df": None,
        
        # ML Artifacts
        "ml_data": None,
        "classification_results": None,
        "clustering_results": None,
        "clustering_df": None,
        
        # Pipeline State
        "current_stage": "Raw Data",
        "pipeline_stages_completed": {stage: False for stage in STAGES},
        "processing_history": [],
        
        # Validation
        "validation_report": None,
        
        # Navigation
        "active_page": "Dashboard"
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def log_action(stage: str, description: str):
    """Appends an event to the processing audit history."""
    if "processing_history" not in st.session_state:
        st.session_state.processing_history = []
    
    st.session_state.processing_history.append({
        "stage": stage,
        "description": description
    })


def mark_stage_complete(stage: str):
    """Marks a pipeline stage as completed and updates current_stage."""
    if "pipeline_stages_completed" not in st.session_state:
        st.session_state.pipeline_stages_completed = {s: False for s in STAGES}
    st.session_state.pipeline_stages_completed[stage] = True
    st.session_state.current_stage = stage


def get_current_working_df() -> Optional[pd.DataFrame]:
    """
    Returns the most refined dataset available according to pipeline flow:
    Reduced -> Transformed -> Integrated -> Cleaned -> Raw.
    """
    if st.session_state.get("reduced_df") is not None:
        return st.session_state.reduced_df
    if st.session_state.get("transformed_df") is not None:
        return st.session_state.transformed_df
    if st.session_state.get("integrated_df") is not None:
        return st.session_state.integrated_df
    if st.session_state.get("cleaned_df") is not None:
        return st.session_state.cleaned_df
    if st.session_state.get("raw_df") is not None:
        return st.session_state.raw_df
    return None


def reset_pipeline():
    """Resets all processed datasets back to the initial raw dataset."""
    st.session_state.cleaned_df = None
    st.session_state.integrated_df = None
    st.session_state.transformed_df = None
    st.session_state.reduced_df = None
    st.session_state.validated_df = None
    st.session_state.ml_data = None
    st.session_state.classification_results = None
    st.session_state.clustering_results = None
    st.session_state.clustering_df = None
    st.session_state.validation_report = None
    st.session_state.current_stage = "Raw Data"
    st.session_state.pipeline_stages_completed = {stage: (stage == "Raw Data") for stage in STAGES}
    st.session_state.processing_history = [
        {"stage": "System", "description": "Pipeline reset to initial raw state."}
    ]
