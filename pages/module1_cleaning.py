"""
Page: Module 1 – Data Collection, Cleaning & Integration
Implements end-to-end data ingestion, missing value imputation, duplicate purging,
outlier detection (IQR / Z-score) & treatment, type conversions, categorical standardization,
and relational dataset joins.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.helpers import render_page_header, render_info_box, render_workflow_progress
from utils.session_manager import mark_stage_complete, log_action
from utils.file_handler import dataframe_to_csv_bytes, dataframe_to_excel_bytes

from modules.data_collection import load_file, load_sample_dataset, get_dataset_summary
from modules.data_cleaning import (
    get_missing_value_report, handle_missing_values, detect_and_remove_duplicates,
    detect_outliers_iqr, detect_outliers_zscore, treat_outliers,
    convert_data_types, standardize_categorical_data
)
from modules.data_integration import join_datasets, concatenate_datasets


def render():
    render_page_header(
        title="Module 1: Data Collection, Cleaning & Integration",
        subtitle="Ingest heterogeneous sources, resolve anomalies, purge duplicates, and join relational tables.",
        icon="🧹",
        stage_badge=st.session_state.current_stage
    )

    render_workflow_progress(st.session_state.current_stage, st.session_state.pipeline_stages_completed)

    # 3 Main Tabs: Ingestion & Collection, Cleaning & Remediation, Integration
    tab_ingest, tab_clean, tab_integrate = st.tabs([
        "📥 1. Ingestion & Collection",
        "🧼 2. Data Cleaning & Remediation",
        "🔗 3. Relational Data Integration"
    ])

    # =========================================================================
    # TAB 1: DATA COLLECTION
    # =========================================================================
    with tab_ingest:
        st.subheader("1.1 Data Ingestion")
        st.markdown("Upload your e-commerce dataset (CSV, Excel, JSON) or load our pre-configured realistic benchmark dataset.")

        col_up, col_sample = st.columns([2, 1])

        with col_up:
            uploaded_file = st.file_uploader(
                "Choose a file to upload",
                type=["csv", "xlsx", "xls", "json"],
                help="Supports CSV, Excel spreadsheets, and structured JSON"
            )
            if uploaded_file is not None:
                df_loaded, err = load_file(uploaded_file)
                if err:
                    st.error(err)
                elif df_loaded is not None:
                    st.session_state.raw_df = df_loaded.copy()
                    st.session_state.dataset_name = uploaded_file.name
                    mark_stage_complete("Raw Data")
                    log_action("Collection", f"Uploaded dataset '{uploaded_file.name}' with {len(df_loaded)} rows.")
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df_loaded):,} rows, {len(df_loaded.columns)} columns)!")

        with col_sample:
            st.markdown("#### Sample Dataset")
            st.caption("Realistic e-commerce dataset containing customer, product, order, price, rating, and address attributes with realistic noise.")
            if st.button("📥 Load Built-in E-Commerce Dataset", use_container_width=True):
                sample_df, err = load_sample_dataset("ecommerce_main.csv")
                if err:
                    st.error(err)
                elif sample_df is not None:
                    st.session_state.raw_df = sample_df.copy()
                    st.session_state.dataset_name = "ecommerce_main.csv"
                    mark_stage_complete("Raw Data")
                    log_action("Collection", "Loaded built-in sample e-commerce dataset.")
                    st.success(f"Loaded 'ecommerce_main.csv' ({len(sample_df):,} rows, {len(sample_df.columns)} columns)!")
                    st.rerun()

        # Display Ingested Summary
        if st.session_state.get("raw_df") is not None:
            raw_df = st.session_state.raw_df
            st.divider()
            st.subheader("1.2 Ingested Dataset Overview")
            
            summary = get_dataset_summary(raw_df)
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Rows", f"{summary['total_rows']:,}")
            col_m2.metric("Columns", f"{summary['total_cols']}")
            col_m3.metric("Missing Cells", f"{summary['missing_cells']:,}", f"{summary['missing_percentage']}%")
            col_m4.metric("Duplicates", f"{summary['duplicate_rows']:,}", f"{summary['duplicate_percentage']}%")

            st.dataframe(raw_df.head(10), use_container_width=True)

    # =========================================================================
    # TAB 2: DATA CLEANING
    # =========================================================================
    with tab_clean:
        # Determine working base: cleaned_df if already created, else raw_df
        base_df = st.session_state.get("cleaned_df", st.session_state.get("raw_df"))

        if base_df is None:
            st.warning("⚠️ Please ingest or load a dataset in Tab 1 first.")
            return

        st.subheader("2.1 Missing Value Detection & Imputation")
        missing_rep = get_missing_value_report(base_df)
        cols_with_missing = missing_rep[missing_rep["Missing Count"] > 0]

        if cols_with_missing.empty:
            st.success("🎉 No missing values detected in the current dataset!")
        else:
            st.dataframe(cols_with_missing, use_container_width=True)

            st.markdown("#### Configure Imputation Strategies")
            col_strat1, col_strat2 = st.columns(2)
            
            strat_dict = {}
            for i, (_, row) in enumerate(cols_with_missing.iterrows()):
                col_name = row["Column"]
                dtype = row["Data Type"]
                target_col = col_strat1 if (i % 2 == 0) else col_strat2
                
                with target_col:
                    is_num = "int" in dtype.lower() or "float" in dtype.lower()
                    opts = ["Keep Original", "Drop Rows", "Mean", "Median", "Mode", "Custom Constant"] if is_num else ["Keep Original", "Drop Rows", "Mode", "Custom Constant"]
                    def_idx = 2 if is_num else 2 # Default Mean or Mode
                    
                    choice = st.selectbox(
                        f"Impute '{col_name}' ({row['Missing Count']} nulls):",
                        options=opts,
                        index=def_idx,
                        key=f"imp_{col_name}"
                    )
                    
                    const_val = None
                    if choice == "Custom Constant":
                        const_val = st.text_input(f"Constant value for '{col_name}':", key=f"const_{col_name}")

                    action_key = choice.lower().replace(" ", "_")
                    if action_key == "keep_original":
                        continue
                    elif action_key == "custom_constant":
                        strat_dict[col_name] = {"action": "constant", "value": const_val}
                    else:
                        strat_dict[col_name] = {"action": action_key}

            if st.button("⚡ Execute Missing Value Imputation", type="primary", use_container_width=True):
                cleaned, res = handle_missing_values(base_df, strat_dict)
                st.session_state.cleaned_df = cleaned
                mark_stage_complete("Cleaning")
                log_action("Cleaning", f"Imputed missing values: {len(res['actions'])} actions applied.")
                st.success(f"Missing values handled! Rows before: {res['rows_before']} → Rows after: {res['rows_after']}. Missing cells remaining: {res['missing_after']}.")
                st.rerun()

        st.divider()

        # 2.2 Duplicate Removal
        st.subheader("2.2 Duplicate Record Purging")
        current_work_df = st.session_state.get("cleaned_df", base_df)
        _, dup_stats = detect_and_remove_duplicates(current_work_df, remove=False)

        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.metric("Identified Duplicates", f"{dup_stats['duplicate_rows']:,}")
            if dup_stats['duplicate_rows'] > 0:
                if st.button("🗑️ Purge Duplicate Rows", use_container_width=True):
                    deduped, res = detect_and_remove_duplicates(current_work_df, remove=True)
                    st.session_state.cleaned_df = deduped
                    mark_stage_complete("Cleaning")
                    log_action("Cleaning", f"Purged {res['removed']} duplicate rows.")
                    st.success(f"Purged {res['removed']} duplicate rows!")
                    st.rerun()
            else:
                st.success("Zero duplicate rows found.")

        with col_d2:
            if dup_stats['duplicate_rows'] > 0:
                st.caption("Sample of duplicate records detected:")
                st.dataframe(dup_stats["duplicated_samples"].head(5), use_container_width=True)

        st.divider()

        # 2.3 Outlier Detection & Treatment
        st.subheader("2.3 Outlier Detection & Treatment (IQR & Z-Score)")
        curr_df = st.session_state.get("cleaned_df", base_df)
        num_cols = curr_df.select_dtypes(include=[np.number]).columns.tolist()

        if num_cols:
            col_out1, col_out2, col_out3 = st.columns(3)
            with col_out1:
                target_num = st.selectbox("Select numerical column:", options=num_cols)
            with col_out2:
                out_method = st.radio("Outlier Method:", ["IQR (Interquartile Range)", "Z-Score"], horizontal=True)
            with col_out3:
                out_action = st.selectbox("Action to take:", ["Cap Extreme Values (Winsorize)", "Remove Outlier Rows", "Keep Original"])

            method_key = "IQR" if "IQR" in out_method else "Z-Score"
            
            # Run detection
            if method_key == "IQR":
                out_info = detect_outliers_iqr(curr_df, target_num, factor=1.5)
            else:
                out_info = detect_outliers_zscore(curr_df, target_num, threshold=3.0)

            # Display stats & box plot
            col_stat, col_plot = st.columns([1, 2])
            with col_stat:
                st.markdown(f"**Method:** {method_key}")
                st.markdown(f"**Outliers Found:** `{out_info.get('outlier_count', 0)}` ({out_info.get('outlier_pct', 0)}%)")
                st.markdown(f"**Lower Bound:** `{out_info.get('lower_bound', 'N/A')}`")
                st.markdown(f"**Upper Bound:** `{out_info.get('upper_bound', 'N/A')}`")

                action_key = "cap" if "Cap" in out_action else ("remove" if "Remove" in out_action else "keep")
                if st.button(f"Apply Outlier Action on '{target_num}'", use_container_width=True):
                    treated_df, treat_res = treat_outliers(curr_df, target_num, action=action_key, method=method_key)
                    st.session_state.cleaned_df = treated_df
                    mark_stage_complete("Cleaning")
                    log_action("Cleaning", treat_res.get("message", "Treated outliers."))
                    st.success(treat_res.get("message", "Done."))
                    st.rerun()

            with col_plot:
                fig_box = px.box(
                    curr_df,
                    y=target_num,
                    points="outliers",
                    title=f"Box Plot: {target_num} Distribution",
                    color_discrete_sequence=["#06b6d4"]
                )
                fig_box.update_layout(
                    paper_bgcolor="#1e293b",
                    plot_bgcolor="#0f172a",
                    font=dict(color="#cbd5e1"),
                    height=280,
                    margin=dict(l=30, r=30, t=40, b=30)
                )
                st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        # 2.4 Data Type Conversion & Inconsistent Categorical Standardizing
        st.subheader("2.4 Type Conversion & Inconsistent Data Handling")
        col_type, col_incon = st.columns(2)

        with col_type:
            st.markdown("#### Cast Column Data Type")
            target_type_col = st.selectbox("Select column to cast:", options=list(curr_df.columns), key="type_cast_col")
            new_type = st.selectbox("Target Data Type:", ["Integer", "Float", "String", "DateTime", "Category"])
            if st.button("Apply Type Conversion", use_container_width=True):
                typed_df, msgs = convert_data_types(curr_df, {target_type_col: new_type})
                st.session_state.cleaned_df = typed_df
                mark_stage_complete("Cleaning")
                log_action("Cleaning", f"Converted '{target_type_col}' to {new_type}.")
                st.success(f"Converted '{target_type_col}' to {new_type}!")
                st.rerun()

        with col_incon:
            st.markdown("#### Standardize Inconsistent Categories")
            str_cols = curr_df.select_dtypes(include=["object", "category"]).columns.tolist()
            if str_cols:
                target_cat_col = st.selectbox("Select categorical column:", options=str_cols, key="cat_std_col")
                std_mode = st.selectbox(
                    "Standardization Rule:",
                    ["Gender Normalization (M/m/male -> Male)", "Trim Whitespace & Title Case", "Uppercase", "Lowercase"]
                )
                
                mode_key = "gender" if "Gender" in std_mode else ("trim_title" if "Title" in std_mode else ("uppercase" if "Upper" in std_mode else "lowercase"))
                if st.button(f"Standardize '{target_cat_col}'", use_container_width=True):
                    std_df, std_summary = standardize_categorical_data(curr_df, target_cat_col, standardization_type=mode_key)
                    st.session_state.cleaned_df = std_df
                    mark_stage_complete("Cleaning")
                    log_action("Cleaning", f"Standardized categorical column '{target_cat_col}'.")
                    st.success(f"Standardized '{target_cat_col}'!")
                    st.rerun()

    # =========================================================================
    # TAB 3: DATA INTEGRATION
    # =========================================================================
    with tab_integrate:
        st.subheader("3. Relational Multi-Dataset Integration")
        render_info_box(
            title="Relational Schema Integration",
            content="Combine multi-source e-commerce tables (Customers, Products, Orders, Transactions) using relational joins or vertical/horizontal concatenations.",
            icon="🔗",
            box_type="info"
        )

        col_int_opt1, col_int_opt2 = st.columns(2)
        with col_int_opt1:
            st.markdown("#### Load Relational Capstone Tables")
            if st.button("📂 Load Multi-Table Schema (Customers, Products, Orders, Txns)", use_container_width=True):
                from modules.data_collection import load_sample_dataset
                cust_df, _ = load_sample_dataset("customers.csv")
                prod_df, _ = load_sample_dataset("products.csv")
                ord_df, _ = load_sample_dataset("orders.csv")
                txn_df, _ = load_sample_dataset("transactions.csv")
                
                st.session_state.customer_df = cust_df
                st.session_state.product_df = prod_df
                st.session_state.order_df = ord_df
                st.session_state.transaction_df = txn_df
                st.success("Loaded 4 relational tables: Customers, Products, Orders, Transactions!")

        with col_int_opt2:
            st.markdown("#### Active Relational Tables Status")
            t_status = []
            for t_name, k in [("Customers", "customer_df"), ("Products", "product_df"), ("Orders", "order_df"), ("Transactions", "transaction_df")]:
                df_k = st.session_state.get(k)
                cnt = len(df_k) if df_k is not None else 0
                t_status.append(f"• **{t_name}:** {cnt} rows")
            st.markdown("\n".join(t_status))

        st.divider()

        # Join Configuration
        st.subheader("Execute Relational Join")
        table_keys = {
            "Current Working Dataset": st.session_state.get("cleaned_df", st.session_state.get("raw_df")),
            "Customers (Relational)": st.session_state.get("customer_df"),
            "Products (Relational)": st.session_state.get("product_df"),
            "Orders (Relational)": st.session_state.get("order_df"),
            "Transactions (Relational)": st.session_state.get("transaction_df")
        }
        # Filter available tables
        avail_tables = {k: v for k, v in table_keys.items() if v is not None}

        if len(avail_tables) >= 2:
            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                left_choice = st.selectbox("Left Table:", options=list(avail_tables.keys()), index=0)
            with col_j2:
                right_choice = st.selectbox("Right Table:", options=list(avail_tables.keys()), index=min(1, len(avail_tables)-1))
            with col_j3:
                join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"])

            left_table = avail_tables[left_choice]
            right_table = avail_tables[right_choice]

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                left_key = st.selectbox(f"Join Key in '{left_choice}':", options=list(left_table.columns))
            with col_k2:
                right_key = st.selectbox(f"Join Key in '{right_choice}':", options=list(right_table.columns))

            if st.button("🚀 Merge Datasets via Relational Join", type="primary", use_container_width=True):
                merged, join_summary = join_datasets(left_table, right_table, left_key, right_key, how=join_type)
                if merged is not None:
                    st.session_state.integrated_df = merged
                    mark_stage_complete("Integration")
                    log_action("Integration", f"Merged {left_choice} and {right_choice} via {join_type.upper()} join on {left_key}={right_key}.")
                    st.success(f"Integration complete! Matched {join_summary['matched_unique_keys']} unique keys. Resulting dataset has {join_summary['merged_rows']} rows and {join_summary['merged_cols']} columns.")
                    st.dataframe(merged.head(10), use_container_width=True)
                else:
                    st.error(join_summary.get("error", "Join failed."))
        else:
            st.info("Load the relational capstone tables above or ingest additional datasets to perform integration joins.")

    # =========================================================================
    # MODULE 1 EXPORTS
    # =========================================================================
    st.divider()
    st.subheader("📥 Export Module 1 Deliverables")
    curr_df_out = st.session_state.get("integrated_df", st.session_state.get("cleaned_df", st.session_state.get("raw_df")))
    if curr_df_out is not None:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📄 Download Cleaned Dataset (CSV)",
                data=dataframe_to_csv_bytes(curr_df_out),
                file_name="cleaned_ecommerce_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            st.download_button(
                label="📊 Download Cleaned Dataset (Excel)",
                data=dataframe_to_excel_bytes(curr_df_out),
                file_name="cleaned_ecommerce_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
