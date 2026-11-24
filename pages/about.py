"""
Page: About Project, SDG Alignment & Course Outcomes
Presents the project background, UN Sustainable Development Goal (SDG) mapping,
academic Course Outcome (CO) mapping for Data Mining & Warehousing (DWDM),
and software architectural documentation.
"""

import streamlit as st
from utils.helpers import render_page_header, render_metric_card


def render():
    render_page_header(
        title="About Project: SDG & Course Outcome Alignment",
        subtitle="Academic curriculum mapping, United Nations Sustainable Development Goals, and framework architecture.",
        icon="🎓",
        stage_badge="Academic Overview"
    )

    tab_sdg, tab_co, tab_arch = st.tabs([
        "🌍 UN Sustainable Development Goals (SDGs)",
        "🎯 Course Outcomes (CO2, CO3, CO4)",
        "🏗️ System Architecture & Technology Stack"
    ])

    # =========================================================================
    # TAB 1: SDG ALIGNMENT
    # =========================================================================
    with tab_sdg:
        st.subheader("Alignment with United Nations Sustainable Development Goals")
        st.markdown("The platform directly contributes toward three crucial UN Sustainable Development Goals:")

        col_sdg1, col_sdg2, col_sdg3 = st.columns(3)

        with col_sdg1:
            st.markdown("""
            <div class="sdg-card" style="border-top: 4px solid #a21942;">
                <div class="sdg-card-header">
                    <span class="sdg-card-num" style="background-color: #a21942;">SDG 8</span>
                    <h3 class="sdg-card-title">Decent Work & Economic Growth</h3>
                </div>
                <div class="sdg-card-desc">
                    <b>Target Focus:</b> Empowering enterprises with optimized decision intelligence.<br/><br/>
                    The framework automates e-commerce data hygiene and customer analytics, enabling small and medium enterprises (SMEs) to optimize order logistics, sales efficiency, and labor productivity through data-driven operational decisions.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sdg2:
            st.markdown("""
            <div class="sdg-card" style="border-top: 4px solid #f26a2e;">
                <div class="sdg-card-header">
                    <span class="sdg-card-num" style="background-color: #f26a2e;">SDG 9</span>
                    <h3 class="sdg-card-title">Industry, Innovation & Infrastructure</h3>
                </div>
                <div class="sdg-card-desc">
                    <b>Target Focus:</b> Scalable data infrastructure and technological innovation.<br/><br/>
                    The platform bridges the gap between raw unstructured e-commerce transactions and advanced AI models, providing modular preprocessing infrastructure that democratizes state-of-the-art data mining techniques.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sdg3:
            st.markdown("""
            <div class="sdg-card" style="border-top: 4px solid #bf8b2e;">
                <div class="sdg-card-header">
                    <span class="sdg-card-num" style="background-color: #bf8b2e;">SDG 12</span>
                    <h3 class="sdg-card-title">Responsible Consumption & Production</h3>
                </div>
                <div class="sdg-card-desc">
                    <b>Target Focus:</b> Demand forecasting and inventory waste minimization.<br/><br/>
                    By uncovering precise customer purchasing cycles, category demand distributions, and return velocities, businesses can eliminate overstocking, optimize packaging, and curb supply chain carbon emissions.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: COURSE OUTCOME ALIGNMENT
    # =========================================================================
    with tab_co:
        st.subheader("Academic Course Outcome (CO) Alignment")
        st.markdown("Designed specifically for evaluation in **Data Warehousing & Data Mining (DWDM) / Machine Learning** Capstone courses:")

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span class="metric-badge" style="background: rgba(6, 182, 212, 0.2); color: #06b6d4; font-size: 0.9rem; padding: 4px 12px;">CO2</span>
                <h4 style="margin: 0; color: #f8fafc;">Apply suitable Data Preprocessing techniques for effective data preparation</h4>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;">
                <b>Demonstrated Implementation:</b> Implemented comprehensively across <b>Module 1</b> and <b>Module 2</b>.
            </p>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                <li><b>Missing Value Remediation:</b> Imputation via Mean, Median, Mode, or Constant, plus row/column pruning.</li>
                <li><b>Duplicate Elimination:</b> Identification, statistical auditing, and lossless deduplication.</li>
                <li><b>Outlier Treatment:</b> Interquartile Range (IQR, 1.5×) and Z-score (|z| > 3.0) detection, box plots, and Winsorization.</li>
                <li><b>Data Integration:</b> Relational joins (Inner, Left, Right, Outer) across Customers, Products, Orders, and Transactions.</li>
                <li><b>Data Transformation:</b> Min-Max Normalization, Z-score Standardization, One-Hot & Label Encoding, and RFM Feature Engineering.</li>
                <li><b>Data Reduction:</b> Multicollinearity filtering, PCA dimensionality reduction with Scree/Loadings, and Stratified Sampling.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span class="metric-badge" style="background: rgba(99, 102, 241, 0.2); color: #6366f1; font-size: 0.9rem; padding: 4px 12px;">CO3</span>
                <h4 style="margin: 0; color: #f8fafc;">Evaluate the effectiveness of classification techniques on pre-processed data</h4>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;">
                <b>Demonstrated Implementation:</b> Implemented within the <b>Classification Analysis</b> section.
            </p>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                <li><b>Supervised Algorithms:</b> Trains Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.</li>
                <li><b>Zero Data Leakage:</b> Preprocessing scaling parameters fitted strictly on training data (X_train) and evaluated out-of-sample on testing data (X_test).</li>
                <li><b>Rigorous Metrics:</b> Real mathematical calculation of Accuracy, Weighted Precision, Recall, and F1-Scores.</li>
                <li><b>Diagnostic Heatmaps:</b> Interactive Confusion Matrix visualization and full Scikit-learn Classification Reports.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span class="metric-badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981; font-size: 0.9rem; padding: 4px 12px;">CO4</span>
                <h4 style="margin: 0; color: #f8fafc;">Examine clustering for customer segmentation with visualizations</h4>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px;">
                <b>Demonstrated Implementation:</b> Implemented within the <b>Customer Segmentation</b> section.
            </p>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                <li><b>Unsupervised Clustering:</b> K-Means clustering applied on customer-level numerical dimensions.</li>
                <li><b>Hyperparameter Tuning:</b> Dual validation of cluster count (K) via Elbow Method (Inertia curve) and Silhouette Score analysis.</li>
                <li><b>Interactive Visuals:</b> 2D and 3D scatter projections including Principal Component mapping.</li>
                <li><b>Data-Driven Persona Profiling:</b> Automated synthesis of customer cohorts (e.g., High-Value Spenders, Loyalists) based on centroid deviations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: ARCHITECTURE & SPECS
    # =========================================================================
    with tab_arch:
        st.subheader("Software Architecture & Technology Stack")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("""
            #### Technology Specifications
            - **Frontend Framework:** Streamlit (Multi-page modular router)
            - **Styling Architecture:** Modern SaaS Dark Navy & Cyan Palette (Custom CSS)
            - **Data Manipulation Engine:** Pandas 2.2+, NumPy 2.1+
            - **Machine Learning Core:** Scikit-learn 1.7+
            - **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
            - **Document Generation:** ReportLab 5.1+ (PDF generation engine)
            - **Spreadsheet Handlers:** OpenPyXL (Excel .xlsx read/write)
            """)

        with col_t2:
            st.markdown("""
            #### Architectural Guarantees
            1. **Lineage Preservation:** Original uploaded raw data is never mutated in-place.
            2. **Audit Logging:** Every transformation step is stamped in the session audit registry.
            3. **Data Leakage Prevention:** Scalers & imputers fit strictly on training splits.
            4. **Graceful Fault Tolerance:** Handles unparseable formats and missing attributes with clear diagnostic feedback.
            """)
