# E-Commerce Data Preprocessing & ML Readiness Platform
> **Academic Capstone Project | Data Warehousing & Data Mining (DWDM) / Machine Learning**

---

## 📌 Executive Overview

The **E-Commerce Data Preprocessing & ML Readiness Platform** is a professional, full-stack, interactive analytics web application built in Python and Streamlit. It provides an end-to-end framework that transforms raw, uncurated, and unstructured e-commerce transactions into clean, consistent, mathematically optimized, validated, and machine-learning-ready datasets.

The platform bridges real-world data mining concepts with production machine learning pipelines, featuring:
1. **Module 1: Data Collection, Cleaning & Integration**
2. **Module 2: Data Transformation & Data Reduction**
3. **Module 3: Data Validation, Visualization & ML Readiness**
4. **Supervised Classification Analysis (Logistic Regression, Decision Tree, Random Forest)**
5. **Unsupervised Customer Segmentation (K-Means, Elbow Method, Silhouette Optimization)**
6. **Certified 14-Section PDF Audit Reports & Data Export Hub (CSV/Excel)**

---

## 🏗️ System Architecture & Workflow Pipeline

```
Raw Data Ingestion (CSV / XLSX / JSON)
        │
        ▼
[Module 1: Data Cleaning & Integration]
  ├── Missing Value Imputation (Mean, Median, Mode, Constant, Dropping)
  ├── Exact & Partial Duplicate Purging
  ├── Outlier Handling (IQR & Z-Score Detection, Box Plots, Winsorization)
  ├── Data Type Casting (Integer, Float, String, DateTime, Category)
  ├── Categorical Standardizing (Gender Normalization, Title Case, Trimming)
  └── Multi-Table Joins (Inner, Left, Right, Outer across Customers, Orders, etc.)
        │
        ▼
[Module 2: Transformation & Reduction]
  ├── Normalization (Min-Max Scaling [0, 1] & Z-score Standardization)
  ├── Categorical Encoding (One-Hot Encoding, Label Encoding)
  ├── Feature Engineering (Total Amount = Price × Qty, RFM Metrics, Date Parsing)
  ├── Discretization / Binning (Equal-Width, Equal-Frequency)
  ├── Multicollinearity Filtering (|r| ≥ 0.75 Pearson Heatmaps)
  ├── Principal Component Analysis (PCA Scree, Loadings & 2D Projection)
  └── Sampling (Random, Stratified, Systematic)
        │
        ▼
[Module 3: Validation, Visualization & ML Readiness]
  ├── 8-Point Data Quality Audit & Weighted Quality Score Certification (0-100%)
  ├── Interactive E-Commerce Visualizations (Sales Trends, Cohorts, Orders)
  └── Zero-Leakage Train/Test Partitioning (Fitting exclusively on X_train)
        │
        ├────────────────────────────────┬────────────────────────────────┐
        ▼                                ▼                                ▼
[Classification Analysis]     [Customer Segmentation]         [Reports & Exports]
  • Logistic Regression         • K-Means Clustering            • Checkpoint Downloads
  • Decision Tree               • Elbow Method (Inertia)        • X_train / X_test CSV
  • Random Forest               • Silhouette Optimization       • Certified PDF Audit
  • Real Confusion Matrix       • Behavioral Personas           • 14-Section Report
```

---

## 🎯 Academic Course Outcome Alignment

This project satisfies the practical requirements of **Data Warehousing & Data Mining (DWDM)**:

| Course Outcome | Academic Competency | Platform Implementation |
| :--- | :--- | :--- |
| **CO2** | Apply suitable Data Preprocessing techniques for effective data preparation | **Module 1 & Module 2:** Automated missing value imputation, deduplication, IQR/Z-score outlier capping, relational joins, Min-Max/Z-Score scaling, One-Hot/Label encoding, and PCA dimensionality reduction. |
| **CO3** | Evaluate the effectiveness of classification techniques on pre-processed data | **Classification Analysis:** Evaluates Logistic Regression, Decision Tree, and Random Forest on leakage-free test splits with mathematical Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps. |
| **CO4** | Examine clustering for customer segmentation with visualizations | **Customer Segmentation:** K-Means clustering with dual optimization via Elbow curve and Silhouette analysis, 2D/3D scatter plots, and data-driven customer personas. |

---

## 🌍 UN Sustainable Development Goals (SDG) Alignment

- **SDG 8: Decent Work & Economic Growth:** Enhances small business productivity and logistical decision-making through automated retail analytics.
- **SDG 9: Industry, Innovation & Infrastructure:** Establishes resilient, reproducible data engineering infrastructure that bridges raw enterprise data with artificial intelligence.
- **SDG 12: Responsible Consumption & Production:** Uncovers product demand patterns, inventory velocity, and returns to minimize retail overproduction and material waste.

---

## 💻 Tech Stack

- **Frontend & App Framework:** Streamlit (Multi-page router, reactive state management)
- **Styling:** Custom CSS with Dark Navy SaaS Palette, Cyan/Teal Accents, and Responsive Cards
- **Data Engineering:** Pandas 2.2+, NumPy 2.1+
- **Machine Learning & Preprocessing:** Scikit-learn 1.7+ (`StandardScaler`, `MinMaxScaler`, `PCA`, `KMeans`, `RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression`)
- **Data Visualizations:** Plotly Express & Plotly Graph Objects (Interactive 2D/3D charts)
- **Reporting Engine:** ReportLab 5.1+ (Two-pass NumberedCanvas PDF generation)
- **Spreadsheets:** OpenPyXL (Excel `.xlsx` serialization)

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Directory Structure

```
DWDM Capstone/
│
├── app.py                     # Main application entry point & router
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive architectural documentation
│
├── pages/
│   ├── dashboard.py           # Dashboard overview & summary metrics
│   ├── module1_cleaning.py    # Module 1: Ingestion, Cleaning & Integration
│   ├── module2_transformation.py # Module 2: Transformation & Reduction
│   ├── module3_validation.py  # Module 3: Validation, EDA & ML Readiness
│   ├── classification.py      # Classification models & performance benchmark
│   ├── clustering.py          # Customer segmentation & K-Means clustering
│   ├── reports.py             # Export downloads & 14-section audit PDF
│   └── about.py               # SDGs, Course Outcomes & System Architecture
│
├── modules/
│   ├── data_collection.py     # Multi-format ingestion & structural analysis
│   ├── data_cleaning.py       # Imputation, deduplication, outliers, casting
│   ├── data_integration.py    # Relational joins & table concatenations
│   ├── data_transformation.py # Scaling, encoding, feature engineering, binning
│   ├── data_reduction.py      # Multicollinearity, PCA decomposition, sampling
│   ├── data_validation.py     # 8-point checks & Data Quality Scoring
│   ├── visualization.py       # Domain-specific Plotly visualizations
│   ├── classification_models.py # Model training & evaluation (no leakage)
│   ├── clustering_models.py   # K-Means, Elbow, Silhouette & persona derivation
│   └── report_generation.py   # Publication-quality ReportLab PDF engine
│
├── utils/
│   ├── session_manager.py     # Streamlit session state & lineage tracker
│   ├── file_handler.py        # CSV, Excel, and JSON binary encoders
│   └── helpers.py             # Reusable SaaS cards & workflow breadcrumbs
│
├── assets/
│   └── style.css              # Custom SaaS dark theme styling
│
└── sample_data/
    ├── ecommerce_main.csv     # 1,235-row benchmark dataset with realistic noise
    ├── customers.csv          # Relational customers entity
    ├── products.csv           # Relational products catalog
    ├── orders.csv             # Relational orders ledger
    └── transactions.csv       # Relational financial transactions
```

---

## 🛡️ Data Leakage Prevention Guarantee
All feature scaling transformations (`StandardScaler`, `MinMaxScaler`) and missing value estimators are strictly **fitted on the training partition (`X_train`)** and applied out-of-sample to the testing partition (`X_test`), preventing optimistic performance bias and model contamination.
