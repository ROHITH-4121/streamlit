"""
Module 3: Exploratory Data Analysis & Visualization
Provides interactive Plotly charts with a modern dark SaaS aesthetic:
- Sales Analysis (Trends, Revenue by Category, Payment Methods)
- Customer Analysis (Age, Gender, Frequency, Top Customers)
- Product Analysis (Best sellers, Categories, Quantity, Prices)
- Order Analysis (Status, Delivery, Monthly Counts)
- Generic Interactive Multi-Type Chart Builder
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# SaaS Theme Palette
PALETTE = {
    "primary": "#06b6d4",       # Cyan
    "secondary": "#6366f1",     # Indigo
    "accent": "#38bdf8",        # Sky
    "success": "#10b981",       # Emerald
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Rose
    "background": "#0f172a",    # Dark slate
    "card_bg": "#1e293b",       # Slate 800
    "text": "#f8fafc",
    "grid": "#334155",
    "colorscale": ["#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e"]
}

def apply_saas_layout(fig, title: str = "", height: int = 420):
    """Applies a consistent sleek dark SaaS theme to all Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": "#f8fafc", "family": "Inter, sans-serif"}
        },
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        margin=dict(l=40, r=30, t=55, b=40),
        height=height,
        font=dict(color="#94a3b8", family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor="#334155",
            showline=True,
            linecolor="#475569",
            tickfont=dict(color="#cbd5e1")
        ),
        yaxis=dict(
            gridcolor="#334155",
            showline=True,
            linecolor="#475569",
            tickfont=dict(color="#cbd5e1")
        )
    )
    return fig


def plot_sales_trends(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots monthly sales and order revenue over time."""
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    spend_col = next((c for c in ["Total Amount", "Amount Paid", "Price"] if c in df.columns), None)
    
    if not date_col or not spend_col:
        return None

    temp_df = df.dropna(subset=[date_col, spend_col]).copy()
    try:
        temp_df["ParsedDate"] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=["ParsedDate"])
        temp_df["YearMonth"] = temp_df["ParsedDate"].dt.to_period("M").astype(str)
        monthly = temp_df.groupby("YearMonth").agg(
            TotalRevenue=(spend_col, "sum"),
            OrderCount=(spend_col, "count")
        ).reset_index().sort_values(by="YearMonth")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["YearMonth"],
            y=monthly["TotalRevenue"],
            name="Revenue ($)",
            marker_color="#06b6d4",
            opacity=0.85
        ))
        fig.add_trace(go.Scatter(
            x=monthly["YearMonth"],
            y=monthly["OrderCount"] * (monthly["TotalRevenue"].max() / max(monthly["OrderCount"].max(), 1)),
            name="Order Volume (Scaled)",
            mode="lines+markers",
            line=dict(color="#ec4899", width=3)
        ))
        return apply_saas_layout(fig, "Monthly Revenue Trends & Order Velocity")
    except Exception:
        return None


def plot_revenue_by_category(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots total sales by product category."""
    cat_col = next((c for c in df.columns if "category" in c.lower()), None)
    spend_col = next((c for c in ["Total Amount", "Amount Paid", "Price"] if c in df.columns), None)
    
    if not cat_col or not spend_col:
        return None

    cat_sales = df.groupby(cat_col)[spend_col].sum().reset_index().sort_values(by=spend_col, ascending=False)
    fig = px.bar(
        cat_sales,
        x=cat_col,
        y=spend_col,
        color=cat_col,
        color_discrete_sequence=PALETTE["colorscale"],
        text_auto=".2s"
    )
    fig.update_traces(textposition="outside")
    return apply_saas_layout(fig, f"Revenue by {cat_col}")


def plot_revenue_by_payment_method(df: pd.DataFrame) -> Optional[go.Figure]:
    """Donut chart for payment method distribution."""
    pay_col = next((c for c in df.columns if "payment" in c.lower()), None)
    if not pay_col:
        return None

    counts = df[pay_col].value_counts().reset_index()
    counts.columns = [pay_col, "Count"]
    fig = px.pie(
        counts,
        names=pay_col,
        values="Count",
        hole=0.45,
        color_discrete_sequence=PALETTE["colorscale"]
    )
    return apply_saas_layout(fig, f"Share of Orders by {pay_col}")


def plot_customer_age_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots age histogram with box plot overlay."""
    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    if not age_col or not pd.api.types.is_numeric_dtype(df[age_col]):
        return None

    fig = px.histogram(
        df.dropna(subset=[age_col]),
        x=age_col,
        nbins=20,
        marginal="box",
        color_discrete_sequence=["#38bdf8"]
    )
    return apply_saas_layout(fig, "Customer Age Distribution & Quartiles")


def plot_gender_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots customer gender distribution."""
    gen_col = next((c for c in df.columns if "gender" in c.lower()), None)
    if not gen_col:
        return None

    counts = df[gen_col].value_counts(dropna=False).reset_index()
    counts.columns = [gen_col, "Count"]
    fig = px.bar(
        counts,
        x=gen_col,
        y="Count",
        color=gen_col,
        color_discrete_sequence=["#06b6d4", "#a855f7", "#64748b"],
        text_auto=True
    )
    return apply_saas_layout(fig, "Customer Gender Distribution")


def plot_top_customers(df: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
    """Horizontal bar chart of top spending customers."""
    cust_col = next((c for c in df.columns if "customer" in c.lower() and ("name" in c.lower() or "id" in c.lower())), None)
    spend_col = next((c for c in ["Total Amount", "Amount Paid", "Price"] if c in df.columns), None)
    
    if not cust_col or not spend_col:
        return None

    top_cust = df.groupby(cust_col)[spend_col].sum().reset_index().sort_values(by=spend_col, ascending=True).tail(top_n)
    fig = px.bar(
        top_cust,
        y=cust_col,
        x=spend_col,
        orientation="h",
        color=spend_col,
        color_continuous_scale="Tealgrn",
        text_auto=".2s"
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_saas_layout(fig, f"Top {top_n} Customers by Total Spend")


def plot_delivery_status(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plots order fulfillment and delivery status breakdown."""
    status_col = next((c for c in df.columns if "status" in c.lower()), None)
    if not status_col:
        return None

    counts = df[status_col].value_counts().reset_index()
    counts.columns = [status_col, "Orders"]
    fig = px.pie(
        counts,
        names=status_col,
        values="Orders",
        hole=0.4,
        color_discrete_sequence=["#10b981", "#06b6d4", "#f59e0b", "#ef4444", "#8b5cf6"]
    )
    return apply_saas_layout(fig, f"Order Breakdown by {status_col}")


def plot_best_selling_products(df: pd.DataFrame, top_n: int = 8) -> Optional[go.Figure]:
    """Plots top selling products by units and revenue."""
    prod_col = next((c for c in df.columns if "product" in c.lower() and "name" in c.lower()), None)
    qty_col = next((c for c in df.columns if "quantity" in c.lower() or "qty" in c.lower()), None)
    
    if not prod_col or not qty_col:
        return None

    top_prods = df.groupby(prod_col)[qty_col].sum().reset_index().sort_values(by=qty_col, ascending=False).head(top_n)
    fig = px.bar(
        top_prods,
        x=prod_col,
        y=qty_col,
        color=qty_col,
        color_continuous_scale="Viridis",
        text_auto=True
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_saas_layout(fig, f"Top {top_n} Best Selling Products by Volume")


def build_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Renders an arbitrary interactive Plotly visualization selected by the user.
    """
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True, color_discrete_sequence=PALETTE["colorscale"])
        elif chart_type == "Pie Chart":
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.pie(counts, names=x_col, values="Count", color_discrete_sequence=PALETTE["colorscale"], hole=0.35)
        else:
            return None
        return apply_saas_layout(fig, f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col else ""))
    except Exception:
        return None
