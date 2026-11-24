"""
Utility: Helpers & UI Component Generators
Generates custom HTML/CSS SaaS components:
- Workflow Progress Bar (Raw Data → Cleaning → Integration → Transformation → Reduction → Validation → ML Ready)
- Metric Cards
- Custom Alert Banners
- Section Header Banners
"""

import streamlit as st
from typing import List, Dict, Any


def render_workflow_progress(current_stage: str, completed_dict: Dict[str, bool]):
    """
    Renders an interactive horizontal breadcrumb workflow indicator:
    Raw Data → Cleaning → Integration → Transformation → Reduction → Validation → ML Ready
    """
    stages = [
        ("Raw Data", "1"),
        ("Cleaning", "2"),
        ("Integration", "3"),
        ("Transformation", "4"),
        ("Reduction", "5"),
        ("Validation", "6"),
        ("ML Ready", "7")
    ]

    html = '<div class="workflow-container">'
    for i, (stage, step_num) in enumerate(stages):
        is_current = (stage == current_stage)
        is_done = completed_dict.get(stage, False)
        
        status_class = "active" if is_current else ("completed" if is_done else "pending")
        icon = "✓" if is_done and not is_current else step_num

        html += f"""
        <div class="workflow-step {status_class}">
            <div class="step-circle">{icon}</div>
            <span class="step-label">{stage}</span>
        </div>
        """
        if i < len(stages) - 1:
            line_class = "completed-line" if is_done else ""
            html += f'<div class="workflow-line {line_class}"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    icon: str = "📊",
    badge: str = "",
    badge_color: str = "#06b6d4"
):
    """Renders a sleek, elevated SaaS metric card."""
    badge_html = f'<span class="metric-badge" style="background-color: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55;">{badge}</span>' if badge else ""
    subtitle_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    
    html = f"""
    <div class="metric-card">
        <div class="metric-top">
            <span class="metric-icon">{icon}</span>
            {badge_html}
        </div>
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, icon: str = "⚡", stage_badge: str = ""):
    """Renders a unified SaaS page title banner."""
    badge_html = f'<span class="stage-tag">{stage_badge}</span>' if stage_badge else ""
    html = f"""
    <div class="page-header">
        <div class="header-icon-box">{icon}</div>
        <div class="header-text-box">
            <div class="header-title-row">
                <h1 class="header-title">{title}</h1>
                {badge_html}
            </div>
            <p class="header-subtitle">{subtitle}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_info_box(title: str, content: str, icon: str = "💡", box_type: str = "info"):
    """Renders a styled informational callout card."""
    type_class = f"callout-{box_type}"
    html = f"""
    <div class="callout-card {type_class}">
        <div class="callout-header">
            <span class="callout-icon">{icon}</span>
            <span class="callout-title">{title}</span>
        </div>
        <div class="callout-body">{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
