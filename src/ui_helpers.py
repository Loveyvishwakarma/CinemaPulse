"""
ui_helpers.py
=============
Centralized design system, UI styling, Plotly theming, number formatting,
and cached data loaders for the CinemaPulse Business Intelligence Dashboard.
"""

from typing import Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.data_loader import load_cleaned_data, load_daily_sales, load_forecast_data


# ---------------------------------------------------------------------------
# Global Design System & Palette (Cinematic Violet & Restrained Teal Theme)
# ---------------------------------------------------------------------------
COLOR_PALETTE = {
    "revenue": "#8B5CF6",          # Cinematic Violet (Primary Revenue)
    "revenue_soft": "rgba(139, 92, 246, 0.35)",
    "admissions": "#06B6D4",       # Restrained Teal (Primary Admissions)
    "admissions_soft": "rgba(6, 182, 212, 0.35)",
    "forecast": "#F59E0B",         # Amber / Gold (Out-of-sample Forecast)
    "forecast_soft": "rgba(245, 158, 11, 0.2)",
    "baseline": "#64748B",         # Slate Muted (Baselines)
    "success": "#10B981",          # Emerald Green (Healthy indicators)
    "warning": "#F59E0B",          # Warm Amber (Threshold alerts)
    "danger": "#EF4444",           # Crimson (High risk)
    "bg_main": "#0A0E17",          # Deep Charcoal / Navy App Canvas
    "bg_sidebar": "#0E1424",       # Slightly distinct Dark Navy Sidebar
    "card_surface": "#141C2E",     # Elevated Card Surface
    "card_border": "rgba(148, 163, 184, 0.14)", # Subtle Card Border
    "text_primary": "#F8FAFC",     # High-contrast Off-White
    "text_secondary": "#94A3B8",   # Muted Cool Gray
    "text_caption": "#64748B",     # De-emphasized Caption
    "weekend": "#EC4899",          # Vibrant Pink-Violet for Weekend highlights
}


def inject_custom_css():
    """Inject cohesive, professional executive SaaS styling into Streamlit."""
    st.markdown(
        """
        <style>
        /* Import Google Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Base Typography & Background Styling */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F8FAFC;
        }

        /* App canvas & Sidebar styling */
        .stApp {
            background-color: #0A0E17;
        }
        [data-testid="stSidebar"] {
            background-color: #0E1424 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
        }

        /* Header typography adjustments */
        h1 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.25rem !important;
        }
        h2, h3 {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
        }
        h4 {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.6rem !important;
        }

        /* Executive KPI Cards */
        .metric-card {
            background: linear-gradient(180deg, rgba(20, 28, 46, 0.95) 0%, rgba(14, 20, 36, 0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.35);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        .metric-card:hover {
            border-color: rgba(139, 92, 246, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px -4px rgba(139, 92, 246, 0.15);
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.6) 0%, rgba(6, 182, 212, 0.6) 100%);
            opacity: 0.8;
        }
        .metric-title {
            color: #94A3B8;
            font-size: 0.775rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.15;
            letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
            margin-bottom: 0.3rem;
        }
        .metric-subtitle {
            color: #64748B;
            font-size: 0.775rem;
            font-weight: 400;
            line-height: 1.35;
        }

        /* Insight and Narrative Callout Cards */
        .insight-card {
            background: rgba(20, 28, 46, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-left: 4px solid #8B5CF6;
            border-radius: 0 10px 10px 0;
            padding: 1.05rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        .insight-card-teal {
            border-left-color: #06B6D4;
        }
        .insight-card-amber {
            border-left-color: #F59E0B;
        }
        .insight-title {
            color: #F8FAFC;
            font-size: 0.925rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .insight-body {
            color: #CBD5E1;
            font-size: 0.85rem;
            line-height: 1.45;
        }

        /* Section Header Component */
        .section-header {
            margin-top: 1.25rem;
            margin-bottom: 0.85rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }
        .section-header-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #F8FAFC;
            letter-spacing: -0.01em;
        }
        .section-header-desc {
            font-size: 0.825rem;
            color: #94A3B8;
            margin-top: 0.15rem;
        }

        /* Filter Panel Styling */
        .filter-panel {
            background: rgba(20, 28, 46, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.25rem;
        }

        /* Clean Badges */
        .badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-violet { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
        .badge-teal   { background: rgba(6, 182, 212, 0.15); color: #38BDF8; border: 1px solid rgba(6, 182, 212, 0.3); }
        .badge-amber  { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-emerald{ background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }

        /* General Streamlit widget enhancements */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(14, 20, 36, 0.6);
            padding: 6px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: #94A3B8;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 0.875rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(139, 92, 246, 0.2) !important;
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        /* Streamlit metric default override */
        [data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """
    Apply a unified, premium executive dark analytics theme to Plotly figures.
    """
    fig.update_layout(
        title={
            "text": title,
            "y": 0.96,
            "x": 0.015,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 14, "color": "#F8FAFC", "family": "Inter, sans-serif"},
        },
        height=height,
        margin={"l": 45, "r": 25, "t": 55, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 20, 36, 0.5)",
        font={"family": "Inter, sans-serif", "color": "#94A3B8", "size": 11.5},
        hoverlabel={
            "bgcolor": "#1E293B",
            "font_size": 12,
            "font_family": "Inter, sans-serif",
            "bordercolor": "rgba(148, 163, 184, 0.3)",
        },
        xaxis={
            "gridcolor": "rgba(148, 163, 184, 0.08)",
            "zerolinecolor": "rgba(148, 163, 184, 0.15)",
            "showgrid": True,
            "tickfont": {"color": "#94A3B8", "size": 10.5},
            "title_font": {"color": "#CBD5E1", "size": 11.5},
        },
        yaxis={
            "gridcolor": "rgba(148, 163, 184, 0.08)",
            "zerolinecolor": "rgba(148, 163, 184, 0.15)",
            "showgrid": True,
            "tickfont": {"color": "#94A3B8", "size": 10.5},
            "title_font": {"color": "#CBD5E1", "size": 11.5},
        },
        legend={
            "bgcolor": "rgba(14, 20, 36, 0.7)",
            "bordercolor": "rgba(148, 163, 184, 0.15)",
            "borderwidth": 1,
            "font": {"color": "#CBD5E1", "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    return fig


# ---------------------------------------------------------------------------
# Number Formatting Helpers (Exact behavior preserved for tests)
# ---------------------------------------------------------------------------
def format_revenue(val: float) -> str:
    """Format nominal revenue with B/M/K scale without invented currency symbols."""
    if val is None or np.isnan(val):
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{val / 1e3:.1f}K"
    else:
        return f"{val:,.0f}"


def format_tickets(val: float) -> str:
    """Format ticket volumes with M/K scale."""
    if val is None or np.isnan(val):
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1e6:
        return f"{val / 1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{val / 1e3:.1f}K"
    else:
        return f"{val:,.0f}"


def format_percentage(val: float) -> str:
    """Format percentage value."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.1f}%"


def render_custom_kpi(title: str, value: str, subtitle: str):
    """Render an executive styled HTML KPI card."""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_insight_card(title: str, finding: str, implication: str, accent: str = "violet"):
    """Render a structured executive takeaway card separating finding and business implication."""
    card_cls = "insight-card"
    if accent == "teal":
        card_cls += " insight-card-teal"
    elif accent == "amber":
        card_cls += " insight-card-amber"

    html = f"""
    <div class="{card_cls}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">
            <strong>Finding:</strong> {finding}<br>
            <span style="color: #94A3B8;"><strong>Strategic Implication:</strong> {implication}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render consistent, professional CinemaPulse identity in the Streamlit sidebar."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.12); margin-bottom: 1rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
                🎬 CinemaPulse
            </div>
            <div style="font-size: 0.775rem; color: #94A3B8; margin-top: 0.15rem;">
                Cinema Sales & Demand Intelligence
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="badge badge-violet">● Historical BI</span>
                <span class="badge badge-teal">v1.0 Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cleaned_data() -> pd.DataFrame:
    """Load cleaned transaction data once into memory."""
    return load_cleaned_data()


@st.cache_data(show_spinner=False)
def get_daily_data() -> pd.DataFrame:
    """Load daily aggregate data once into memory."""
    return load_daily_sales()


@st.cache_data(show_spinner=False)
def get_forecast_data() -> pd.DataFrame:
    """Load 7-day forecast artifact once into memory."""
    return load_forecast_data()
