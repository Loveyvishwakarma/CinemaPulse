"""
CinemaPulse — Main Application Landing Page
===========================================
Streamlit multi-page BI portal entry point.
"""

import streamlit as st
import pandas as pd
from src.ui_helpers import (
    inject_custom_css,
    render_custom_kpi,
    render_sidebar_brand,
    get_cleaned_data,
    get_daily_data,
    format_revenue,
    format_tickets,
    format_percentage,
    COLOR_PALETTE,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CinemaPulse — Cinema Sales Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar_brand()

# ---------------------------------------------------------------------------
# Header & Hero Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding: 0.5rem 0 1.25rem 0;">
        <div style="display: flex; align-items: baseline; gap: 0.75rem;">
            <h1 style="margin: 0;">🎬 CinemaPulse</h1>
            <span class="badge badge-violet" style="font-size: 0.825rem; padding: 0.25rem 0.65rem;">
                Enterprise Analytics
            </span>
        </div>
        <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0.35rem; max-width: 900px; line-height: 1.5;">
            An interactive Business Intelligence platform that unifies theatrical box office transactions, 
            diagnoses seat occupancy patterns, evaluates demand forecasting models, and guides operational decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# High-Level Metric Ribbon (5 Principal Metrics)
# ---------------------------------------------------------------------------
try:
    df_clean = get_cleaned_data()
    df_daily = get_daily_data()

    total_rev = df_clean["total_sales"].sum()
    total_tix = df_clean["tickets_sold"].sum()
    valid_cap = df_clean[df_clean["capacity"] > 0]
    weighted_occ = (valid_cap["tickets_sold"].sum() / valid_cap["capacity"].sum()) * 100.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_custom_kpi(
            "Gross Revenue",
            format_revenue(total_rev),
            "142,418 verified screenings",
        )
    with col2:
        render_custom_kpi(
            "Total Admissions",
            format_tickets(total_tix),
            "Audience ticket sales",
        )
    with col3:
        render_custom_kpi(
            "Weighted Occupancy",
            format_percentage(weighted_occ),
            "Tickets / valid capacity",
        )
    with col4:
        render_custom_kpi(
            "Film Footprint",
            f"{df_clean['film_code'].nunique()} Titles",
            "Across 246 cinema halls",
        )
    with col5:
        render_custom_kpi(
            "Operating History",
            f"{df_daily['date'].nunique()} Days",
            "Feb 21 – Nov 04, 2018",
        )

except Exception as e:
    st.error(f"Error loading summary metrics: {e}")

st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Business Intelligence Workflow Architecture
# ---------------------------------------------------------------------------
st.markdown("#### 🔄 End-to-End Business Intelligence Pipeline")

st.markdown(
    """
    <div style="background: rgba(20, 28, 46, 0.6); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;">
        <div style="display: flex; flex-wrap: wrap; gap: 0.65rem; align-items: center; justify-content: space-between;">
            <div style="text-align: center;">
                <span class="badge badge-violet">1. Raw Data</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">142.5K Rows</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-violet">2. Clean Pipeline</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">Deduplication & Imputation</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-violet">3. In-Depth EDA</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">Statistical Distributions</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-amber">4. Insights</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">Risk Quantification</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-amber">5. Prediction</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">ML Rolling Backtest</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-teal">6. Dashboard</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">Multi-Page UI</div>
            </div>
            <span style="color: #64748B;">➔</span>
            <div style="text-align: center;">
                <span class="badge badge-emerald">7. Decisions</span>
                <div style="font-size: 0.725rem; color: #94A3B8; margin-top: 0.25rem;">Action Matrix</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dashboard Pages Directory (Interactive-styled Feature Cards)
# ---------------------------------------------------------------------------
st.markdown("#### 📑 Analytical Workspace Modules")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown(
        """
        <div class="metric-card" style="min-height: 235px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div class="metric-title" style="color: #A78BFA; font-size: 0.85rem;">Page 1: Executive Overview</div>
                <span class="badge badge-violet">Strategic</span>
            </div>
            <p style="font-size: 0.875rem; color: #CBD5E1; margin: 0.35rem 0 0.75rem 0; line-height: 1.45;">
                Executive performance tracking synthesizing revenue momentum and capacity health for senior management.
            </p>
            <div style="border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 0.65rem;">
                <ul style="font-size: 0.8rem; color: #94A3B8; padding-left: 1.15rem; margin: 0; line-height: 1.55;">
                    <li>7-day rolling revenue momentum trendline</li>
                    <li>Monthly revenue vs admissions seasonality</li>
                    <li>Weighted vs unweighted utilization rates</li>
                    <li>Seamless historical forecast bridge</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        """
        <div class="metric-card" style="min-height: 235px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div class="metric-title" style="color: #38BDF8; font-size: 0.85rem;">Page 2: Sales & Performance</div>
                <span class="badge badge-teal">Operational</span>
            </div>
            <p style="font-size: 0.875rem; color: #CBD5E1; margin: 0.35rem 0 0.75rem 0; line-height: 1.45;">
                Granular diagnostic intelligence analyzing venue rankings, title concentration, and weekly demand cycles.
            </p>
            <div style="border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 0.65rem;">
                <ul style="font-size: 0.8rem; color: #94A3B8; padding-left: 1.15rem; margin: 0; line-height: 1.55;">
                    <li>Top 10 films and cinema venue revenue concentration</li>
                    <li>Weekly demand cycle (Friday peak & Discount Tuesday)</li>
                    <li>Auditorium occupancy spread and ticket price analysis</li>
                    <li>Multi-dimensional filters and sortable tables</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p3:
    st.markdown(
        """
        <div class="metric-card" style="min-height: 235px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div class="metric-title" style="color: #FBBF24; font-size: 0.85rem;">Page 3: Forecast, Risk & Decisions</div>
                <span class="badge badge-amber">Prescriptive</span>
            </div>
            <p style="font-size: 0.875rem; color: #CBD5E1; margin: 0.35rem 0 0.75rem 0; line-height: 1.45;">
                Predictive demand models with empirical prediction intervals, quantified operational risks, and recommendations.
            </p>
            <div style="border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 0.65rem;">
                <ul style="font-size: 0.8rem; color: #94A3B8; padding-left: 1.15rem; margin: 0; line-height: 1.55;">
                    <li>7-day out-of-sample Random Forest forecast & 80% intervals</li>
                    <li>Rigorous chronological rolling-origin backtest metrics</li>
                    <li>Interactive low-occupancy risk threshold simulator</li>
                    <li>Actionable evidence-based decision matrix</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Governance & Dataset Specifications Card
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Data Governance, Privacy & Dataset Specifications", expanded=False):
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
            <ul>
                <li><strong>Dataset Source:</strong> Kaggle Cinema Ticket Reference Dataset (<code>cinemaTicket_Ref.csv</code>).</li>
                <li><strong>Temporal Span:</strong> 142,418 cleaned transaction records across 234 observed operating dates (February 21 – November 04, 2018).</li>
                <li><strong>Entity Privacy:</strong> Film titles and cinema halls are anonymized with unique numerical identifiers (<code>film_code</code>, <code>cinema_code</code>).</li>
                <li><strong>Capacity & Utilization:</strong> Auditorium capacity is derived; weighted occupancy is calculated as $\\sum \\text{Tickets} / \\sum \\text{Capacity}$ on valid screenings (excluding 55 negative capacity rows).</li>
                <li><strong>Nominal Units:</strong> Revenue metrics represent nominal currency units without invented currency signs (empirical scale corresponds to Rial).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color: rgba(148, 163, 184, 0.12); margin: 1.5rem 0 0.75rem 0;'>", unsafe_allow_html=True)
st.caption("🎬 CinemaPulse — IBM SkillsBuild Internship Project • Professional Business Intelligence Prototype")
