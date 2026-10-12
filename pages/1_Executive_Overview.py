"""
1_Executive_Overview.py
=======================
Streamlit Page 1: Executive Overview
High-level historical revenue momentum, admissions, occupancy, and seasonality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui_helpers import (
    inject_custom_css,
    apply_plotly_theme,
    render_custom_kpi,
    render_insight_card,
    render_sidebar_brand,
    get_cleaned_data,
    get_daily_data,
    get_forecast_data,
    format_revenue,
    format_tickets,
    format_percentage,
    COLOR_PALETTE,
)
from src.analysis import compute_kpis, compute_monthly_patterns

st.set_page_config(
    page_title="Executive Overview — CinemaPulse",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()
render_sidebar_brand()

# ---------------------------------------------------------------------------
# Header & Context
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="padding-bottom: 0.75rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <h1 style="margin: 0;">📊 Executive Overview</h1>
            <span class="badge badge-violet" style="font-size: 0.8rem; padding: 0.3rem 0.7rem;">
                Historical Scope: Feb 21 – Nov 04, 2018
            </span>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.25rem; line-height: 1.45;">
            Macro-level executive briefing tracking commercial revenue momentum, audience admissions, and network seat utilization.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load Data & Sidebar Filters
# ---------------------------------------------------------------------------
try:
    df_clean = get_cleaned_data()
    df_daily = get_daily_data()
    df_forecast = get_forecast_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.sidebar.markdown("### 🔍 Historical Date Filter")

min_date = df_daily["date"].min().date()
max_date = df_daily["date"].max().date()

selected_date_range = st.sidebar.date_input(
    "Date Range Window",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filter historical performance to a specific time window.",
)

# Handle single date or range selection
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_dt, end_dt = selected_date_range
else:
    start_dt, end_dt = min_date, max_date

# Filter datasets
filtered_daily = df_daily[
    (df_daily["date"].dt.date >= start_dt) & (df_daily["date"].dt.date <= end_dt)
].copy()

filtered_clean = df_clean[
    (df_clean["date"].dt.date >= start_dt) & (df_clean["date"].dt.date <= end_dt)
].copy()

if filtered_daily.empty or filtered_clean.empty:
    st.warning("No data found for the selected date range. Please broaden your date filter.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: Executive KPI Strip
# ---------------------------------------------------------------------------
kpis = compute_kpis(filtered_clean)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    render_custom_kpi(
        "Total Revenue",
        format_revenue(kpis["total_revenue"]),
        f"Across {kpis['n_days']} observed dates",
    )
with col2:
    render_custom_kpi(
        "Total Admissions",
        format_tickets(kpis["total_tickets"]),
        f"{kpis['total_screenings']:,} logged screenings",
    )
with col3:
    render_custom_kpi(
        "Avg Daily Revenue",
        format_revenue(kpis["avg_daily_revenue"]),
        f"Sum / {kpis['n_days']} active days",
    )
with col4:
    render_custom_kpi(
        "Weighted Occupancy",
        format_percentage(kpis["weighted_occupancy"]),
        "Total tickets / valid capacity",
    )
with col5:
    render_custom_kpi(
        "Active Footprint",
        f"{kpis['active_films']} Films",
        f"Across {kpis['active_cinemas']} cinema halls",
    )

st.caption(
    "💡 *Note: Average daily revenue is computed strictly over observed operating dates in the selected scope. "
    "Weighted occupancy is calculated on records with valid capacity > 0 (excluding 55 invalid capacity rows).*"
)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Primary Chart — Daily Revenue Trend with 7-Day Rolling Average
# ---------------------------------------------------------------------------
st.markdown("#### 📈 Daily Box Office Revenue & 7-Day Moving Trend")

df_trend = filtered_daily.sort_values("date").copy()
df_trend["rolling_7_sales"] = df_trend["total_sales"].rolling(7, min_periods=1).mean()

fig_trend = go.Figure()

# Bar trace for daily revenue (soft violet)
fig_trend.add_trace(
    go.Bar(
        x=df_trend["date"],
        y=df_trend["total_sales"],
        name="Daily Revenue",
        marker_color="rgba(139, 92, 246, 0.45)",
        marker_line_color="rgba(139, 92, 246, 0.8)",
        marker_line_width=0.8,
        hovertemplate="<b>%{x|%Y-%m-%d} (%{text})</b><br>Daily Revenue: %{y:,.0f}<extra></extra>",
        text=df_trend["day_name"],
    )
)

# Line trace for 7-day rolling average (amber)
fig_trend.add_trace(
    go.Scatter(
        x=df_trend["date"],
        y=df_trend["rolling_7_sales"],
        name="7-Day Rolling Trend",
        mode="lines",
        line={"color": "#F59E0B", "width": 2.8},
        hovertemplate="<b>7-Day Trend</b>: %{y:,.0f}<extra></extra>",
    )
)

apply_plotly_theme(fig_trend, title="Daily Revenue Evolution with 7-Day Smoothed Trendline", height=380)
fig_trend.update_layout(
    yaxis_title="Nominal Revenue Units",
    xaxis_title="Date",
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Monthly Seasonality & Historical Forecast Bridge
# ---------------------------------------------------------------------------
col_m1, col_m2 = st.columns([3, 2])

with col_m1:
    st.markdown("#### 📅 Monthly Performance Seasonality")
    monthly_data = compute_monthly_patterns(filtered_daily)

    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])

    # Monthly revenue bars (Violet)
    fig_monthly.add_trace(
        go.Bar(
            x=monthly_data["month_name"],
            y=monthly_data["total_sales"],
            name="Monthly Revenue",
            marker_color="rgba(139, 92, 246, 0.85)",
            hovertemplate="<b>%{x}</b><br>Revenue: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Monthly admissions line (Teal)
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_data["month_name"],
            y=monthly_data["total_tickets"],
            name="Admissions (Tickets)",
            mode="lines+markers",
            line={"color": "#06B6D4", "width": 2.5},
            marker={"size": 7, "color": "#06B6D4", "symbol": "circle"},
            hovertemplate="<b>%{x}</b><br>Admissions: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    apply_plotly_theme(fig_monthly, title="Monthly Revenue vs Ticket Volume", height=360)
    fig_monthly.update_yaxes(
        title_text="Total Revenue",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.08)",
    )
    fig_monthly.update_yaxes(
        title_text="Total Admissions",
        secondary_y=True,
        showgrid=False,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with col_m2:
    st.markdown("#### 🔮 Historical Actuals to 7-Day Forecast Bridge")
    
    # Historical tail (last 14 days of data) + 7 days forecast
    tail_history = df_daily.sort_values("date").tail(14).copy()
    
    fig_hist_fc = go.Figure()
    
    # Historical actual trace (Teal)
    fig_hist_fc.add_trace(
        go.Scatter(
            x=tail_history["date"],
            y=tail_history["total_sales"],
            name="Historical Actuals",
            mode="lines+markers",
            line={"color": "#06B6D4", "width": 2},
            marker={"size": 6, "color": "#06B6D4"},
            hovertemplate="<b>Actual (%{x|%Y-%m-%d})</b>: %{y:,.0f}<extra></extra>",
        )
    )

    # Historical forecast trace (Nov 5 - Nov 11, 2018) (Amber dashed)
    fig_hist_fc.add_trace(
        go.Scatter(
            x=pd.to_datetime(df_forecast["date"]),
            y=df_forecast["forecast_total_sales"],
            name="7-Day Model Forecast",
            mode="lines+markers",
            line={"color": "#F59E0B", "width": 2.5, "dash": "dash"},
            marker={"size": 7, "symbol": "diamond", "color": "#F59E0B"},
            hovertemplate="<b>Forecast (%{x|%Y-%m-%d})</b>: %{y:,.0f}<extra></extra>",
        )
    )

    apply_plotly_theme(
        fig_hist_fc,
        title="Historical Actuals & 7-Day Out-of-Sample Forecast (Nov 2018)",
        height=360,
    )
    fig_hist_fc.update_layout(yaxis_title="Revenue Units", xaxis_title="Date")
    st.plotly_chart(fig_hist_fc, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Verified Executive Findings
# ---------------------------------------------------------------------------
st.markdown("#### 📋 Core Verified Executive Findings")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    render_insight_card(
        "1. Blockbuster Concentration Risk",
        "Top 5 films generate 54.2% of total box office revenue; top 10 drive 75.6%. Top film (code 1554) alone generated 372.7B across 212 venues.",
        "Revenue is heavily exposed to hit releases. Theatrical scheduling must buffer for quiet release cycles.",
        accent="violet",
    )

with col_f2:
    render_insight_card(
        "2. Regional Weekend Demand Surge",
        "Thursday (10.16B) and Friday (11.22B) generate a 1.73x daily revenue premium over Saturday–Wednesday workdays (10.71B vs 6.20B/day).",
        "Staffing, premium pricing, and screening density should peak on Thursday/Friday (the Middle Eastern rest cycle).",
        accent="amber",
    )

with col_f3:
    render_insight_card(
        "3. Capacity Rightsizing Disparity",
        "Overall weighted occupancy is 16.39%, significantly below unweighted mean (19.95%). Over 49.0% of screenings operate below 10% capacity.",
        "A large volume of empty seat inventory represents operational inefficiency. Midweek off-peak bundling is necessary.",
        accent="teal",
    )
