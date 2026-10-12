"""
2_Sales_Performance.py
======================
Streamlit Page 2: Sales & Performance Analysis
Granular breakdowns across films, cinema venues, weekly cycles, and seat occupancy.
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
    render_sidebar_brand,
    get_cleaned_data,
    get_daily_data,
    format_revenue,
    format_tickets,
    format_percentage,
    COLOR_PALETTE,
)
from src.analysis import (
    compute_top_films,
    compute_top_cinemas,
    compute_dow_patterns,
    compute_kpis,
    DOW_ORDER,
)

st.set_page_config(
    page_title="Sales Performance — CinemaPulse",
    page_icon="💰",
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
            <h1 style="margin: 0;">💰 Sales & Performance Analysis</h1>
            <span class="badge badge-teal" style="font-size: 0.8rem; padding: 0.3rem 0.7rem;">
                Operational Diagnostics
            </span>
        </div>
        <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.25rem; line-height: 1.45;">
            Granular exploration of revenue concentration, venue commercial yield, weekly pricing dynamics, and auditorium occupancy distribution.
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
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.sidebar.markdown("### 🔍 Performance Filters")

# 1. Date Range
min_date = df_clean["date"].min().date()
max_date = df_clean["date"].max().date()
selected_dates = st.sidebar.date_input(
    "Date Range Window",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filter transactions by date range.",
)
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    s_dt, e_dt = selected_dates
else:
    s_dt, e_dt = min_date, max_date

# 2. Day of Week Filter
selected_dows = st.sidebar.multiselect(
    "Day of Week Schedule",
    options=DOW_ORDER,
    default=DOW_ORDER,
    help="Filter transactions by day of the week.",
)

# 3. Film Code Filter
film_options = sorted(df_clean["film_code"].unique().tolist())
selected_films = st.sidebar.multiselect(
    "Film Code (optional filter)",
    options=film_options,
    default=[],
    help="Leave empty to include all films.",
)

# 4. Cinema Code Filter
cinema_options = sorted(df_clean["cinema_code"].unique().tolist())
selected_cinemas = st.sidebar.multiselect(
    "Cinema Code (optional filter)",
    options=cinema_options,
    default=[],
    help="Leave empty to include all cinemas.",
)

# Apply filters to transaction data
filtered_df = df_clean[
    (df_clean["date"].dt.date >= s_dt) & (df_clean["date"].dt.date <= e_dt)
].copy()

if selected_dows:
    filtered_df = filtered_df[filtered_df["day_name"].isin(selected_dows)]

if selected_films:
    filtered_df = filtered_df[filtered_df["film_code"].isin(selected_films)]

if selected_cinemas:
    filtered_df = filtered_df[filtered_df["cinema_code"].isin(selected_cinemas)]

if filtered_df.empty:
    st.warning("No records match your selected filter criteria. Please broaden your selection.")
    st.stop()

# Filtered daily data for DOW analysis
filtered_daily = df_daily[
    (df_daily["date"].dt.date >= s_dt) & (df_daily["date"].dt.date <= e_dt)
].copy()
if selected_dows:
    filtered_daily = filtered_daily[filtered_daily["day_name"].isin(selected_dows)]

# ---------------------------------------------------------------------------
# Section 0: Scope Summary Ribbon
# ---------------------------------------------------------------------------
scope_kpis = compute_kpis(filtered_df)

col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
with col_s1:
    render_custom_kpi("Filtered Revenue", format_revenue(scope_kpis["total_revenue"]), f"{scope_kpis['n_days']} active dates in scope")
with col_s2:
    render_custom_kpi("Filtered Admissions", format_tickets(scope_kpis["total_tickets"]), f"{scope_kpis['total_screenings']:,} showtimes")
with col_s3:
    render_custom_kpi("Weighted Occupancy", format_percentage(scope_kpis["weighted_occupancy"]), "Filtered capacity fill rate")
with col_s4:
    render_custom_kpi("Films in Scope", f"{scope_kpis['active_films']} Titles", "Selected catalog")
with col_s5:
    render_custom_kpi("Cinemas in Scope", f"{scope_kpis['active_cinemas']} Halls", "Selected footprint")

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: Top Films & Top Cinemas (Horizontal Rankings)
# ---------------------------------------------------------------------------
col_f, col_c = st.columns(2)

with col_f:
    st.markdown("#### 🎬 Top 10 Films by Revenue")
    top_films_df = compute_top_films(filtered_df, n=10)

    if not top_films_df.empty:
        fig_films = go.Figure(
            go.Bar(
                x=top_films_df["total_revenue"],
                y=top_films_df["film_code"].astype(str),
                orientation="h",
                marker=dict(
                    color=top_films_df["total_revenue"],
                    colorscale=[[0, "rgba(139, 92, 246, 0.4)"], [1, "rgba(139, 92, 246, 0.95)"]],
                ),
                hovertemplate=(
                    "<b>Film Code: %{y}</b><br>"
                    "Total Revenue: %{x:,.0f}<br>"
                    "Tickets: %{customdata[0]:,.0f}<br>"
                    "Avg Occupancy: %{customdata[1]:.1f}%<extra></extra>"
                ),
                customdata=top_films_df[["total_tickets", "avg_occupancy"]].values,
            )
        )
        fig_films.update_layout(yaxis={"autorange": "reversed"})
        apply_plotly_theme(fig_films, title="Top Grossing Films (film_code)", height=370)
        fig_films.update_xaxes(title_text="Total Revenue Units")
        fig_films.update_yaxes(title_text="Film Code")
        st.plotly_chart(fig_films, use_container_width=True)
    else:
        st.info("No film data available.")

with col_c:
    st.markdown("#### 🏢 Top 10 Cinemas by Revenue")
    top_cinemas_df = compute_top_cinemas(filtered_df, n=10)

    if not top_cinemas_df.empty:
        fig_cinemas = go.Figure(
            go.Bar(
                x=top_cinemas_df["total_revenue"],
                y=top_cinemas_df["cinema_code"].astype(str),
                orientation="h",
                marker=dict(
                    color=top_cinemas_df["total_revenue"],
                    colorscale=[[0, "rgba(6, 182, 212, 0.4)"], [1, "rgba(6, 182, 212, 0.95)"]],
                ),
                hovertemplate=(
                    "<b>Cinema Code: %{y}</b><br>"
                    "Total Revenue: %{x:,.0f}<br>"
                    "Tickets: %{customdata[0]:,.0f}<br>"
                    "Screenings: %{customdata[1]:,.0f}<extra></extra>"
                ),
                customdata=top_cinemas_df[["total_tickets", "n_screenings"]].values,
            )
        )
        fig_cinemas.update_layout(yaxis={"autorange": "reversed"})
        apply_plotly_theme(fig_cinemas, title="Top Grossing Venues (cinema_code)", height=370)
        fig_cinemas.update_xaxes(title_text="Total Revenue Units")
        fig_cinemas.update_yaxes(title_text="Cinema Code")
        st.plotly_chart(fig_cinemas, use_container_width=True)
    else:
        st.info("No cinema data available.")

st.caption(
    "⚠️ *Analytical Guidance: Film and cinema metadata (names, genres, city locations) are anonymised in the dataset. "
    "Low total revenue for a cinema does not automatically imply underperformance—venues with fewer recorded dates "
    "or smaller auditoriums may have high per-screening efficiency.*"
)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Weekly Demand Cycle & Tuesday Discount Dynamics
# ---------------------------------------------------------------------------
st.markdown("#### 🗓️ Day-of-Week Demand Cycle & Discount Tuesday Dynamics")
dow_df = compute_dow_patterns(filtered_daily)

if not dow_df.empty:
    fig_dow = make_subplots(specs=[[{"secondary_y": True}]])

    # Revenue bar (Violet for regular, Pink for weekend peaks, Amber for Tuesday)
    bar_colors = []
    for d in dow_df["day_name"]:
        if d in ["Thursday", "Friday"]:
            bar_colors.append("rgba(139, 92, 246, 0.9)")  # Violet peak
        elif d == "Tuesday":
            bar_colors.append("rgba(245, 158, 11, 0.85)")  # Amber promo
        else:
            bar_colors.append("rgba(100, 116, 139, 0.65)") # Muted slate

    fig_dow.add_trace(
        go.Bar(
            x=dow_df["day_name"],
            y=dow_df["avg_sales"],
            name="Avg Daily Revenue",
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Avg Revenue: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Tickets line (Teal)
    fig_dow.add_trace(
        go.Scatter(
            x=dow_df["day_name"],
            y=dow_df["avg_tickets"],
            name="Avg Daily Admissions",
            mode="lines+markers",
            line={"color": "#06B6D4", "width": 2.6},
            marker={"size": 8, "color": "#06B6D4"},
            hovertemplate="<b>%{x}</b><br>Avg Tickets: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    apply_plotly_theme(fig_dow, title="Day-of-Week Performance: Revenue vs Admissions Dynamics", height=380)
    fig_dow.update_yaxes(
        title_text="Avg Daily Revenue",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.08)",
    )
    fig_dow.update_yaxes(
        title_text="Avg Daily Tickets Sold",
        secondary_y=True,
        showgrid=False,
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown(
        """
        <div style="background: rgba(20, 28, 46, 0.7); border-left: 4px solid #8B5CF6; border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem; margin-bottom: 1.25rem;">
            <div style="font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">Cycle Insight: Dual Peak Structure</div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.45;">
                <strong>Revenue Peak:</strong> <strong>Friday</strong> (11.22B) and <strong>Thursday</strong> (10.16B) drive the highest box office revenues, capturing regional weekend demand.<br>
                <strong>Volume Peak:</strong> <strong>Tuesday</strong> exhibits an enormous ticket admission surge (<strong>169.7K tickets/day</strong>, 37.1% occupancy), demonstrating strong consumer price sensitivity to promotional discount pricing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 3: Occupancy Analysis & Ticket Price Spread
# ---------------------------------------------------------------------------
st.markdown("#### 💺 Auditorium Occupancy Patterns & Ticket Price Spread")

valid_cap_df = filtered_df[filtered_df["capacity"] > 0]
unweighted_mean = valid_cap_df["occu_perc"].mean() if not valid_cap_df.empty else 0.0
weighted_mean = (
    (valid_cap_df["tickets_sold"].sum() / valid_cap_df["capacity"].sum()) * 100.0
    if not valid_cap_df.empty and valid_cap_df["capacity"].sum() > 0
    else 0.0
)
under_10_share = (
    (valid_cap_df["occu_perc"] < 10.0).sum() / len(valid_cap_df) * 100.0
    if not valid_cap_df.empty
    else 0.0
)

col_o1, col_o2, col_o3 = st.columns(3)
with col_o1:
    render_custom_kpi("Unweighted Mean Occupancy", format_percentage(unweighted_mean), "Simple arithmetic mean of occu_perc")
with col_o2:
    render_custom_kpi("Weighted Utilization", format_percentage(weighted_mean), "Total tickets / Total valid capacity")
with col_o3:
    render_custom_kpi("Under-Occupied Shows", format_percentage(under_10_share), "Screenings running below 10% fill rate")

col_hist, col_price = st.columns(2)

with col_hist:
    # Occupancy distribution histogram (Teal)
    fig_hist = go.Figure(
        go.Histogram(
            x=valid_cap_df["occu_perc"],
            nbinsx=40,
            marker_color="rgba(6, 182, 212, 0.75)",
            marker_line_color="rgba(6, 182, 212, 0.9)",
            marker_line_width=0.8,
            hovertemplate="Occupancy: %{x:.1f}%<br>Count: %{y:,}<extra></extra>",
        )
    )
    apply_plotly_theme(fig_hist, title="Distribution of Screening Occupancy Rates (%)", height=340)
    fig_hist.update_xaxes(title_text="Occupancy Rate (%)", range=[0, 100])
    fig_hist.update_yaxes(title_text="Screening Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_price:
    # Price distribution boxplot by day of week (Violet)
    fig_price = go.Figure(
        go.Box(
            x=filtered_df["day_name"],
            y=filtered_df["ticket_price"],
            marker_color="#8B5CF6",
            line_color="#A78BFA",
            boxmean=True,
            hovertemplate="<b>%{x}</b><br>Ticket Price: %{y:,.0f}<extra></extra>",
        )
    )
    apply_plotly_theme(fig_price, title="Ticket Price Distribution Across Weekdays", height=340)
    fig_price.update_xaxes(categoryorder="array", categoryarray=DOW_ORDER)
    fig_price.update_yaxes(title_text="Nominal Ticket Price")
    st.plotly_chart(fig_price, use_container_width=True)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Detailed Performance Data Tables
# ---------------------------------------------------------------------------
st.markdown("#### 📊 Granular Catalog Performance Tables")

tab_films, tab_cinemas = st.tabs(["🎬 Film Performance Table", "🏢 Cinema Performance Table"])

with tab_films:
    all_films_df = compute_top_films(filtered_df, n=0)
    if not all_films_df.empty:
        display_films = all_films_df.rename(
            columns={
                "film_code": "Film Code",
                "total_revenue": "Total Revenue",
                "total_tickets": "Admissions",
                "avg_occupancy": "Avg Occupancy (%)",
                "avg_price": "Avg Ticket Price",
                "n_cinemas": "Cinemas Screened",
                "n_dates": "Active Days",
                "n_screenings": "Total Screenings",
                "revenue_share_pct": "Rev Share (%)",
            }
        )
        st.dataframe(
            display_films.style.format({
                "Total Revenue": "{:,.0f}",
                "Admissions": "{:,.0f}",
                "Avg Occupancy (%)": "{:.1f}%",
                "Avg Ticket Price": "{:,.1f}",
                "Rev Share (%)": "{:.2f}%",
            }),
            use_container_width=True,
            height=360,
        )
    else:
        st.info("No film data available.")

with tab_cinemas:
    all_cinemas_df = compute_top_cinemas(filtered_df, n=0)
    if not all_cinemas_df.empty:
        display_cinemas = all_cinemas_df.rename(
            columns={
                "cinema_code": "Cinema Code",
                "total_revenue": "Total Revenue",
                "total_tickets": "Admissions",
                "avg_occupancy": "Avg Occupancy (%)",
                "avg_capacity": "Avg Capacity",
                "n_films": "Films Screened",
                "n_dates": "Active Days",
                "n_screenings": "Total Screenings",
                "revenue_per_screening": "Rev / Screening",
                "revenue_share_pct": "Rev Share (%)",
            }
        )
        st.dataframe(
            display_cinemas.style.format({
                "Total Revenue": "{:,.0f}",
                "Admissions": "{:,.0f}",
                "Avg Occupancy (%)": "{:.1f}%",
                "Avg Capacity": "{:.1f}",
                "Rev / Screening": "{:,.0f}",
                "Rev Share (%)": "{:.2f}%",
            }),
            use_container_width=True,
            height=360,
        )
    else:
        st.info("No cinema data available.")
