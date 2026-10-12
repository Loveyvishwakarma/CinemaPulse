"""
analysis.py
===========
Reusable analytical functions and metrics calculations for CinemaPulse.
Supports the Streamlit dashboard and reporting pipelines.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


DOW_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

MONTH_NAMES = {
    2: "February", 3: "March", 4: "April", 5: "May",
    6: "June", 7: "July", 8: "August", 9: "September",
    10: "October", 11: "November"
}


def compute_kpis(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Compute core executive KPIs from transaction-level or aggregated cinema data.

    Calculations:
      - total_revenue       : Sum of total_sales across scope.
      - total_tickets       : Sum of tickets_sold across scope.
      - n_days              : Count of unique calendar dates observed.
      - avg_daily_revenue   : Total revenue divided by observed dates count.
      - weighted_occupancy  : Sum of tickets_sold divided by sum of valid capacity (x 100).
                              Only evaluated on rows with capacity > 0.
      - unweighted_occupancy: Simple arithmetic mean of occu_perc on valid capacity rows.
      - active_films        : Number of unique film_code values.
      - active_cinemas      : Number of unique cinema_code values.
      - total_screenings    : Total record count / showtime instances.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level cinema DataFrame.

    Returns
    -------
    dict
        Dictionary containing calculated metrics and counts.
    """
    if df is None or df.empty:
        return {
            "total_revenue": 0.0,
            "total_tickets": 0,
            "n_days": 0,
            "avg_daily_revenue": 0.0,
            "weighted_occupancy": 0.0,
            "unweighted_occupancy": 0.0,
            "active_films": 0,
            "active_cinemas": 0,
            "total_screenings": 0,
        }

    total_revenue = float(df["total_sales"].sum())
    total_tickets = int(df["tickets_sold"].sum())
    n_days = int(df["date"].nunique())
    avg_daily_revenue = total_revenue / n_days if n_days > 0 else 0.0

    # Capacity & Occupancy
    # Exclude invalid capacity (NaN or <= 0) to avoid division by zero or distortion
    valid_cap = df[df["capacity"] > 0] if "capacity" in df.columns else pd.DataFrame()
    if not valid_cap.empty and valid_cap["capacity"].sum() > 0:
        weighted_occupancy = float(
            (valid_cap["tickets_sold"].sum() / valid_cap["capacity"].sum()) * 100.0
        )
        unweighted_occupancy = float(valid_cap["occu_perc"].mean())
    else:
        weighted_occupancy = float(df["occu_perc"].mean()) if "occu_perc" in df.columns else 0.0
        unweighted_occupancy = weighted_occupancy

    active_films = int(df["film_code"].nunique()) if "film_code" in df.columns else 0
    active_cinemas = int(df["cinema_code"].nunique()) if "cinema_code" in df.columns else 0
    total_screenings = len(df)

    return {
        "total_revenue": total_revenue,
        "total_tickets": total_tickets,
        "n_days": n_days,
        "avg_daily_revenue": avg_daily_revenue,
        "weighted_occupancy": weighted_occupancy,
        "unweighted_occupancy": unweighted_occupancy,
        "active_films": active_films,
        "active_cinemas": active_cinemas,
        "total_screenings": total_screenings,
    }


def compute_top_films(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Aggregate and rank films by total revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level cinema DataFrame.
    n : int, default 10
        Number of top films to return.

    Returns
    -------
    pd.DataFrame
        Ranked film summary table.
    """
    if df is None or df.empty or "film_code" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby("film_code").agg(
        total_revenue=("total_sales", "sum"),
        total_tickets=("tickets_sold", "sum"),
        avg_occupancy=("occu_perc", "mean"),
        avg_price=("ticket_price", "mean"),
        n_cinemas=("cinema_code", "nunique"),
        n_dates=("date", "nunique"),
        n_screenings=("total_sales", "count"),
    ).sort_values("total_revenue", ascending=False).reset_index()

    total_all_revenue = stats["total_revenue"].sum()
    if total_all_revenue > 0:
        stats["revenue_share_pct"] = (stats["total_revenue"] / total_all_revenue) * 100.0
    else:
        stats["revenue_share_pct"] = 0.0

    return stats.head(n) if n else stats


def compute_top_cinemas(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Aggregate and rank cinemas by total revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level cinema DataFrame.
    n : int, default 10
        Number of top cinemas to return.

    Returns
    -------
    pd.DataFrame
        Ranked cinema summary table.
    """
    if df is None or df.empty or "cinema_code" not in df.columns:
        return pd.DataFrame()

    stats = df.groupby("cinema_code").agg(
        total_revenue=("total_sales", "sum"),
        total_tickets=("tickets_sold", "sum"),
        avg_occupancy=("occu_perc", "mean"),
        avg_capacity=("capacity", "mean"),
        n_films=("film_code", "nunique"),
        n_dates=("date", "nunique"),
        n_screenings=("total_sales", "count"),
    ).sort_values("total_revenue", ascending=False).reset_index()

    total_all_revenue = stats["total_revenue"].sum()
    if total_all_revenue > 0:
        stats["revenue_share_pct"] = (stats["total_revenue"] / total_all_revenue) * 100.0
        stats["revenue_per_screening"] = stats["total_revenue"] / stats["n_screenings"]
    else:
        stats["revenue_share_pct"] = 0.0
        stats["revenue_per_screening"] = 0.0

    return stats.head(n) if n else stats


def compute_dow_patterns(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales by day of week.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily aggregate sales DataFrame.

    Returns
    -------
    pd.DataFrame
        Indexed by standard Monday-Sunday ordering.
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    d = df_daily.copy()
    if "day_name" not in d.columns:
        d["day_name"] = pd.to_datetime(d["date"]).dt.day_name()

    agg = d.groupby("day_name").agg(
        avg_sales=("total_sales", "mean"),
        avg_tickets=("tickets_sold", "mean"),
        avg_occupancy=("occu_perc", "mean"),
        total_sales=("total_sales", "sum"),
        n_days=("date", "count"),
    ).reindex(DOW_ORDER).reset_index()

    return agg


def compute_monthly_patterns(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales by calendar month.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily aggregate sales DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary table by month.
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()

    d = df_daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month

    agg = d.groupby("month").agg(
        total_sales=("total_sales", "sum"),
        total_tickets=("tickets_sold", "sum"),
        avg_daily_sales=("total_sales", "mean"),
        avg_occupancy=("occu_perc", "mean"),
        n_days=("date", "count"),
    ).reset_index()

    agg["month_name"] = agg["month"].map(MONTH_NAMES).fillna(agg["month"].astype(str))
    return agg.sort_values("month").reset_index(drop=True)
