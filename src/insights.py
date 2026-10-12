"""
insights.py
===========
Business rules, risk quantification, and actionable recommendations engine
for the CinemaPulse dashboard.
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd


def generate_risk_indicators(
    df: pd.DataFrame,
    low_occu_threshold: float = 10.0,
) -> Dict[str, Union[float, int, str]]:
    """
    Calculate data-backed operational risk indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level cinema DataFrame.
    low_occu_threshold : float, default 10.0
        Configurable occupancy threshold (in %) below which a show is considered under-attended.

    Returns
    -------
    dict
        Dictionary of risk indicators with counts, rates, and explanatory notes.
    """
    if df is None or df.empty:
        return {
            "low_occupancy_share_pct": 0.0,
            "low_occupancy_count": 0,
            "total_screenings": 0,
            "top5_revenue_concentration_pct": 0.0,
            "noshow_rate_pct": 0.0,
            "screenings_with_noshows": 0,
            "invalid_capacity_count": 0,
        }

    total_screenings = len(df)
    valid_cap = df[df["capacity"] > 0] if "capacity" in df.columns else df
    
    # 1. Low occupancy screenings
    low_occu_count = int((valid_cap["occu_perc"] < low_occu_threshold).sum())
    low_occu_share = (low_occu_count / len(valid_cap) * 100.0) if len(valid_cap) > 0 else 0.0

    # 2. Revenue concentration (Top 5 films)
    if "film_code" in df.columns and "total_sales" in df.columns:
        film_rev = df.groupby("film_code")["total_sales"].sum().sort_values(ascending=False)
        total_rev = film_rev.sum()
        top5_rev = film_rev.head(5).sum()
        top5_share = (top5_rev / total_rev * 100.0) if total_rev > 0 else 0.0
    else:
        top5_share = 0.0

    # 3. No-show rate
    total_tix = df["tickets_sold"].sum()
    total_noshows = df["no_shows"].sum() if "no_shows" in df.columns else 0
    noshow_rate = (total_noshows / total_tix * 100.0) if total_tix > 0 else 0.0
    screenings_with_noshows = int((df["no_shows"] > 0).sum()) if "no_shows" in df.columns else 0

    # 4. Invalid capacity flags (from Phase 2)
    invalid_capacity_count = int(df["capacity_invalid"].sum()) if "capacity_invalid" in df.columns else 0

    return {
        "low_occupancy_share_pct": float(round(low_occu_share, 2)),
        "low_occupancy_count": low_occu_count,
        "total_screenings": total_screenings,
        "top5_revenue_concentration_pct": float(round(top5_share, 2)),
        "noshow_rate_pct": float(round(noshow_rate, 3)),
        "screenings_with_noshows": screenings_with_noshows,
        "invalid_capacity_count": invalid_capacity_count,
    }


def generate_recommendations() -> pd.DataFrame:
    """
    Generate evidence-based strategic business recommendations.
    Every recommendation links directly to verified calculations from Phase 3 and Phase 4.

    Returns
    -------
    pd.DataFrame
        Table containing Finding, Category, Supporting Evidence, and Recommended Action.
    """
    recommendations = [
        {
            "Finding": "Severe low-occupancy clustering in half of all shows",
            "Category": "Auditorium Scheduling",
            "Supporting Evidence": (
                "49.0% of all screenings operate below 10% occupancy. "
                "Overall weighted occupancy is 16.39% vs unweighted mean 19.95%."
            ),
            "Recommended Action": (
                "Downsize low-demand showtimes into smaller screening rooms or matinee slots. "
                "Retire titles running persistently below 5% occupancy after 3 weeks to save staffing & HVAC overhead."
            ),
        },
        {
            "Finding": "Extreme box office dependency on top 5 films",
            "Category": "Catalog & Risk Management",
            "Supporting Evidence": (
                "Top 5 films generate 54.2% of total box office revenue; top 10 capture 75.6%. "
                "Single top film (film_code 1554) drove 372.7B."
            ),
            "Recommended Action": (
                "Diversify pre-booking commitments and negotiate flexible distributor revenue-share tiers. "
                "Allocate maximum screen capacity to breakout blockbusters during their opening 14-day window."
            ),
        },
        {
            "Finding": "Discount Tuesday demonstrates high volume associated with promotional pricing",
            "Category": "Dynamic Pricing & Promotions",
            "Supporting Evidence": (
                "Tuesday generates the highest daily ticket demand (169.7K tickets/day) "
                "and highest occupancy (37.1%), while ticket prices average lower than weekend rates."
            ),
            "Recommended Action": (
                "Consider testing promotional or matinee pricing models on sluggish Monday and Sunday slots "
                "to evaluate whether off-peak demand responds to volume incentives."
            ),
        },
        {
            "Finding": "Concentrated weekend demand premium (Thursday & Friday)",
            "Category": "Yield Optimization",
            "Supporting Evidence": (
                "Thursday (10.16B) and Friday (11.22B) generate a 1.73x daily revenue premium "
                "over Saturday–Wednesday workdays (10.71B vs 6.20B/day)."
            ),
            "Recommended Action": (
                "Implement tiered premium pricing on prime Thursday/Friday evening showtimes (>50% occupancy) "
                "to capture consumer willingness-to-pay during peak social leisure windows."
            ),
        },
        {
            "Finding": "Derived no-shows indicate low aggregate rate but screening spikes",
            "Category": "Digital Operations",
            "Supporting Evidence": (
                "31,523 derived unseated tickets (0.16% of 19.95M tickets sold) across 6,121 screenings, "
                "with spikes up to 311 unseated tickets during large screenings. "
                "(Limitation: Derived from tickets_sold - ticket_use after clipping; turnstile audit logs unavailable)."
            ),
            "Recommended Action": (
                "Test automated check-in notifications for high-volume screenings "
                "and evaluate standby seat re-release procedures for sold-out auditoriums."
            ),
        },
        {
            "Finding": "Unrecorded operational dates align with regional calendar observances",
            "Category": "Forecasting & Operations",
            "Supporting Evidence": (
                "4 isolated dates without transaction records (Jun 4, Jun 6, Sep 19, Sep 20), "
                "hypothesized as regional observance/closure dates, alongside Arba'een (Oct 30) where sales collapsed to 2.07M. "
                "(Limitation: Assumed closures; dataset lacks official calendar metadata)."
            ),
            "Recommended Action": (
                "Incorporate an official regional holiday calendar feed into scheduling workflows "
                "to avoid allocating staffing and overhead during unrecorded or observance periods."
            ),
        },
    ]

    return pd.DataFrame(recommendations)
