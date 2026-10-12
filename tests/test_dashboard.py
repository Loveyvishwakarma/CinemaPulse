"""
test_dashboard.py
=================
Test suite for Streamlit dashboard modules, data loaders, analysis helpers,
and insights engine.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_cleaned_data,
    load_daily_sales,
    load_forecast_data,
)
from src.analysis import (
    compute_kpis,
    compute_top_films,
    compute_top_cinemas,
    compute_dow_patterns,
    compute_monthly_patterns,
)
from src.insights import (
    generate_risk_indicators,
    generate_recommendations,
)
from src.ui_helpers import (
    format_revenue,
    format_tickets,
    format_percentage,
)


@pytest.fixture
def cleaned_df():
    return load_cleaned_data()


@pytest.fixture
def daily_df():
    return load_daily_sales()


@pytest.fixture
def forecast_df():
    return load_forecast_data()


def test_data_loaders(cleaned_df, daily_df, forecast_df):
    """Verify all datasets load properly with non-empty rows and expected schema."""
    assert not cleaned_df.empty
    assert len(cleaned_df) == 142418
    assert "date" in cleaned_df.columns
    assert "total_sales" in cleaned_df.columns

    assert not daily_df.empty
    assert len(daily_df) == 234
    assert "total_sales" in daily_df.columns

    assert not forecast_df.empty
    assert len(forecast_df) == 7
    assert forecast_df["date"].iloc[0].strftime("%Y-%m-%d") == "2018-11-05"
    assert forecast_df["date"].iloc[-1].strftime("%Y-%m-%d") == "2018-11-11"


def test_compute_kpis_accuracy(cleaned_df):
    """Verify compute_kpis calculates metrics correctly and handles weighted occupancy."""
    kpis = compute_kpis(cleaned_df)

    assert kpis["total_revenue"] > 0
    assert kpis["total_tickets"] > 0
    assert kpis["n_days"] == 234
    assert kpis["avg_daily_revenue"] == kpis["total_revenue"] / 234

    # Weighted occupancy must be lower than unweighted mean due to capacity skew
    assert 15.0 < kpis["weighted_occupancy"] < 18.0
    assert 18.0 < kpis["unweighted_occupancy"] < 22.0
    assert kpis["weighted_occupancy"] < kpis["unweighted_occupancy"]


def test_compute_kpis_empty():
    """Ensure compute_kpis handles empty DataFrame gracefully."""
    kpis = compute_kpis(pd.DataFrame())
    assert kpis["total_revenue"] == 0.0
    assert kpis["total_tickets"] == 0
    assert kpis["weighted_occupancy"] == 0.0


def test_top_films_and_cinemas(cleaned_df):
    """Test top films and top cinemas aggregation."""
    top_films = compute_top_films(cleaned_df, n=10)
    assert len(top_films) == 10
    assert top_films["total_revenue"].is_monotonic_decreasing
    # Top 1 film should be film_code 1554
    assert top_films["film_code"].iloc[0] == 1554

    top_cinemas = compute_top_cinemas(cleaned_df, n=10)
    assert len(top_cinemas) == 10
    assert top_cinemas["total_revenue"].is_monotonic_decreasing


def test_dow_and_monthly_patterns(daily_df):
    """Test day-of-week and monthly patterns."""
    dow = compute_dow_patterns(daily_df)
    assert len(dow) == 7
    # Friday should be highest average sales
    friday_row = dow[dow["day_name"] == "Friday"]
    monday_row = dow[dow["day_name"] == "Monday"]
    assert friday_row["avg_sales"].values[0] > monday_row["avg_sales"].values[0]

    monthly = compute_monthly_patterns(daily_df)
    assert len(monthly) == 10 # Feb to Nov


def test_risk_indicators_and_recommendations(cleaned_df):
    """Test risk engine and recommendation matrix."""
    risks = generate_risk_indicators(cleaned_df, low_occu_threshold=10.0)
    assert 45.0 < risks["low_occupancy_share_pct"] < 55.0
    assert 50.0 < risks["top5_revenue_concentration_pct"] < 60.0
    assert 0.10 < risks["noshow_rate_pct"] < 0.25

    recs = generate_recommendations()
    assert len(recs) >= 5
    assert set(recs.columns) == {"Finding", "Category", "Supporting Evidence", "Recommended Action"}


def test_formatters():
    """Test number formatting helpers."""
    assert format_revenue(1.5e9) == "1.50B"
    assert format_revenue(2.5e6) == "2.50M"
    assert format_revenue(5000) == "5.0K"
    assert format_tickets(1.2e6) == "1.20M"
    assert format_tickets(45000) == "45.0K"
    assert format_percentage(16.39) == "16.4%"
    assert format_revenue(None) == "N/A"


def test_data_consistency_across_aggregates(cleaned_df, daily_df):
    """Verify that transaction-level and daily aggregates sum to identical totals."""
    assert cleaned_df["total_sales"].sum() == daily_df["total_sales"].sum()
    assert cleaned_df["tickets_sold"].sum() == daily_df["tickets_sold"].sum()
    assert cleaned_df["date"].nunique() == len(daily_df)


def test_raw_dataset_integrity():
    """Verify that the raw CSV in data/raw/ matches original shape and remains untouched."""
    import hashlib
    raw_path = "data/raw/cinemaTicket_Ref.csv"
    assert os.path.exists(raw_path)
    raw = pd.read_csv(raw_path)
    assert raw.shape == (142524, 14)

    # Check sha256 against root original
    if os.path.exists("cinemaTicket_Ref.csv"):
        with open(raw_path, "rb") as f1, open("cinemaTicket_Ref.csv", "rb") as f2:
            assert hashlib.sha256(f1.read()).hexdigest() == hashlib.sha256(f2.read()).hexdigest()


def test_forecast_dates_historical(forecast_df, daily_df):
    """Verify forecast covers strictly 2018-11-05 to 2018-11-11 without historical overlap."""
    hist_max = daily_df["date"].max()
    fc_min = forecast_df["date"].min()
    fc_max = forecast_df["date"].max()

    assert fc_min == pd.to_datetime("2018-11-05")
    assert fc_max == pd.to_datetime("2018-11-11")
    assert fc_min > hist_max, "Forecast must strictly follow the historical timeline"
    assert len(forecast_df) == 7


def test_recommendations_no_causation_claims():
    """Audit recommendations text to ensure absence of unwarranted causation claims."""
    recs = generate_recommendations()
    for _, row in recs.iterrows():
        text = (row["Finding"] + " " + row["Supporting Evidence"]).lower()
        assert "proves causation" not in text
        assert "proven fact" not in text


def test_streamlit_apptest_pages():
    """Execute Streamlit headless AppTest across all application pages."""
    from streamlit.testing.v1 import AppTest

    for file_path in [
        "app.py",
        "pages/1_Executive_Overview.py",
        "pages/2_Sales_Performance.py",
        "pages/3_Forecast_Risk_Decisions.py",
    ]:
        at = AppTest.from_file(file_path).run()
        assert not at.exception, f"Streamlit AppTest raised exception on {file_path}: {at.exception}"

