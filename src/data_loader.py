"""
data_loader.py
==============
Utility functions for loading raw and processed cinema ticket data.

Paths are derived from the project root (one level above src/).
All loaders include validation checks.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RAW_CSV = os.path.join(RAW_DATA_DIR, "cinemaTicket_Ref.csv")
CLEANED_CSV = os.path.join(PROCESSED_DATA_DIR, "cinema_cleaned.csv")
DAILY_CSV = os.path.join(PROCESSED_DATA_DIR, "daily_sales.csv")
FORECAST_CSV = os.path.join(PROCESSED_DATA_DIR, "sales_forecast.csv")

# ---------------------------------------------------------------------------
# Expected schema for the raw dataset
# ---------------------------------------------------------------------------
EXPECTED_RAW_COLUMNS = [
    "film_code", "cinema_code", "total_sales", "tickets_sold",
    "tickets_out", "show_time", "occu_perc", "ticket_price",
    "ticket_use", "capacity", "date", "month", "quarter", "day",
]


def _validate_columns(df: pd.DataFrame, expected: list[str], label: str) -> None:
    """Raise if the DataFrame is missing any expected columns."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] Missing expected columns: {sorted(missing)}"
        )


def _validate_not_empty(df: pd.DataFrame, label: str) -> None:
    """Raise if the DataFrame is empty."""
    if df.empty:
        raise ValueError(f"[{label}] DataFrame is empty (0 rows).")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Load the original raw CSV without any transformations.

    Returns
    -------
    pd.DataFrame
        Unmodified raw data straight from the Kaggle CSV.

    Raises
    ------
    FileNotFoundError
        If the raw CSV is missing from data/raw/.
    ValueError
        If required columns are absent or the file is empty.
    """
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_CSV}.\n"
            "Please place 'cinemaTicket_Ref.csv' inside data/raw/."
        )

    df = pd.read_csv(RAW_CSV)
    _validate_not_empty(df, "raw")
    _validate_columns(df, EXPECTED_RAW_COLUMNS, "raw")
    return df


def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned record-level dataset produced by preprocessing.

    The 'date' column is parsed to datetime automatically.

    Returns
    -------
    pd.DataFrame
        Cleaned data with validated types.

    Raises
    ------
    FileNotFoundError
        If the cleaned CSV has not been generated yet.
    """
    if not os.path.exists(CLEANED_CSV):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_CSV}.\n"
            "Run the preprocessing pipeline first:\n"
            "  python -m src.preprocessing"
        )

    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    _validate_not_empty(df, "cleaned")
    return df


def load_daily_sales() -> pd.DataFrame:
    """Load the daily aggregate sales dataset.

    Returns
    -------
    pd.DataFrame
        One row per date with aggregated metrics.

    Raises
    ------
    FileNotFoundError
        If the daily CSV has not been generated yet.
    """
    if not os.path.exists(DAILY_CSV):
        raise FileNotFoundError(
            f"Daily sales dataset not found at {DAILY_CSV}.\n"
            "Run the preprocessing pipeline first:\n"
            "  python -m src.preprocessing"
        )

    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    _validate_not_empty(df, "daily_sales")
    return df


def load_forecast_data() -> pd.DataFrame:
    """Load the 7-day historical forecast dataset.

    Returns
    -------
    pd.DataFrame
        Forecast dataset with columns: date, day_name, forecast_total_sales,
        sales_lower_80, sales_upper_80, forecast_tickets_sold, tickets_lower_80, tickets_upper_80.

    Raises
    ------
    FileNotFoundError
        If the forecast CSV has not been generated yet.
    """
    if not os.path.exists(FORECAST_CSV):
        raise FileNotFoundError(
            f"Forecast dataset not found at {FORECAST_CSV}.\n"
            "Run the forecasting pipeline first:\n"
            "  python -m src.forecasting"
        )

    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    _validate_not_empty(df, "sales_forecast")
    return df

