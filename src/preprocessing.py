"""
preprocessing.py
================
Reproducible data-cleaning pipeline for the Cinema Ticket dataset.

Run as a script to generate cleaned outputs:
    python -m src.preprocessing

Design decisions are documented inline. The raw dataset is never modified.

Pipeline overview
-----------------
1. Load raw data and validate schema.
2. Remove exact duplicate rows.
3. Parse dates and drop redundant derived columns (month, quarter, day).
4. Handle missing values in occu_perc / capacity.
5. Flag and handle invalid numeric values (negative capacity, negative
   ticket_use, occupancy > 100 %).
6. Add useful derived time features.
7. Save cleaned record-level dataset  →  cinema_cleaned.csv
8. Build daily aggregate dataset       →  daily_sales.csv

Important: no rows are deleted for having missing values alone.
Negative capacity rows are flagged, not deleted — they represent a specific
cinema (code 557) with a data-entry error that still has valid sales data.
"""

import os
import logging
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data,
    PROCESSED_DATA_DIR,
    CLEANED_CSV,
    DAILY_CSV,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Columns that are perfectly derivable from the 'date' column and therefore
# redundant.  Keeping them would risk inconsistency during feature engineering.
REDUNDANT_DATE_COLS = ["month", "quarter", "day"]

# Physical seat occupancy cannot exceed 100 %.  Values above this threshold
# are flagged; they appear to represent overselling or standing-room admission
# at specific venues (notably cinema_code 414).
OCCU_PERC_HARD_CAP = 100.0


# ===================================================================
# STEP 1 — Remove exact duplicate rows
# ===================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence.

    All 106 duplicates in the raw data originate from film_code 1484 on
    2018-03-24.  They appear to be a bulk data-entry duplication rather than
    genuinely separate transactions, because every column value is identical.
    """
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    log.info(f"Duplicates removed: {n_dropped}  ({n_before} → {len(df)} rows)")
    return df


# ===================================================================
# STEP 2 — Parse dates and drop redundant columns
# ===================================================================
def parse_dates_and_drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'date' to datetime and drop month / quarter / day columns.

    Rationale for dropping month, quarter, day:
    - Inspection confirmed 100 % consistency with the parsed date.
    - We will re-derive richer time features (day_of_week, is_weekend, etc.)
      from the parsed date so the information is not lost.
    """
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        log.warning(f"Unparseable dates dropped: {bad_dates}")
        df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Drop redundant columns
    df = df.drop(columns=REDUNDANT_DATE_COLS, errors="ignore")
    log.info(
        f"Date parsed. Dropped redundant columns: {REDUNDANT_DATE_COLS}. "
        f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ===================================================================
# STEP 3 — Handle missing values
# ===================================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the 125 rows where occu_perc and capacity are NaN.

    Decision: *impute* capacity from the same cinema's other records rather
    than deleting the rows, because the sales / ticket columns are valid.

    Logic:
    - For each cinema_code that has NaN capacity, compute the median capacity
      from that cinema's non-null records.
    - If the cinema has *no* non-null capacity records at all, use the
      dataset-wide median.
    - Then recompute occu_perc = (tickets_sold / capacity) * 100 for the
      imputed rows only.

    This avoids deleting valid sales data and produces a physically meaningful
    occupancy estimate.
    """
    null_mask = df["capacity"].isna()
    n_null = null_mask.sum()

    if n_null == 0:
        log.info("No missing values in capacity / occu_perc.")
        return df

    # Cinema-level median capacity (from non-null rows)
    cinema_median_cap = (
        df.loc[~null_mask]
        .groupby("cinema_code")["capacity"]
        .median()
    )
    global_median_cap = df.loc[~null_mask, "capacity"].median()

    # Impute capacity
    imputed_cap = df.loc[null_mask, "cinema_code"].map(cinema_median_cap)
    imputed_cap = imputed_cap.fillna(global_median_cap)
    df.loc[null_mask, "capacity"] = imputed_cap.values

    # Recompute occu_perc for imputed rows
    df.loc[null_mask, "occu_perc"] = (
        df.loc[null_mask, "tickets_sold"] / df.loc[null_mask, "capacity"] * 100
    )

    log.info(
        f"Missing values imputed: {n_null} rows. "
        f"capacity filled with cinema-level median; occu_perc recomputed."
    )
    return df


# ===================================================================
# STEP 4 — Flag / handle invalid numeric values
# ===================================================================
def handle_invalid_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and correct implausible numeric values.

    Negative capacity (54 rows, all capacity == -2.0):
        - All from cinema_code 557 (and one from 115).
        - Sales and ticket data are valid; only capacity is wrong.
        - Decision: set capacity to NaN and recompute occu_perc as NaN for
          these rows.  We add a boolean flag 'capacity_invalid' so downstream
          analysis can filter them if needed.

    Negative ticket_use (61 rows):
        - ticket_use represents tickets actually scanned / used.
        - Negative values likely represent returns or adjustments.
        - Decision: clip to 0 and add flag 'ticket_use_adjusted'.
        - This does NOT affect tickets_sold or total_sales.

    Occupancy > 100 % (164 rows):
        - Genuine overselling at certain venues (e.g. standing room).
        - Decision: cap at 100 % and add flag 'occu_perc_capped'.
        - The raw tickets_sold and capacity are kept unchanged for
          transparency.
    """
    # --- Negative capacity → NaN + flag ---
    neg_cap_mask = df["capacity"] < 0
    n_neg_cap = neg_cap_mask.sum()
    df["capacity_invalid"] = neg_cap_mask
    if n_neg_cap:
        df.loc[neg_cap_mask, "capacity"] = np.nan
        df.loc[neg_cap_mask, "occu_perc"] = np.nan
        log.info(f"Negative capacity → NaN: {n_neg_cap} rows flagged.")

    # --- Negative ticket_use → clip to 0 + flag ---
    neg_use_mask = df["ticket_use"] < 0
    n_neg_use = neg_use_mask.sum()
    df["ticket_use_adjusted"] = neg_use_mask
    if n_neg_use:
        df.loc[neg_use_mask, "ticket_use"] = 0
        log.info(f"Negative ticket_use clipped to 0: {n_neg_use} rows.")

    # --- occu_perc > 100 % → cap + flag ---
    over_mask = df["occu_perc"] > OCCU_PERC_HARD_CAP
    n_over = over_mask.sum()
    df["occu_perc_capped"] = over_mask
    if n_over:
        df.loc[over_mask, "occu_perc"] = OCCU_PERC_HARD_CAP
        log.info(f"occu_perc capped at {OCCU_PERC_HARD_CAP}%: {n_over} rows.")

    return df


# ===================================================================
# STEP 5 — Add derived time features
# ===================================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the 'date' column.

    New columns:
        year        – year (all 2018 in this dataset, but future-proof)
        month       – month number (1–12)
        quarter     – quarter number (1–4)
        day         – day of month
        day_of_week – 0=Monday … 6=Sunday
        day_name    – 'Monday', 'Tuesday', etc.
        week_number – ISO week number
        is_weekend  – True for Saturday/Sunday
    """
    dt = df["date"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["day"] = dt.day
    df["day_of_week"] = dt.dayofweek
    df["day_name"] = dt.day_name()
    df["week_number"] = dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    log.info(
        f"Time features added: year, month, quarter, day, "
        f"day_of_week, day_name, week_number, is_weekend."
    )
    return df


# ===================================================================
# STEP 6 — Compute no-show count
# ===================================================================
def add_no_show_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'no_shows' column = tickets_sold - ticket_use.

    In the raw data ticket_use ≤ tickets_sold for all rows (after clipping
    negatives to 0).  A non-zero no_shows indicates attendees who bought but
    did not attend.
    """
    df["no_shows"] = df["tickets_sold"] - df["ticket_use"]
    log.info(f"no_shows column added.  Mean no-shows per record: "
             f"{df['no_shows'].mean():.2f}")
    return df


# ===================================================================
# FULL CLEANING PIPELINE
# ===================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw cinema ticket data as loaded by load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned, validated DataFrame ready for EDA and modelling.
    """
    log.info(f"Starting cleaning pipeline.  Raw shape: {df.shape}")

    df = remove_duplicates(df)
    df = parse_dates_and_drop_redundant(df)
    df = handle_missing_values(df)
    df = handle_invalid_numerics(df)
    df = add_time_features(df)
    df = add_no_show_count(df)

    log.info(f"Cleaning complete.  Final shape: {df.shape}")
    return df


# ===================================================================
# DAILY AGGREGATION
# ===================================================================
def build_daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Build a date-level aggregate from cleaned record-level data.

    Each row in the source represents one (film, cinema, date) combination.
    Aggregating by date sums across all films and cinemas for that day.

    Aggregation rules
    -----------------
    - total_sales      : SUM   — total revenue for the day
    - tickets_sold     : SUM   — total tickets sold
    - tickets_out      : SUM   — total complimentary / promo tickets
    - ticket_use       : SUM   — total tickets actually used
    - no_shows         : SUM   — total no-shows
    - show_time        : SUM   — total number of screenings
    - occu_perc        : MEAN  — average occupancy across all shows that day
                                 (NaN rows excluded via default)
    - ticket_price     : MEAN  — average ticket price that day
    - capacity         : SUM   — total seat capacity across all shows
    - n_records        : COUNT — number of (film, cinema) records that day
    - n_films          : NUNIQUE — distinct films showing that day
    - n_cinemas        : NUNIQUE — distinct cinemas operating that day
    - is_weekend       : FIRST — whether the day is a weekend (same for all
                                  records on a given date)
    - day_of_week      : FIRST
    - day_name         : FIRST

    Why SUM for sales/tickets:
        Each source row is a distinct (film, cinema, date) event.
        Summing gives the true daily total without double-counting.
    Why MEAN for occu_perc and ticket_price:
        Occupancy and price vary per show; the mean gives a representative
        daily overview.  Summing would be meaningless.
    """
    agg_spec = {
        "total_sales":  "sum",
        "tickets_sold": "sum",
        "tickets_out":  "sum",
        "ticket_use":   "sum",
        "no_shows":     "sum",
        "show_time":    "sum",
        "occu_perc":    "mean",
        "ticket_price": "mean",
        "capacity":     "sum",
        "film_code":    "nunique",
        "cinema_code":  "nunique",
        "is_weekend":   "first",
        "day_of_week":  "first",
        "day_name":     "first",
    }

    daily = (
        df.groupby("date", as_index=False)
        .agg(agg_spec)
        .rename(columns={
            "film_code":   "n_films",
            "cinema_code": "n_cinemas",
        })
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Add a convenience count column
    daily["n_records"] = (
        df.groupby("date").size().reindex(daily["date"]).values
    )

    # Re-derive calendar columns for the daily table
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["week_number"] = daily["date"].dt.isocalendar().week.astype(int)

    log.info(
        f"Daily aggregate built: {len(daily)} days, "
        f"{daily['date'].min().date()} → {daily['date'].max().date()}"
    )
    return daily


# ===================================================================
# VALIDATION CHECKS
# ===================================================================
def validate_cleaned(df: pd.DataFrame) -> None:
    """Run post-cleaning validation checks.  Raises on failure."""
    assert not df.empty, "Cleaned dataset is empty."
    assert "date" in df.columns, "Missing 'date' column."
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), \
        "'date' is not datetime."
    assert (df["tickets_sold"] >= 0).all(), "Negative tickets_sold found."
    assert (df["total_sales"] >= 0).all(), "Negative total_sales found."
    assert (df["ticket_use"] >= 0).all(), \
        "Negative ticket_use found after clipping."
    assert not df.duplicated().any(), "Duplicates still present."
    log.info("✓ Cleaned dataset passed all validation checks.")


def validate_daily(daily: pd.DataFrame) -> None:
    """Run post-aggregation validation checks.  Raises on failure."""
    assert not daily.empty, "Daily dataset is empty."
    assert pd.api.types.is_datetime64_any_dtype(daily["date"]), \
        "'date' is not datetime."
    assert daily["date"].is_monotonic_increasing, \
        "Daily dataset is not sorted by date."
    assert not daily["date"].duplicated().any(), \
        "Duplicate dates in daily dataset."
    assert (daily["tickets_sold"] >= 0).all(), \
        "Negative daily tickets_sold."
    assert (daily["total_sales"] >= 0).all(), \
        "Negative daily total_sales."
    log.info("✓ Daily aggregate passed all validation checks.")


# ===================================================================
# MAIN — run as: python -m src.preprocessing
# ===================================================================
def main() -> None:
    """Execute the full preprocessing pipeline and save outputs."""
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load raw
    raw = load_raw_data()
    log.info(f"Raw data loaded: {raw.shape[0]} rows × {raw.shape[1]} cols")

    # 2. Clean
    cleaned = clean_data(raw)
    validate_cleaned(cleaned)

    # 3. Save cleaned
    cleaned.to_csv(CLEANED_CSV, index=False)
    log.info(f"Cleaned dataset saved → {CLEANED_CSV}")

    # 4. Build daily aggregate
    daily = build_daily_aggregate(cleaned)
    validate_daily(daily)

    # 5. Save daily
    daily.to_csv(DAILY_CSV, index=False)
    log.info(f"Daily aggregate saved → {DAILY_CSV}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Raw rows:          {raw.shape[0]:>10,}")
    print(f"  Cleaned rows:      {cleaned.shape[0]:>10,}")
    print(f"  Rows removed:      {raw.shape[0] - cleaned.shape[0]:>10,}")
    print(f"  Cleaned columns:   {cleaned.shape[1]:>10}")
    print(f"  Daily rows:        {daily.shape[0]:>10}")
    print(f"  Date range:        {cleaned['date'].min().date()} -> "
          f"{cleaned['date'].max().date()}")
    print(f"  Unique films:      {cleaned['film_code'].nunique():>10}")
    print(f"  Unique cinemas:    {cleaned['cinema_code'].nunique():>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
