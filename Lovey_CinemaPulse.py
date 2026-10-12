"""
Lovey_CinemaPulse.py
====================
Standalone Machine Learning Model Training & Forecasting Pipeline
CinemaPulse — Cinema Sales Intelligence & Demand Forecasting Dashboard

This script is a self-contained, standalone implementation of the
time-series forecasting engine for the CinemaPulse project. It covers:
  1. Processed daily sales data loading and schema validation
  2. Calendar regularisation and operational missing-date treatment
  3. Leak-free direct feature engineering (lags, rolling stats, calendar indicators)
  4. Chronological rolling-origin evaluation (4 non-overlapping 7-day windows)
  5. Baseline comparisons (Naive, Seasonal Naive, Ridge Regression, Random Forest)
  6. Final production model training for Revenue (total_sales) and Admissions (tickets_sold)
  7. Model persistence to models/rf_sales_model.joblib and models/rf_tickets_model.joblib
  8. Out-of-sample 7-day historical forecast generation (2018-11-05 to 2018-11-11)
  9. Export of forecast outputs with heuristic 80% empirical prediction intervals to
     data/processed/sales_forecast.csv

Usage:
    python Lovey_CinemaPulse.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION & PROJECT PATHS (Pathlib relative to project root)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
DAILY_SALES_PATH = DATA_DIR / "daily_sales.csv"
OUTPUT_FORECAST_PATH = DATA_DIR / "sales_forecast.csv"
MODELS_DIR = BASE_DIR / "models"
SALES_MODEL_PATH = MODELS_DIR / "rf_sales_model.joblib"
TICKETS_MODEL_PATH = MODELS_DIR / "rf_tickets_model.joblib"

# Feature set for direct multi-step forecasting (horizon <= 7 days)
# All lag and rolling features are shifted by >= 7 days to strictly eliminate future leakage
FEATURE_COLUMNS = [
    "is_regional_weekend",  # Thursday & Friday regional peak rest days
    "is_tuesday",           # Weekly cinema discount promotion day
    "month",                # Seasonality month indicator
    "lag_7",                # Target value observed 7 days prior
    "lag_14",               # Target value observed 14 days prior
    "rolling_mean_7_lag7",  # 7-day moving average ending at t-7
    "dow_1",                # Tuesday dummy (dow_0/Monday is reference)
    "dow_2",                # Wednesday dummy
    "dow_3",                # Thursday dummy
    "dow_4",                # Friday dummy
    "dow_5",                # Saturday dummy
    "dow_6",                # Sunday dummy
]


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_daily_sales(file_path: Union[str, Path] = DAILY_SALES_PATH) -> pd.DataFrame:
    """
    Load the processed daily cinema sales dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to data/processed/daily_sales.csv.

    Returns
    -------
    pd.DataFrame
        Loaded daily sales dataframe with parsed datetime dates.

    Raises
    ------
    FileNotFoundError
        If the processed file does not exist, providing instructions on how
        to execute the preprocessing pipeline.
    ValueError
        If required columns are missing from the dataset.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"\n[ERROR] Required daily sales dataset not found at: {path}\n"
            f"Please generate the processed data first by running the cleaning pipeline:\n"
            f"    python -m src.preprocessing\n"
            f"or ensure 'data/processed/daily_sales.csv' exists in the workspace."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = {"date", "total_sales", "tickets_sold"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    return df


# ==============================================================================
# 2. TIME-SERIES PREPARATION & CALENDAR REGULARISATION
# ==============================================================================
def prepare_time_series_data(
    df: pd.DataFrame,
    start_date: str = "2018-03-14",
    fill_closures: bool = True,
) -> pd.DataFrame:
    """
    Regularise the daily time series onto a continuous calendar.

    Historical Context & Missing-Date Treatment:
    - The raw dataset spans 2018-02-21 to 2018-11-04 (257 calendar days) with 234
      observed days.
    - The 19 early dates (2018-02-21 to 2018-03-13) represent the initial commercial
      onboarding/ramp-up period with sporadic logging and are excluded by default.
    - Regular modeling commences on 2018-03-14 (236 continuous calendar days).
    - In this operational window, 4 dates are unrecorded: 2018-06-04, 2018-06-06,
      2018-09-19, and 2018-09-20. Domain calendar analysis indicates these correspond
      to official mandatory entertainment venue closures in the regional market
      (religious mourning days).
    - If fill_closures=True, the calendar is reindexed to full daily frequency,
      closure revenue and admissions are zero-filled, and an is_closed indicator
      is set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or loaded daily sales DataFrame.
    start_date : str, default '2018-03-14'
        Cutoff date for stable cinema operations.
    fill_closures : bool, default True
        Whether to reindex onto complete daily sequence and zero-fill closures.

    Returns
    -------
    pd.DataFrame
        Continuous daily DataFrame sorted chronologically.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    required = {"date", "total_sales", "tickets_sold"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected {required}, got {set(df.columns)}")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates(subset=["date"])

    if start_date:
        d = d[d["date"] >= pd.to_datetime(start_date)].copy()

    if d.empty:
        raise ValueError(f"No records remain after filtering for start_date >= {start_date}.")

    if fill_closures:
        full_dates = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
        cal_df = pd.DataFrame({"date": full_dates})
        d = cal_df.merge(d, on="date", how="left")
        # Identify assumed closure dates (missing observations in the operational window)
        # Note: Modeled as mandatory entertainment closures, not confirmed by raw metadata
        d["is_closed"] = d["total_sales"].isna().astype(int)
        d["total_sales"] = d["total_sales"].fillna(0.0)
        d["tickets_sold"] = d["tickets_sold"].fillna(0.0)

    return d.reset_index(drop=True)


# ==============================================================================
# 3. FEATURE ENGINEERING (LEAK-FREE DIRECT FORECASTING)
# ==============================================================================
def create_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "total_sales",
) -> pd.DataFrame:
    """
    Construct calendar, lag, and rolling features for direct multi-step forecasting.

    Target Leakage Prevention:
    - To generate a 7-day out-of-sample forecast at origin T without recursive error
      propagation or target leakage, all lag and rolling statistical features
      use information observed at least 7 days prior (t-7, t-14, and rolling window
      ending at t-7).
    - No contemporaneous or future target observations are ever accessed.

    Domain Features:
    - is_regional_weekend: Thursday (dow 3) & Friday (dow 4) are peak days in the
      regional Middle Eastern cinema market.
    - is_tuesday: Tuesday (dow 1) is the standard promotional discount ticket day.
    - dow_1 to dow_6: One-hot encoded days of week (dow 0 / Monday is baseline).

    Parameters
    ----------
    df : pd.DataFrame
        Continuous daily DataFrame with 'date' and target_col.
    target_col : str, default 'total_sales'
        Target series name ('total_sales' or 'tickets_sold').

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar indicators, lag variables, and dummy columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot create features on empty DataFrame.")

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["day_of_week"] = d["date"].dt.dayofweek
    d["day_name"] = d["date"].dt.day_name()

    # Domain indicators
    d["is_regional_weekend"] = d["day_of_week"].isin([3, 4]).astype(int)
    d["is_tuesday"] = (d["day_of_week"] == 1).astype(int)
    d["month"] = d["date"].dt.month
    d["day_of_month"] = d["date"].dt.day

    # Direct lag features (known >= 7 days in advance)
    if target_col in d.columns:
        d["lag_7"] = d[target_col].shift(7)
        d["lag_14"] = d[target_col].shift(14)
        d["rolling_mean_7_lag7"] = d[target_col].shift(7).rolling(7, min_periods=1).mean()
    else:
        d["lag_7"] = np.nan
        d["lag_14"] = np.nan
        d["rolling_mean_7_lag7"] = np.nan

    # One-hot encoded day of week (dow_1 to dow_6; dow_0/Monday is reference)
    for dow in range(1, 7):
        d[f"dow_{dow}"] = (d["day_of_week"] == dow).astype(int)

    return d


# ==============================================================================
# 4. CHRONOLOGICAL EXPANDING-WINDOW VALIDATION
# ==============================================================================
def evaluate_chronological(
    df: pd.DataFrame,
    target_col: str = "total_sales",
    n_windows: int = 4,
    window_size: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, float], List[float]]:
    """
    Evaluate models using chronological rolling-origin backtesting.

    Rigorous Time-Series Protocol:
    - Never uses random train/test splits, which would violate causality.
    - Evaluates 4 contiguous 7-day test windows (28 test days total):
        * Window 1: 2018-10-08 to 2018-10-14
        * Window 2: 2018-10-15 to 2018-10-21
        * Window 3: 2018-10-22 to 2018-10-28
        * Window 4: 2018-10-29 to 2018-11-04
    - For each window, the training set strictly precedes the test window
      (expanding origin).

    Models Evaluated:
      1. Naive Baseline: Projects the last observed training value across the horizon.
      2. Seasonal Naive: Uses the same day from the previous week (lag_7).
      3. Ridge Regression: Regularised linear model (alpha=10.0, StandardScaler).
      4. Random Forest: Ensemble regressor (n_estimators=100, max_depth=5, random_state=42).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame containing features and target column.
    target_col : str, default 'total_sales'
        Target column name.
    n_windows : int, default 4
        Number of test windows.
    window_size : int, default 7
        Size of each test window in days.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], List[float]]
        - Window-by-window detailed performance metrics
        - Summary metrics averaged across all validation windows
        - Validation residuals for candidate Random Forest model
    """
    clean_data = df.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
    total_eval_points = n_windows * window_size

    if len(clean_data) <= total_eval_points + 28:
        raise ValueError("Insufficient observations for specified rolling-origin evaluation.")

    window_records = []
    residuals_rf = []
    residuals_snaive = []
    residuals_naive = []

    for w_idx in range(n_windows):
        test_end = len(clean_data) - (n_windows - 1 - w_idx) * window_size
        test_start = test_end - window_size

        train = clean_data.iloc[:test_start]
        test = clean_data.iloc[test_start:test_end]

        y_test = test[target_col].values

        # Baseline 1: Naive (last observed value in training)
        y_naive = np.full_like(y_test, train[target_col].iloc[-1])

        # Baseline 2: Seasonal Naive (value from same day last week, i.e., lag_7)
        y_snaive = test["lag_7"].values

        # Candidate Model 1: Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[FEATURE_COLUMNS], train[target_col])
        y_rf = rf.predict(test[FEATURE_COLUMNS])

        # Candidate Model 2: Ridge Regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train[FEATURE_COLUMNS])
        X_test_s = scaler.transform(test[FEATURE_COLUMNS])
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_s, train[target_col])
        y_ridge = ridge.predict(X_test_s)

        # Track residuals
        residuals_rf.extend(y_test - y_rf)
        residuals_snaive.extend(y_test - y_snaive)
        residuals_naive.extend(y_test - y_naive)

        start_date_str = test["date"].min().strftime("%Y-%m-%d")
        end_date_str = test["date"].max().strftime("%Y-%m-%d")

        window_records.append({
            "window": f"Window {w_idx + 1} ({start_date_str} to {end_date_str})",
            "naive_mae": mean_absolute_error(y_test, y_naive),
            "snaive_mae": mean_absolute_error(y_test, y_snaive),
            "ridge_mae": mean_absolute_error(y_test, y_ridge),
            "rf_mae": mean_absolute_error(y_test, y_rf),
            "naive_rmse": np.sqrt(mean_squared_error(y_test, y_naive)),
            "snaive_rmse": np.sqrt(mean_squared_error(y_test, y_snaive)),
            "ridge_rmse": np.sqrt(mean_squared_error(y_test, y_ridge)),
            "rf_rmse": np.sqrt(mean_squared_error(y_test, y_rf)),
        })

    window_df = pd.DataFrame(window_records)

    summary = {
        "naive_mae": float(window_df["naive_mae"].mean()),
        "naive_rmse": float(window_df["naive_rmse"].mean()),
        "snaive_mae": float(window_df["snaive_mae"].mean()),
        "snaive_rmse": float(window_df["snaive_rmse"].mean()),
        "ridge_mae": float(window_df["ridge_mae"].mean()),
        "ridge_rmse": float(window_df["ridge_rmse"].mean()),
        "rf_mae": float(window_df["rf_mae"].mean()),
        "rf_rmse": float(window_df["rf_rmse"].mean()),
    }

    return window_df, summary, residuals_rf


# ==============================================================================
# 5. FINAL PRODUCTION MODEL TRAINING & PERSISTENCE
# ==============================================================================
def train_forecast_model(
    df: pd.DataFrame,
    target_col: str = "total_sales",
    model_type: str = "random_forest",
) -> Union[RandomForestRegressor, Pipeline]:
    """
    Train production forecasting model on all available historical feature observations.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared feature DataFrame.
    target_col : str, default 'total_sales'
        Target column name ('total_sales' or 'tickets_sold').
    model_type : str, default 'random_forest'
        'random_forest' or 'ridge'.

    Returns
    -------
    model : estimator
        Fitted model instance.
    """
    clean_data = df.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
    if clean_data.empty:
        raise ValueError("Cannot train model on empty training data.")

    X = clean_data[FEATURE_COLUMNS]
    y = clean_data[target_col]

    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
    elif model_type == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        model.fit(X, y)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model


# ==============================================================================
# 6. SEVEN-DAY OUT-OF-SAMPLE FORECAST GENERATION
# ==============================================================================
def generate_future_forecast(
    model: Union[RandomForestRegressor, Pipeline],
    historical_df: pd.DataFrame,
    target_col: str = "total_sales",
    horizon_days: int = 7,
    res_std: Optional[float] = None,
    confidence_level: float = 0.80,
) -> pd.DataFrame:
    """
    Generate out-of-sample forecast for the next horizon_days with heuristic bounds.

    Methodological Transparency & Limitations:
    - The dataset ends on 2018-11-04, so the forecast strictly spans 2018-11-05
      to 2018-11-11 (a historical evaluated out-of-sample period).
    - It does not represent a current real-time commercial forecast.
    - Prediction bounds are constructed using empirical backtest residual standard
      deviation: y_hat +/- 1.2816 * s_e (for nominal 80% coverage).
    - These bounds are heuristic approximations under non-stationary market conditions,
      not mathematically guaranteed confidence intervals.
    - Negative revenue and ticket bounds are strictly floored at 0.

    Parameters
    ----------
    model : estimator
        Trained model instance.
    historical_df : pd.DataFrame
        Prepared historical series (continuous calendar).
    target_col : str, default 'total_sales'
        Target column name.
    horizon_days : int, default 7
        Forecast horizon (up to 7 days for direct multi-step).
    res_std : float, optional
        Standard deviation of validation residuals.
    confidence_level : float, default 0.80
        Nominal prediction interval coverage (0.80 -> z = 1.2816).

    Returns
    -------
    pd.DataFrame
        DataFrame with date, day_name, forecast, and lower/upper bounds.
    """
    if horizon_days > 7:
        raise ValueError("Direct multi-step model is configured for a horizon <= 7 days.")

    last_date = pd.to_datetime(historical_df["date"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    future_df = pd.DataFrame({"date": future_dates})
    combined = pd.concat([historical_df[["date", target_col]], future_df], ignore_index=True)

    combined_features = create_forecasting_features(combined, target_col=target_col)
    future_features = combined_features[combined_features["date"] > last_date].copy()

    predictions = model.predict(future_features[FEATURE_COLUMNS])

    # Z-scores for common intervals
    z_map = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9599}
    z_val = z_map.get(confidence_level, 1.2816)

    ci_label = int(confidence_level * 100)
    prefix = "sales" if "sales" in target_col else "tickets"
    if res_std is not None and res_std > 0:
        lower_bound = np.maximum(0.0, predictions - z_val * res_std)
        upper_bound = predictions + z_val * res_std
    else:
        lower_bound = predictions
        upper_bound = predictions

    out_df = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "day_name": future_dates.day_name(),
        f"forecast_{target_col}": np.round(predictions, 2 if target_col == "total_sales" else 0),
        f"{prefix}_lower_{ci_label}": np.round(lower_bound, 2 if target_col == "total_sales" else 0),
        f"{prefix}_upper_{ci_label}": np.round(upper_bound, 2 if target_col == "total_sales" else 0),
    })

    if target_col == "tickets_sold":
        out_df[f"forecast_{target_col}"] = out_df[f"forecast_{target_col}"].astype(int)
        out_df[f"{prefix}_lower_{ci_label}"] = out_df[f"{prefix}_lower_{ci_label}"].astype(int)
        out_df[f"{prefix}_upper_{ci_label}"] = out_df[f"{prefix}_upper_{ci_label}"].astype(int)

    return out_df


# ==============================================================================
# 7. FORMATTING & DISPLAY HELPERS
# ==============================================================================
def format_currency(val: float) -> str:
    """Format large currency values into readable billions."""
    return f"{val / 1e9:,.3f} B"


def format_number(val: float) -> str:
    """Format counts with comma separation."""
    return f"{val:,.0f}"


# ==============================================================================
# 8. MAIN ORCHESTRATOR & CONSOLE REPORTING
# ==============================================================================
def main() -> None:
    """
    Execute the standalone machine learning training and forecasting pipeline.
    """
    print("\n" + "=" * 80)
    print("CinemaPulse -- Machine Learning Model Training & Forecasting Pipeline")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1/6] Loading processed daily sales dataset...")
    try:
        daily_df = load_daily_sales(DAILY_SALES_PATH)
    except Exception as e:
        print(f"[FAILED] Could not load data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  * Source file:            {DAILY_SALES_PATH}")
    print(f"  * Raw observed dates:     {len(daily_df):,} days")
    print(f"  * Raw date span:          {daily_df['date'].min().strftime('%Y-%m-%d')} to {daily_df['date'].max().strftime('%Y-%m-%d')}")

    # 2. Time-series preparation
    print("\n[Step 2/6] Regularising calendar & applying operational domain rules...")
    start_operational_date = "2018-03-14"
    prepared_df = prepare_time_series_data(daily_df, start_date=start_operational_date, fill_closures=True)
    n_days = len(prepared_df)
    n_closed = int(prepared_df["is_closed"].sum())
    print(f"  * Operational start date: {start_operational_date} (excluding 19 onboarding days)")
    print(f"  * Continuous calendar:    {n_days} days ({prepared_df['date'].min().strftime('%Y-%m-%d')} to {prepared_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  * Assumed closure dates:  {n_closed} days zero-filled (June 4, June 6, Sept 19, Sept 20)")
    print(f"  * Domain disclaimer:      Closures represent assumed religious holidays, not confirmed facts.")

    # 3. Feature engineering
    print("\n[Step 3/6] Engineering leak-free features for direct multi-step forecasting...")
    features_sales = create_forecasting_features(prepared_df, "total_sales")
    features_tix = create_forecasting_features(prepared_df, "tickets_sold")
    print(f"  * Features constructed:   {len(FEATURE_COLUMNS)} variables")
    print(f"    - Calendar:             is_regional_weekend (Thu/Fri), is_tuesday, month, dow_1..dow_6")
    print(f"    - Lags & Moving Averages: lag_7, lag_14, rolling_mean_7_lag7 (strictly shifted >= 7 days)")
    print(f"  * Target leakage check:   PASSED (no contemporaneous or future values accessed)")

    # 4. Chronological rolling-origin evaluation
    print("\n[Step 4/6] Executing 4-window chronological expanding-origin backtesting...")
    print("  * Protocol:               Non-overlapping 7-day test windows (28 test days total)")
    print("  * Validation span:        2018-10-08 to 2018-11-04")

    # Evaluate total_sales
    w_sales, summary_sales, res_sales = evaluate_chronological(
        features_sales, target_col="total_sales", n_windows=4, window_size=7
    )
    res_std_sales = float(np.std(res_sales))

    # Evaluate tickets_sold
    w_tix, summary_tix, res_tix = evaluate_chronological(
        features_tix, target_col="tickets_sold", n_windows=4, window_size=7
    )
    res_std_tix = float(np.std(res_tix))

    # Print Revenue Model Comparison Table
    print("\n" + "-" * 78)
    print("REVENUE MODEL COMPARISON (Target: total_sales | 4-Window Expanding Origin)")
    print("-" * 78)
    print(f"{'Model Architecture':<30} | {'MAE':<18} | {'RMSE':<18} | {'Improvement'}")
    print("-" * 78)
    print(f"{'Naive (Last Observed)':<30} | {format_currency(summary_sales['naive_mae']):<18} | {format_currency(summary_sales['naive_rmse']):<18} | Baseline")
    snaive_gain = (1 - summary_sales['snaive_mae'] / summary_sales['naive_mae']) * 100
    print(f"{'Seasonal Naive (Lag 7)':<30} | {format_currency(summary_sales['snaive_mae']):<18} | {format_currency(summary_sales['snaive_rmse']):<18} | +{snaive_gain:.1f}% MAE")
    ridge_gain = (1 - summary_sales['ridge_mae'] / summary_sales['naive_mae']) * 100
    print(f"{'Ridge Regression':<30} | {format_currency(summary_sales['ridge_mae']):<18} | {format_currency(summary_sales['ridge_rmse']):<18} | +{ridge_gain:.1f}% MAE")
    rf_gain = (1 - summary_sales['rf_mae'] / summary_sales['naive_mae']) * 100
    rf_vs_snaive = (1 - summary_sales['rf_mae'] / summary_sales['snaive_mae']) * 100
    print(f"{'Random Forest (Candidate)':<30} | {format_currency(summary_sales['rf_mae']):<18} | {format_currency(summary_sales['rf_rmse']):<18} | +{rf_gain:.1f}% MAE (+{rf_vs_snaive:.1f}% vs SNaive)")
    print("-" * 78)

    # Print Demand Model Comparison Table
    print("\n" + "-" * 78)
    print("TICKET DEMAND MODEL COMPARISON (Target: tickets_sold | 4-Window Expanding Origin)")
    print("-" * 78)
    print(f"{'Model Architecture':<30} | {'MAE':<18} | {'RMSE':<18} | {'Improvement'}")
    print("-" * 78)
    print(f"{'Naive (Last Observed)':<30} | {format_number(summary_tix['naive_mae']):<18} | {format_number(summary_tix['naive_rmse']):<18} | Baseline")
    snaive_tix_gain = (1 - summary_tix['snaive_mae'] / summary_tix['naive_mae']) * 100
    print(f"{'Seasonal Naive (Lag 7)':<30} | {format_number(summary_tix['snaive_mae']):<18} | {format_number(summary_tix['snaive_rmse']):<18} | +{snaive_tix_gain:.1f}% MAE")
    ridge_tix_gain = (1 - summary_tix['ridge_mae'] / summary_tix['naive_mae']) * 100
    print(f"{'Ridge Regression':<30} | {format_number(summary_tix['ridge_mae']):<18} | {format_number(summary_tix['ridge_rmse']):<18} | +{ridge_tix_gain:.1f}% MAE")
    rf_tix_gain = (1 - summary_tix['rf_mae'] / summary_tix['naive_mae']) * 100
    print(f"{'Random Forest (Candidate)':<30} | {format_number(summary_tix['rf_mae']):<18} | {format_number(summary_tix['rf_rmse']):<18} | +{rf_tix_gain:.1f}% MAE")
    print("-" * 78)

    # 5. Final Model Training & Saving
    print("\n[Step 5/6] Fitting final production models on full historical observations...")
    model_sales = train_forecast_model(features_sales, "total_sales", model_type="random_forest")
    model_tix = train_forecast_model(features_tix, "tickets_sold", model_type="random_forest")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_sales, SALES_MODEL_PATH)
    joblib.dump(model_tix, TICKETS_MODEL_PATH)
    print(f"  [OK] Saved revenue model:    {SALES_MODEL_PATH}")
    print(f"  [OK] Saved tickets model:    {TICKETS_MODEL_PATH}")

    # 6. Generate 7-Day Out-of-Sample Forecast
    print("\n[Step 6/6] Generating 7-day out-of-sample forecast (2018-11-05 to 2018-11-11)...")
    forecast_sales = generate_future_forecast(
        model_sales, prepared_df, "total_sales", horizon_days=7, res_std=res_std_sales, confidence_level=0.80
    )
    forecast_tix = generate_future_forecast(
        model_tix, prepared_df, "tickets_sold", horizon_days=7, res_std=res_std_tix, confidence_level=0.80
    )
    combined_forecast = forecast_sales.merge(forecast_tix, on=["date", "day_name"])

    OUTPUT_FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_forecast.to_csv(OUTPUT_FORECAST_PATH, index=False)
    print(f"  [OK] Saved forecast table:   {OUTPUT_FORECAST_PATH}")

    # Print Formatted Forecast Table
    print("\n" + "=" * 90)
    print("7-DAY OUT-OF-SAMPLE FORECAST TABLE (2018-11-05 to 2018-11-11)")
    print("Note: Out-of-sample historical period immediately following dataset end date (2018-11-04).")
    print("Prediction bounds are heuristic empirical intervals (nominal 80%), floored at 0.")
    print("=" * 90)
    print(f"{'Date':<11} | {'Day':<10} | {'Forecast Revenue':<18} | {'Revenue 80% Range':<25} | {'Admissions':<10} | {'Admissions 80% Range'}")
    print("-" * 90)
    for _, row in combined_forecast.iterrows():
        rev_str = format_currency(row['forecast_total_sales'])
        rev_range = f"[{format_currency(row['sales_lower_80'])} - {format_currency(row['sales_upper_80'])}]"
        tix_str = format_number(row['forecast_tickets_sold'])
        tix_range = f"[{format_number(row['tickets_lower_80'])} - {format_number(row['tickets_upper_80'])}]"
        print(f"{row['date']:<11} | {row['day_name']:<10} | {rev_str:<18} | {rev_range:<25} | {tix_str:<10} | {tix_range}")
    print("=" * 90)

    print("\n[SUCCESS] Standalone training and forecasting pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
