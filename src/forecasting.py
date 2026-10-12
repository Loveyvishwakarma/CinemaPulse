"""
forecasting.py
==============
Demand and sales forecasting module for CinemaPulse.

Provides:
  - Calendar-regularised time series preparation and missing date treatment
  - Leak-free direct feature construction (lags, rolling stats, calendar features)
  - Chronological rolling-origin evaluation against baseline methods (Naive, Seasonal Naive)
  - Model training (Random Forest, Ridge Regression) for primary (total_sales)
    and secondary (tickets_sold) targets
  - 7-day future forecast generation with residual-based empirical prediction intervals
  - Pipeline orchestrator for automated dashboard consumption
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "is_regional_weekend",
    "is_tuesday",
    "month",
    "lag_7",
    "lag_14",
    "rolling_mean_7_lag7",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
]


def prepare_time_series_data(
    df: pd.DataFrame,
    start_date: str = "2018-03-14",
    fill_closures: bool = True,
) -> pd.DataFrame:
    """
    Prepare and regularise the daily sales time series.

    Parameters
    ----------
    df : pd.DataFrame
        Daily sales DataFrame containing at least 'date', 'total_sales', and 'tickets_sold'.
    start_date : str, default '2018-03-14'
        Starting date for regular cinema operations (filtering out the initial ramp-up).
    fill_closures : bool, default True
        Whether to reindex to a complete daily calendar and fill closure days with 0.

    Returns
    -------
    pd.DataFrame
        Continuous daily time series DataFrame sorted chronologically.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    required = {"date", "total_sales", "tickets_sold"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected {required}, got {set(df.columns)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)].copy()

    if df.empty:
        raise ValueError(f"No records remain after filtering for start_date >= {start_date}.")

    if fill_closures:
        full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        cal_df = pd.DataFrame({"date": full_dates})
        df = cal_df.merge(df, on="date", how="left")
        df["total_sales"] = df["total_sales"].fillna(0.0)
        df["tickets_sold"] = df["tickets_sold"].fillna(0.0)
        df["is_closed"] = df["total_sales"].isna().astype(int)

    return df.reset_index(drop=True)


def create_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "total_sales",
) -> pd.DataFrame:
    """
    Construct calendar and lag features for direct multi-step forecasting (horizon <= 7 days).

    To prevent data leakage in multi-step forecasting, all lag and rolling features
    use information observed at least 7 days prior (t - 7, t - 14, and rolling window ending at t - 7).
    This guarantees that a 7-day forecast made at time T uses strictly historical data.

    Parameters
    ----------
    df : pd.DataFrame
        Continuous daily DataFrame.
    target_col : str, default 'total_sales'
        Target series name.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame containing calendar indicators, lag variables, and dummy columns.
    """
    if df is None or df.empty:
        raise ValueError("Cannot create features on empty DataFrame.")

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["day_of_week"] = d["date"].dt.dayofweek
    d["day_name"] = d["date"].dt.day_name()

    # Domain indicators:
    # Middle Eastern workweek: Thursday (3) & Friday (4) are peak days
    d["is_regional_weekend"] = d["day_of_week"].isin([3, 4]).astype(int)
    # Promotional discount day: Tuesday (1)
    d["is_tuesday"] = (d["day_of_week"] == 1).astype(int)
    d["month"] = d["date"].dt.month
    d["day_of_month"] = d["date"].dt.day

    # Direct lag features (known 7 days in advance)
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


def evaluate_chronological(
    df: pd.DataFrame,
    target_col: str = "total_sales",
    n_windows: int = 4,
    window_size: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, float], List[float]]:
    """
    Perform rolling-origin backtesting over non-overlapping chronological test windows.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with features and target.
    target_col : str
        Target column name ('total_sales' or 'tickets_sold').
    n_windows : int, default 4
        Number of 7-day test windows (default 4 = 28 days backtesting).
    window_size : int, default 7
        Forecast horizon size in days.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], List[float]]
        - Window-by-window metrics table
        - Summary metrics across all windows
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

        # Candidate Model 1: Random Forest
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

        # Record residuals
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


def train_forecast_model(
    df: pd.DataFrame,
    target_col: str = "total_sales",
    model_type: str = "random_forest",
) -> Union[RandomForestRegressor, Ridge]:
    """
    Train a forecasting model on all available feature data.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared feature DataFrame.
    target_col : str
        Target column name.
    model_type : str, default 'random_forest'
        'random_forest' or 'ridge'.

    Returns
    -------
    model
        Trained model instance.
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
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
        model.fit(X, y)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model


def generate_future_forecast(
    model: Union[RandomForestRegressor, Ridge],
    historical_df: pd.DataFrame,
    target_col: str = "total_sales",
    horizon_days: int = 7,
    res_std: Optional[float] = None,
    confidence_level: float = 0.80,
) -> pd.DataFrame:
    """
    Generate an out-of-sample forecast for the next horizon_days.

    Parameters
    ----------
    model : trained estimator
        Fitted model object.
    historical_df : pd.DataFrame
        Prepared historical series (continuous calendar).
    target_col : str, default 'total_sales'
        Target column name.
    horizon_days : int, default 7
        Forecast horizon (up to 7 days for direct multi-step).
    res_std : float, optional
        Standard deviation of validation residuals.
    confidence_level : float, default 0.80
        Nominal prediction interval coverage (e.g. 0.80 -> z = 1.28).

    Returns
    -------
    pd.DataFrame
        DataFrame containing forecast dates, day names, predictions, and prediction intervals.
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


def run_forecasting_pipeline(
    daily_sales_path: str = "data/processed/daily_sales.csv",
    output_forecast_path: str = "data/processed/sales_forecast.csv",
    models_dir: str = "models",
    save_artifacts: bool = True,
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Execute the complete forecasting pipeline for revenue and ticket demand.

    Returns
    -------
    dict
        Dictionary containing forecast dataframe, evaluation summaries, and models.
    """
    daily_df = pd.read_csv(daily_sales_path, parse_dates=["date"])
    prepared_df = prepare_time_series_data(daily_df, start_date="2018-03-14", fill_closures=True)

    # 1. Evaluate total_sales
    features_sales = create_forecasting_features(prepared_df, "total_sales")
    w_sales, summary_sales, res_sales = evaluate_chronological(
        features_sales, target_col="total_sales", n_windows=4, window_size=7
    )
    res_std_sales = float(np.std(res_sales))

    # 2. Evaluate tickets_sold
    features_tix = create_forecasting_features(prepared_df, "tickets_sold")
    w_tix, summary_tix, res_tix = evaluate_chronological(
        features_tix, target_col="tickets_sold", n_windows=4, window_size=7
    )
    res_std_tix = float(np.std(res_tix))

    # 3. Fit final production models
    model_sales = train_forecast_model(features_sales, "total_sales", model_type="random_forest")
    model_tix = train_forecast_model(features_tix, "tickets_sold", model_type="random_forest")

    # 4. Generate 7-day future forecasts
    forecast_sales = generate_future_forecast(
        model_sales, prepared_df, "total_sales", horizon_days=7, res_std=res_std_sales, confidence_level=0.80
    )
    forecast_tix = generate_future_forecast(
        model_tix, prepared_df, "tickets_sold", horizon_days=7, res_std=res_std_tix, confidence_level=0.80
    )

    combined_forecast = forecast_sales.merge(forecast_tix, on=["date", "day_name"])

    # 5. Persist artifacts
    if save_artifacts:
        os.makedirs(os.path.dirname(output_forecast_path), exist_ok=True)
        combined_forecast.to_csv(output_forecast_path, index=False)

        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model_sales, os.path.join(models_dir, "rf_sales_model.joblib"))
        joblib.dump(model_tix, os.path.join(models_dir, "rf_tickets_model.joblib"))

    return {
        "forecast": combined_forecast,
        "sales_summary": summary_sales,
        "tickets_summary": summary_tix,
        "sales_windows": w_sales,
        "tickets_windows": w_tix,
        "model_sales": model_sales,
        "model_tickets": model_tix,
    }


if __name__ == "__main__":
    results = run_forecasting_pipeline()
    print("=== Pipeline Complete ===")
    print("\nRevenue Summary (4-window rolling origin):")
    for k, v in results["sales_summary"].items():
        print(f"  {k:<15}: {v/1e9:.3f} B")
    print("\nTicket Summary (4-window rolling origin):")
    for k, v in results["tickets_summary"].items():
        print(f"  {k:<15}: {v:,.0f} tickets")
    print("\n7-Day Future Forecast (2018-11-05 to 2018-11-11):")
    print(results["forecast"].to_string(index=False))
