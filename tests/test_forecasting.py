"""
test_forecasting.py
===================
Unit and integration test suite for the forecasting module (src/forecasting.py).
Validates:
  - Data loading and continuous calendar regularisation
  - Missing-date handling and closure zero-filling
  - Leak-free feature construction
  - Chronological rolling-origin evaluation (strictly train < test)
  - Model training (Random Forest & Ridge)
  - 7-day future forecast generation and prediction intervals
  - Edge cases and invalid input validation
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecasting import (
    FEATURE_COLUMNS,
    prepare_time_series_data,
    create_forecasting_features,
    evaluate_chronological,
    train_forecast_model,
    generate_future_forecast,
    run_forecasting_pipeline,
)


@pytest.fixture
def sample_daily_data():
    """Load actual processed daily sales data for testing."""
    path = "data/processed/daily_sales.csv"
    assert os.path.exists(path), f"Processed daily data not found at {path}"
    return pd.read_csv(path, parse_dates=["date"])


def test_prepare_time_series_data_regularisation(sample_daily_data):
    """Test calendar regularisation and closure handling on actual daily sales."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)

    # Must start at 2018-03-14 and end at 2018-11-04
    assert df_prep["date"].min() == pd.to_datetime("2018-03-14")
    assert df_prep["date"].max() == pd.to_datetime("2018-11-04")
    
    # Must be strictly continuous (236 days)
    assert len(df_prep) == 236
    date_diffs = df_prep["date"].diff().dropna()
    assert (date_diffs == pd.Timedelta(days=1)).all(), "Time series must have no date gaps"

    # Known closure dates should have total_sales == 0
    closure_dates = ["2018-06-04", "2018-06-06", "2018-09-19", "2018-09-20"]
    for c_date in closure_dates:
        row = df_prep[df_prep["date"] == pd.to_datetime(c_date)]
        assert len(row) == 1
        assert row["total_sales"].values[0] == 0.0
        assert row["tickets_sold"].values[0] == 0.0


def test_prepare_time_series_invalid_inputs():
    """Test validation on empty, None, or ill-formed input DataFrames."""
    with pytest.raises(ValueError, match="empty or None"):
        prepare_time_series_data(None)

    with pytest.raises(ValueError, match="empty or None"):
        prepare_time_series_data(pd.DataFrame())

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_time_series_data(pd.DataFrame({"date": ["2018-01-01"]}))


def test_create_forecasting_features_leak_free(sample_daily_data):
    """Ensure lag features do not leak target values from t into t-7."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)
    df_feat = create_forecasting_features(df_prep, target_col="total_sales")

    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns, f"Expected feature {col} missing from output"

    # Verify lag_7 definition: row i's lag_7 must equal row i-7's total_sales
    for idx in range(14, 25):
        assert df_feat.loc[idx, "lag_7"] == df_prep.loc[idx - 7, "total_sales"]
        assert df_feat.loc[idx, "lag_14"] == df_prep.loc[idx - 14, "total_sales"]

    # Verify domain calendar flags:
    # Thursday (dow=3) and Friday (dow=4) must have is_regional_weekend == 1
    thu_rows = df_feat[df_feat["day_of_week"] == 3]
    fri_rows = df_feat[df_feat["day_of_week"] == 4]
    mon_rows = df_feat[df_feat["day_of_week"] == 0]
    tue_rows = df_feat[df_feat["day_of_week"] == 1]

    assert (thu_rows["is_regional_weekend"] == 1).all()
    assert (fri_rows["is_regional_weekend"] == 1).all()
    assert (mon_rows["is_regional_weekend"] == 0).all()
    assert (tue_rows["is_tuesday"] == 1).all()
    assert (mon_rows["is_tuesday"] == 0).all()


def test_chronological_rolling_origin_validation(sample_daily_data):
    """Verify that training always strictly precedes testing in backtesting."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)
    df_feat = create_forecasting_features(df_prep, target_col="total_sales")

    windows_df, summary, residuals = evaluate_chronological(
        df_feat, target_col="total_sales", n_windows=4, window_size=7
    )

    assert len(windows_df) == 4
    assert len(residuals) == 28

    # All metrics must be finite and positive
    for col in ["naive_mae", "snaive_mae", "rf_mae", "naive_rmse", "snaive_rmse", "rf_rmse"]:
        assert summary[col] > 0
        assert not np.isnan(summary[col])

    # Seasonal Naive and Random Forest must beat simple Naive by a wide margin
    assert summary["snaive_mae"] < summary["naive_mae"]
    assert summary["rf_mae"] < summary["naive_mae"]


def test_train_forecast_model_and_inference(sample_daily_data):
    """Test model training and numeric inference for Random Forest and Ridge."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)
    df_feat = create_forecasting_features(df_prep, target_col="total_sales")

    rf = train_forecast_model(df_feat, target_col="total_sales", model_type="random_forest")
    ridge = train_forecast_model(df_feat, target_col="total_sales", model_type="ridge")

    assert isinstance(rf, RandomForestRegressor)

    clean_X = df_feat.dropna(subset=FEATURE_COLUMNS)[FEATURE_COLUMNS]
    preds_rf = rf.predict(clean_X)
    preds_ridge = ridge.predict(clean_X)

    assert len(preds_rf) == len(clean_X)
    assert not np.isnan(preds_rf).any()
    assert not np.isnan(preds_ridge).any()


def test_generate_future_forecast_structure(sample_daily_data):
    """Test 7-day future forecast generation and prediction bounds."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)
    df_feat = create_forecasting_features(df_prep, target_col="total_sales")
    rf = train_forecast_model(df_feat, target_col="total_sales", model_type="random_forest")

    fc = generate_future_forecast(
        rf, df_prep, target_col="total_sales", horizon_days=7, res_std=2.5e9, confidence_level=0.80
    )

    assert len(fc) == 7
    # Future dates must strictly start on 2018-11-05 and end on 2018-11-11
    assert fc["date"].iloc[0] == "2018-11-05"
    assert fc["date"].iloc[-1] == "2018-11-11"
    assert not fc["date"].duplicated().any()

    # Predictions and upper bounds must be strictly positive
    assert (fc["forecast_total_sales"] > 0).all()
    assert (fc["sales_upper_80"] >= fc["forecast_total_sales"]).all()
    assert (fc["sales_lower_80"] <= fc["forecast_total_sales"]).all()
    assert (fc["sales_lower_80"] >= 0).all()


def test_forecast_horizon_guard(sample_daily_data):
    """Ensure forecasting horizon > 7 raises ValueError."""
    df_prep = prepare_time_series_data(sample_daily_data, start_date="2018-03-14", fill_closures=True)
    df_feat = create_forecasting_features(df_prep, target_col="total_sales")
    rf = train_forecast_model(df_feat, target_col="total_sales", model_type="random_forest")

    with pytest.raises(ValueError, match="horizon <= 7"):
        generate_future_forecast(rf, df_prep, target_col="total_sales", horizon_days=14)


def test_pipeline_saved_outputs():
    """Verify end-to-end pipeline execution and output artifact persistence."""
    results = run_forecasting_pipeline(save_artifacts=True)

    assert "forecast" in results
    assert "sales_summary" in results
    assert "tickets_summary" in results

    fc = results["forecast"]
    assert len(fc) == 7
    expected_cols = {
        "date",
        "day_name",
        "forecast_total_sales",
        "sales_lower_80",
        "sales_upper_80",
        "forecast_tickets_sold",
        "tickets_lower_80",
        "tickets_upper_80",
    }
    assert expected_cols.issubset(set(fc.columns))

    # Verify files on disk
    assert os.path.exists("data/processed/sales_forecast.csv")
    assert os.path.exists("models/rf_sales_model.joblib")
    assert os.path.exists("models/rf_tickets_model.joblib")
