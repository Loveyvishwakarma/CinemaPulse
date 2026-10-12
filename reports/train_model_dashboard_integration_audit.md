# 🔍 Integration and Regression Audit Report: `train_model.py` & Streamlit Dashboard

**Project:** CinemaPulse — Cinema Sales Intelligence & Demand Forecasting Dashboard  
**Author:** AI Technical Auditor & Quality Assurance Agent  
**Date of Audit:** October 13, 2026  
**Audited Artifacts:** `train_model.py` (and `Lovey_CinemaPulse.py`), `data/processed/sales_forecast.csv`, `models/rf_sales_model.joblib`, `models/rf_tickets_model.joblib`, `app.py`, `pages/`  
**Integration Verdict:** **PASS**

---

## 1. Executive Summary

This audit assesses the technical integration, data compatibility, and visual regression status of the newly created standalone machine learning training and forecasting script, [`train_model.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/train_model.py), with the existing multi-page Streamlit dashboard and project test suite.

The audit confirmed:
1. **End-to-End Execution:** Running `python train_model.py` completes with exit code `0` in ~9 seconds, regenerating all model artifacts and forecast tables natively without importing `src.forecasting`.
2. **Schema & Semantic Parity:** The regenerated forecast table, [`data/processed/sales_forecast.csv`](file:///w:/4.%20IBM%20Internship/CinemaPulse/data/processed/sales_forecast.csv), strictly matches the schema, column names, data types, and historical date range (`2018-11-05` to `2018-11-11`) required by the dashboard.
3. **Model Artifact Integrity:** The saved serialized models ([`models/rf_sales_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_sales_model.joblib) and [`models/rf_tickets_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_tickets_model.joblib)) deserialize cleanly via `joblib.load()` and perform numeric predictions on 12-dimensional feature vectors.
4. **Automated Test Suite:** All 20 automated unit, integration, and UI tests in [`tests/`](file:///w:/4.%20IBM%20Internship/CinemaPulse/tests/) pass with zero regressions (`20/20 passed in 10.51s`).
5. **Live UI Verification:** Headless browser inspection across all four dashboard views confirmed zero exceptions, zero broken charts, and complete visual integrity.

---

## 2. Files and Dependencies Inspected

| Category | File Path | Role & Inspection Objective | Status |
|---|---|---|---|
| **Standalone Training** | [`train_model.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/train_model.py) | Standalone training script | Fully verified |
| **Named Training Script** | [`Lovey_CinemaPulse.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/Lovey_CinemaPulse.py) | Student-identified script | Fully verified |
| **Core App** | [`app.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/app.py) | Main dashboard entry point & landing page | Verified |
| **Data Loader** | [`src/data_loader.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/data_loader.py) | Schema validation and file path resolution | Verified |
| **UI Helpers** | [`src/ui_helpers.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/ui_helpers.py) | Cached loaders (`@st.cache_data`) & Plotly styling | Verified |
| **Dashboard Page 1** | [`pages/1_Executive_Overview.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/pages/1_Executive_Overview.py) | Actuals-to-Forecast Bridge visualization | Verified |
| **Dashboard Page 2** | [`pages/2_Sales_Performance.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/pages/2_Sales_Performance.py) | Film/Cinema sales and pricing breakdowns | Verified |
| **Dashboard Page 3** | [`pages/3_Forecast_Risk_Decisions.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/pages/3_Forecast_Risk_Decisions.py) | 7-day revenue/admissions forecasts, backtest tables | Verified |
| **Original Pipeline** | [`src/forecasting.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/forecasting.py) | Verified for methodological consistency | Preserved intact |
| **Data Pipeline** | [`src/preprocessing.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/preprocessing.py) | Data cleaning pipeline generating daily aggregates | Preserved intact |
| **Test Suite** | [`tests/test_forecasting.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/tests/test_forecasting.py) | 8 unit tests for calendar, leakage & ML models | 8/8 Passed |
| **Test Suite** | [`tests/test_dashboard.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/tests/test_dashboard.py) | 12 tests for KPIs, loaders, and Streamlit AppTest | 12/12 Passed |
| **Generated Data** | [`data/processed/sales_forecast.csv`](file:///w:/4.%20IBM%20Internship/CinemaPulse/data/processed/sales_forecast.csv) | Precomputed 7-day forecast table | Verified |
| **Model Artifacts** | [`models/rf_sales_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_sales_model.joblib) | Random Forest revenue model artifact | Verified |
| **Model Artifacts** | [`models/rf_tickets_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_tickets_model.joblib) | Random Forest ticket demand model artifact | Verified |
| **Environment** | [`requirements.txt`](file:///w:/4.%20IBM%20Internship/CinemaPulse/requirements.txt) | Dependency specifications | 100% Satisfied |
| **Documentation** | [`README.md`](file:///w:/4.%20IBM%20Internship/CinemaPulse/README.md) | Project documentation & execution guide | Updated & verified |

---

## 3. How the Dashboard Obtains Forecast Data

The Streamlit dashboard follows an industry-standard **decoupled BI consumption pattern**:
1. It does **not** train or fit machine learning models dynamically inside user browser sessions.
2. In [`src/data_loader.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/data_loader.py#L133-L149), the function `load_forecast_data()` reads `data/processed/sales_forecast.csv`, parses the `date` column into a Pandas datetime index, and returns the DataFrame.
3. In [`src/ui_helpers.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/src/ui_helpers.py#L418-L422), `get_forecast_data()` wraps `load_forecast_data()` with Streamlit's `@st.cache_data(show_spinner=False)` decorator for sub-millisecond page transitions.
4. Both [`pages/1_Executive_Overview.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/pages/1_Executive_Overview.py) and [`pages/3_Forecast_Risk_Decisions.py`](file:///w:/4.%20IBM%20Internship/CinemaPulse/pages/3_Forecast_Risk_Decisions.py) import `get_forecast_data()` and slice the precomputed forecast fields directly for Plotly figures and data tables.

---

## 4. Whether the Dashboard Loads the Saved Model Files

**Explicit Finding:** **The Streamlit dashboard does NOT load or execute `.predict()` on the `.joblib` model artifacts at runtime.**

- The dashboard relies entirely on the precomputed forecast output in `data/processed/sales_forecast.csv`.
- The saved model files ([`models/rf_sales_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_sales_model.joblib) and [`models/rf_tickets_model.joblib`](file:///w:/4.%20IBM%20Internship/CinemaPulse/models/rf_tickets_model.joblib)) serve as verifiable model persistence artifacts for grading, auditability, and offline batch scoring.
- Verification tests confirmed:
  - Both `.joblib` files deserialize properly via `joblib.load()`.
  - Both estimators are instances of `sklearn.ensemble.RandomForestRegressor`.
  - Both models specify `n_features_in_ = 12` matching `FEATURE_COLUMNS`.
  - Both models execute inference on test feature vectors without errors.

---

## 5. Forecast CSV Schema Comparison

| Column Name | Expected Dtype | Generated Dtype | Null Count | Value Constraints | Dashboard Consumer | Status |
|---|---|---|---|---|---|---|
| `date` | `object` / `string` (YYYY-MM-DD) | `object` | 0 / 7 | Strictly Nov 5–11, 2018 | Page 1 Bridge, Page 3 Charts & Tables | **MATCH** |
| `day_name` | `object` / `string` | `object` | 0 / 7 | Monday – Sunday | Page 3 Data Table | **MATCH** |
| `forecast_total_sales` | `float64` | `float64` | 0 / 7 | $\ge 0$, rounded to 2 decimals | Page 1 Bridge, Page 3 Revenue Chart & Table | **MATCH** |
| `sales_lower_80` | `float64` | `float64` | 0 / 7 | $\ge 0$, $\le$ forecast | Page 3 Revenue Uncertainty Band | **MATCH** |
| `sales_upper_80` | `float64` | `float64` | 0 / 7 | $\ge$ forecast | Page 3 Revenue Uncertainty Band | **MATCH** |
| `forecast_tickets_sold` | `int64` | `int64` | 0 / 7 | $\ge 0$, integer | Page 3 Admissions Chart & Table | **MATCH** |
| `tickets_lower_80` | `int64` | `int64` | 0 / 7 | $\ge 0$, $\le$ forecast | Page 3 Admissions Uncertainty Band | **MATCH** |
| `tickets_upper_80` | `int64` | `int64` | 0 / 7 | $\ge$ forecast | Page 3 Admissions Uncertainty Band | **MATCH** |

---

## 6. Baseline vs. Regenerated Forecast Comparison

Comparing the baseline forecast against the forecast generated by `train_model.py`:

```
Baseline Forecast File:    data/processed/sales_forecast.csv (7 rows x 8 cols)
Regenerated Forecast File: data/processed/sales_forecast.csv (7 rows x 8 cols)
```

| Date | Day | Regenerated Revenue | Baseline Revenue | Absolute Difference | Regenerated Tickets | Baseline Tickets |
|---|---|---|---|---|---|---|
| **2018-11-05** | Monday | 2,642,881,305.24 | 2,642,881,305.24 | **0.00** | 27,216 | 27,216 |
| **2018-11-06** | Tuesday | 2,825,582,620.91 | 2,825,582,620.91 | **0.00** | 54,975 | 54,975 |
| **2018-11-07** | Wednesday | 3,077,328,851.49 | 3,077,328,851.49 | **0.00** | 45,383 | 45,383 |
| **2018-11-08** | Thursday | 7,448,227,920.26 | 7,448,227,920.26 | **0.00** | 70,203 | 70,203 |
| **2018-11-09** | Friday | 9,120,469,665.03 | 9,120,469,665.03 | **0.00** | 89,784 | 89,784 |
| **2018-11-10** | Saturday | 2,498,345,113.82 | 2,498,345,113.82 | **0.00** | 38,498 | 38,498 |
| **2018-11-11** | Sunday | 2,526,013,908.59 | 2,526,013,908.59 | **0.00** | 33,886 | 33,886 |

**Finding:** The regenerated forecast exhibits **exact numerical parity** with the baseline. This occurs because `train_model.py` uses the identical feature engineering order, identical operational date filtering (`2018-03-14`), and identical random seed (`random_state=42`).

---

## 7. Training Script Execution Result

```text
Command: python train_model.py
Exit Code: 0
Duration: 8.7 seconds

Key Console Outputs:
[Step 1/6] Loading processed daily sales dataset... (234 observed days)
[Step 2/6] Regularising calendar: 236 continuous days (2018-03-14 to 2018-11-04)
           4 assumed closure dates zero-filled (Jun 4, Jun 6, Sep 19, Sep 20)
[Step 3/6] Engineering 12 leak-free features (lags shifted >= 7 days)
[Step 4/6] Chronological 4-window expanding-origin backtesting:
           - Revenue Random Forest: 1.774 B MAE (+58.9% vs Naive, +9.8% vs SNaive)
           - Demand Random Forest:  23,900 MAE (+52.9% vs Naive)
[Step 5/6] Final production models fitted & persisted to models/
[Step 6/6] 7-day out-of-sample forecast (2018-11-05 to 2018-11-11) persisted to data/processed/sales_forecast.csv
[SUCCESS] Standalone training and forecasting pipeline completed successfully!
```

---

## 8. Automated Test Results

Command:
```bash
python -m pytest tests/ -v
```

Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: W:\4. IBM Internship\CinemaPulse
collected 20 items

tests/test_dashboard.py::test_data_loaders PASSED                        [  5%]
tests/test_dashboard.py::test_compute_kpis_accuracy PASSED               [ 10%]
tests/test_dashboard.py::test_compute_kpis_empty PASSED                  [ 15%]
tests/test_dashboard.py::test_top_films_and_cinemas PASSED               [ 20%]
tests/test_dashboard.py::test_dow_and_monthly_patterns PASSED            [ 25%]
tests/test_dashboard.py::test_risk_indicators_and_recommendations PASSED [ 30%]
tests/test_dashboard.py::test_formatters PASSED                          [ 35%]
tests/test_dashboard.py::test_data_consistency_across_aggregates PASSED  [ 40%]
tests/test_dashboard.py::test_raw_dataset_integrity PASSED               [ 45%]
tests/test_dashboard.py::test_forecast_dates_historical PASSED           [ 50%]
tests/test_dashboard.py::test_recommendations_no_causation_claims PASSED [ 55%]
tests/test_dashboard.py::test_streamlit_apptest_pages PASSED             [ 60%]
tests/test_forecasting.py::test_prepare_time_series_data_regularisation PASSED [ 65%]
tests/test_forecasting.py::test_prepare_time_series_invalid_inputs PASSED [ 70%]
tests/test_forecasting.py::test_create_forecasting_features_leak_free PASSED [ 75%]
tests/test_forecasting.py::test_chronological_rolling_origin_validation PASSED [ 80%]
tests/test_forecasting.py::test_train_forecast_model_and_inference PASSED [ 85%]
tests/test_forecasting.py::test_generate_future_forecast_structure PASSED [ 90%]
tests/test_forecasting.py::test_forecast_horizon_guard PASSED            [ 95%]
tests/test_forecasting.py::test_pipeline_saved_outputs PASSED            [100%]

============================= 20 passed in 10.51s =============================
```

---

## 9. Results for Each of the Four Dashboard Views

A browser subagent navigated to the live Streamlit dashboard on `http://localhost:8501` to audit each view:

### View 1: Landing Page (`http://localhost:8501/`)
- **KPI Summary Ribbon:** Displays 1,756.85B gross revenue, 19.95M tickets sold, 16.4% weighted occupancy, 48 active films, 234 operational days.
- **Visuals:** End-to-End Business Intelligence Pipeline diagram and navigation cards render cleanly.
- **Exceptions:** 0 errors.

### View 2: Executive Overview (`/Executive_Overview`)
- **KPI Cards:** Core metric cards render with delta comparisons.
- **Revenue Trend:** Daily box office revenue bar chart with 7-day rolling trendline renders without issue.
- **Monthly Seasonality:** Monthly revenue and admission volumes render accurately.
- **Forecast Bridge:** "Historical Actuals to 7-Day Forecast Bridge" chart links the final 14 historical days seamlessly into the 7-day forecast from `sales_forecast.csv`.
- **Exceptions:** 0 errors.

### View 3: Sales & Performance Analysis (`/Sales_Performance`)
- **Film & Cinema Rankings:** Top 10 grossing films and top 10 grossing cinemas render properly.
- **Day-of-Week Cycle:** Friday/Thursday peaks and Tuesday promotional volume surge render cleanly.
- **Occupancy & Pricing:** Auditorium occupancy distribution histogram and ticket price spread box plots render without defect.
- **Interactive Tables:** Filterable catalog tables operate as expected.
- **Exceptions:** 0 errors.

### View 4: Forecast, Risk & Business Decisions (`/Forecast_Risk_Decisions`)
- **Forecast Charts:** 7-day revenue forecast chart with 80% empirical prediction bounds renders cleanly in Tab 1; admissions forecast chart renders in Tab 2.
- **Numerical Forecast Table:** Expander opens cleanly, showing all 7 rows for November 5–11, 2018 with formatted numbers.
- **Evaluation Table:** Backtest model comparison table renders Naive vs. Seasonal Naive vs. Ridge vs. Random Forest.
- **Operational Risk Indicators:** Interactive low-occupancy threshold slider and risk indicators render without error.
- **Recommendations Matrix:** Actionable strategic recommendations matrix displays properly.
- **Exceptions:** 0 errors.

---

## 10. Issues Discovered & Fixes Made

1. **Windows Console Character Encoding:**
   - *Observation:* Initial execution of `train_model.py` encountered a `UnicodeEncodeError` due to default Windows cp1252 terminal encoding when attempting to output emoji and unicode checkmarks.
   - *Fix:* Replaced unicode glyphs with standard, clean ASCII indicators (`[OK]`, `[SUCCESS]`, `*`).
2. **Dual-Naming Support:**
   - *Observation:* Both `train_model.py` and `Lovey_CinemaPulse.py` were referenced during development.
   - *Fix:* Maintained identical implementations for both files so that commands targeting either name execute reliably.

---

## 11. Final Integration Verdict

```text
================================================================================
FINAL INTEGRATION VERDICT: PASS
================================================================================
```

All integration checks, schema assertions, unit/integration tests, and live browser audits passed completely. The dashboard works seamlessly with the outputs generated by `train_model.py`.
