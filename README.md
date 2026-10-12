# 🎬 CinemaPulse — Cinema Sales Intelligence & Demand Forecasting Dashboard

## Problem Statement

Cinema operators need data-driven visibility into ticket sales performance, seat
occupancy patterns, and demand trends so they can optimise scheduling, pricing,
and marketing spend.  Manual analysis of raw transactional data is slow, error-prone,
and cannot scale across hundreds of cinema halls and dozens of films.

**CinemaPulse** turns raw cinema ticket data into an interactive Business Intelligence
dashboard that surfaces actionable insights and forecasts.

---

## Project Objectives

1. **Clean & prepare** the Kaggle Cinema Ticket dataset for analysis.
2. **Explore** sales, occupancy, and pricing patterns across films, cinemas, and time.
3. **Forecast** future ticket sales using appropriate time-series or regression models.
4. **Identify risks** (low occupancy, pricing anomalies) and **opportunities** (high-demand periods).
5. **Present** findings in a multi-page Streamlit dashboard suitable for business stakeholders.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Cinema Ticket](https://www.kaggle.com/datasets/arashnic/cinema-ticket) |
| **File** | `cinemaTicket_Ref.csv` |
| **Rows** | 142 524 |
| **Columns** | 14 |
| **Date Range** | 2018-02-21 → 2018-11-04 (234 unique dates, ~8.5 months) |
| **Missing Values** | `occu_perc` (125), `capacity` (125) |
| **Duplicate Rows** | 106 exact duplicates |

### Column Dictionary

| Column | Dtype | Description |
|---|---|---|
| `film_code` | int | Film identifier (48 unique films) |
| `cinema_code` | int | Cinema hall identifier (246 unique cinemas) |
| `total_sales` | int | Total revenue from ticket sales for that show |
| `tickets_sold` | int | Number of tickets sold |
| `tickets_out` | int | Tickets distributed but not sold (complimentary / promo) |
| `show_time` | int | Number of show-times / screenings |
| `occu_perc` | float | Occupancy percentage |
| `ticket_price` | float | Price per ticket |
| `ticket_use` | int | Tickets actually used (attended) |
| `capacity` | float | Hall seating capacity (derived / fractional) |
| `date` | object | Show date (YYYY-MM-DD) |
| `month` | int | Month number (2–11) |
| `quarter` | int | Quarter number (1–4) |
| `day` | int | Day of the month |

### Potential Target Variables for Forecasting

- **`total_sales`** — revenue forecasting.
- **`tickets_sold`** — demand / volume forecasting.
- **`occu_perc`** — seat-utilisation forecasting.

---

## BI Workflow

```
Raw Data → Clean Data → EDA → Insights → Prediction → Dashboard → Decision
```

---

## Planned Dashboard Pages

| # | Page | Purpose |
|---|---|---|
| 1 | **Executive Overview** | KPI cards, trend lines, top films & cinemas |
| 2 | **Sales Performance** | Detailed breakdowns by film, cinema, time |
| 3 | **Forecast, Risk & Decisions** | Predictions, anomaly flags, recommendations |

---

## Technology Stack

| Layer | Tools |
|---|---|
| Data wrangling | Python, pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn |
| Dashboard | Streamlit (multi-page) |
| Notebook EDA | Jupyter Notebook |

---

## Project Structure

```
CinemaPulse/
├── app.py                        # Streamlit main entry point & navigation hub
├── Lovey_CinemaPulse.py          # Standalone machine learning training & forecasting script
├── requirements.txt              # Project dependency specifications
├── README.md                     # Comprehensive project documentation
├── .gitignore                    # Version control exclusions
├── data/
│   ├── raw/
│   │   └── cinemaTicket_Ref.csv  # Original Kaggle dataset (142,524 rows)
│   └── processed/
│       ├── cinema_cleaned.csv    # Cleaned transaction dataset (142,418 rows)
│       ├── daily_sales.csv       # Continuous daily sales aggregate (234 days)
│       └── sales_forecast.csv    # 7-day out-of-sample forecast with intervals
├── notebooks/
│   └── 01_Cinema_Data_Analysis.ipynb # Verified EDA & statistical visualizations
├── src/
│   ├── __init__.py               # Package marker
│   ├── data_loader.py            # Validated loaders with schema enforcement
│   ├── preprocessing.py          # Reproducible data cleaning pipeline
│   ├── analysis.py               # Analytical metric calculations & KPI engine
│   ├── forecasting.py            # Time-series modeling & backtesting engine
│   ├── insights.py               # Risk quantification & recommendations
│   └── ui_helpers.py             # UI styling, Plotly themes & cached loaders
├── pages/
│   ├── 1_Executive_Overview.py   # Page 1: High-level KPIs & revenue trends
│   ├── 2_Sales_Performance.py    # Page 2: Film, cinema & schedule breakdowns
│   └── 3_Forecast_Risk_Decisions.py # Page 3: ML forecasts, risks & decisions
├── models/
│   ├── rf_sales_model.joblib     # Persisted Random Forest revenue model
│   └── rf_tickets_model.joblib   # Persisted Random Forest demand model
├── reports/
│   ├── phase6_qa_report.md       # Phase 6 QA and test suite audit report
│   ├── documentation_dependency_audit.md # Phase 7 documentation & dependencies audit
│   └── figures/                  # Exported analytical charts
├── tests/
│   ├── test_dashboard.py         # Dashboard, KPI & UI AppTest tests (12 tests)
│   └── test_forecasting.py       # Time-series, leakage & ML model tests (8 tests)
└── assets/                       # Visual assets, branding and CSS helpers
```

---

## Current Setup Status

- [x] Dataset located (`cinemaTicket_Ref.csv`, 10.7 MB)
- [x] Dataset inspected — columns, types, shape, missing values, duplicates documented
- [x] Project folder structure created
- [x] `requirements.txt` created and audited
- [x] `.gitignore` created
- [x] Raw dataset copied to `data/raw/`
- [x] Data cleaning pipeline (`src/preprocessing.py`) — 142,524 → 142,418 rows
- [x] Cleaned dataset saved to `data/processed/cinema_cleaned.csv`
- [x] Daily aggregate saved to `data/processed/daily_sales.csv` (234 days)
- [x] Exploratory Data Analysis notebook (`notebooks/01_Cinema_Data_Analysis.ipynb`) (Phase 3)
- [x] Demand forecasting models & validation (`src/forecasting.py`) (Phase 4)
- [x] Multi-page Streamlit Dashboard build (`app.py`, `pages/`) (Phase 5)
- [x] End-to-End Quality Assurance & Validation (`reports/phase6_qa_report.md`) (Phase 6)
- [x] Documentation & Dependency Consistency Audit (`reports/documentation_dependency_audit.md`) (Phase 7)


---

## 🖥️ Streamlit Business Intelligence Dashboard (Phase 5)

The interactive multi-page dashboard translates raw data and machine learning forecasts into actionable business intelligence across three dedicated pages.

### Launching the Dashboard

```bash
# Launch the Streamlit application
streamlit run app.py
```

Navigate across pages using the sidebar menu:
1. **Landing Page (`app.py`):** High-level KPI ribbon, BI workflow architecture, navigation directory, and data governance disclosures.
2. **Executive Overview (`pages/1_Executive_Overview.py`):**
   - KPI cards: Total Revenue, Total Admissions, Average Daily Revenue, Weighted Occupancy, Active Footprint.
   - Interactive daily revenue bar chart with 7-day rolling moving trendline.
   - Monthly revenue and ticket admissions comparison.
   - Historical actual revenue continuation into the 7-day out-of-sample forecast.
   - Three verified executive takeaway callouts.
   - Date range selector.
3. **Sales & Performance Analysis (`pages/2_Sales_Performance.py`):**
   - Top 10 grossing films (horizontal ranking bar chart).
   - Top 10 grossing cinemas (horizontal ranking bar chart).
   - Day-of-week demand cycle showcasing Friday/Thursday peaks and Tuesday discount volume.
   - Screening occupancy distribution histogram.
   - Ticket price spread box plots across days of the week.
   - Sortable, filterable catalog performance data tables for films and cinemas.
   - Multi-dimensional filters: Date range, day of week, film code, cinema code.
4. **Forecast, Risk & Decisions (`pages/3_Forecast_Risk_Decisions.py`):**
   - Historical 7-day forecast visualization (November 5–11, 2018) with empirical 80% prediction intervals for both revenue and ticket demand.
   - Chronological backtesting evaluation metrics table (Random Forest vs Ridge vs Seasonal Naive vs Naive).
   - Operational risk indicators with an interactive low-occupancy alert threshold slider.
   - Actionable strategic recommendations matrix directly tied to verified findings.

---

## 🧪 Running the Automated Test Suite

```bash
# Run the complete test suite (20 tests covering loaders, features, models, formatters, UI, and edge cases)
python -m pytest tests/ -v
```

---

## Quick Start & Reproducibility Guide

### Python Version Compatibility
- **Supported:** Python 3.10 – 3.13
- **Tested Environment:** Python 3.13.9 (64-bit on Windows)
- **Dependencies:** All listed in `requirements.txt` (installable via pip)

### Step 1: Environment Setup
```bash
# Clone or navigate to the project directory
cd "CinemaPulse"

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate        # Windows (PowerShell / Command Prompt)
# source venv/bin/activate   # macOS / Linux

# Install audited dependencies
pip install -r requirements.txt
```

### Step 2: Running Project Workflows

#### Option A: Launch the Streamlit Dashboard (Immediate)
Pre-cleaned datasets and verified forecast artifacts are bundled in `data/processed/`, allowing immediate launch without retraining:
```bash
streamlit run app.py
```
Open your web browser at: `http://localhost:8501`

#### Option B: Run the Automated Test Suite
Execute the 20 automated unit, integration, and UI tests:
```bash
python -m pytest tests/ -v
```

#### Option C: Explore the Jupyter EDA Notebook
Inspect verified statistical findings and exploratory visualizations:
```bash
jupyter notebook notebooks/01_Cinema_Data_Analysis.ipynb
```

#### Option D: Rebuild Pipelines from Raw Data (Optional)
To regenerate cleaned datasets and retrain forecasting models from `data/raw/cinemaTicket_Ref.csv`:
```bash
# 1. Clean raw transactions and generate daily aggregates
python -m src.preprocessing

# 2. Re-run chronological backtesting, retrain models, and generate 7-day forecast
python -m src.forecasting
```

---

## 🤖 Standalone Model Training (`Lovey_CinemaPulse.py`)

For standalone execution, submission, and grading outside of package-level imports, a fully self-contained machine learning script is available at the project root:

```bash
# Execute standalone model training and forecast generation
python Lovey_CinemaPulse.py
```

### Overview & Methodology
- **Purpose:** Implements the complete end-to-end forecasting pipeline in a single self-contained file without relying on external dashboard processes or importing `src.forecasting`.
- **Input Dataset:** `data/processed/daily_sales.csv` (234 observed days). If missing, the script gracefully alerts the user to run `python -m src.preprocessing`.
- **Calendar Regularisation:** Reindexes onto a continuous daily sequence starting on `2018-03-14` (236 continuous days, excluding initial onboarding ramp-up). Fills 4 unrecorded operational closure dates (June 4, June 6, September 19, September 20) with 0.0 sales/tickets and sets an `is_closed` assumption indicator.
- **Feature Engineering:** Constructs 12 leak-free features (`is_regional_weekend`, `is_tuesday`, `month`, `lag_7`, `lag_14`, `rolling_mean_7_lag7`, `dow_1`..`dow_6`). All lags and rolling statistics are shifted $\ge 7$ days to prevent future information leakage over the 7-day multi-step horizon.
- **Models Evaluated:**
  1. *Naive Baseline* (last observed value at $t-7$)
  2. *Seasonal Naive* (same weekday prior week, $t-7$)
  3. *Ridge Regression* (`alpha=10.0` with `StandardScaler`)
  4. *Random Forest Regressor* (`n_estimators=100, max_depth=5, random_state=42`)
- **Evaluation Protocol:** 4-window expanding-origin chronological backtesting (28 test days: Oct 8 – Nov 4, 2018).
- **Generated Output Artifacts:**
  - `models/rf_sales_model.joblib`: Production Random Forest revenue model.
  - `models/rf_tickets_model.joblib`: Production Random Forest ticket demand model.
  - `data/processed/sales_forecast.csv`: 7-day forecast table with empirical 80% prediction intervals.
- **Temporal Alignment:** The historical dataset terminates on November 4, 2018. The generated forecast spans **November 5–11, 2018** (the immediate historical out-of-sample period). It does not represent a real-time current calendar forecast.
- **Methodological Limitations:** Prediction intervals ($\hat{y} \pm 1.2816 \times s_e$) are heuristic empirical intervals derived from backtest residual standard deviations; they are not mathematically guaranteed coverage intervals under regime changes or non-stationary market dynamics.

---

## 🔮 Time-Series Forecasting Architecture

### Problem & Horizon
- **Primary Target:** `total_sales` (daily box office revenue).
- **Secondary Target:** `tickets_sold` (daily admissions).
- **Forecast Horizon:** **7 future calendar days** (`2018-11-05` to `2018-11-11`).
- **Missing-Date Treatment:** Full calendar span contains 257 days (`2018-02-21` to `2018-11-04`) with 234 observed dates. The 23 unobserved days comprise 19 early onboarding days (Feb 21–Mar 13) and 4 isolated operational closure days (Jun 4, Jun 6, Sep 19, Sep 20 — mandatory entertainment closures in Iran). Regular commercial modeling begins on `2018-03-14` on a continuous daily calendar, with official closure days zero-filled.

### Feature Engineering & Leakage Prevention
To guarantee zero future-information leakage over a 7-day forecast horizon, all lag and rolling features are shifted by at least 7 days:
- `is_regional_weekend`: Thursday & Friday indicator (peak Middle Eastern workweek rest days).
- `is_tuesday`: Promotional discount cinema day.
- `dow_1` to `dow_6`: One-hot encoded day of week.
- `lag_7` & `lag_14`: Values strictly observed 7 and 14 days prior.
- `rolling_mean_7_lag7`: Rolling 7-day mean ending at $t-7$.

### Chronological Validation & Model Performance
Evaluated across **4 non-overlapping chronological 7-day windows** (28 test days total, expanding training window):

| Target | Model | MAE | RMSE | vs Baseline |
|---|---|---|---|---|
| **Revenue (`total_sales`)** | **Naive (last observed)** | 4.320 B | 5.717 B | Baseline |
| | **Seasonal Naive (lag 7)** | 1.967 B | 2.467 B | +54.5% MAE reduction |
| | **Ridge Regression** | 1.836 B | 2.414 B | +57.5% MAE reduction |
| | **Random Forest (Candidate)** | **1.774 B** | **2.288 B** | **+58.9% MAE reduction** (+9.8% vs SNaive) |
| **Demand (`tickets_sold`)** | **Naive (last observed)** | 50,734 | 67,695 | Baseline |
| | **Seasonal Naive (lag 7)** | 24,384 | 34,554 | +52.0% |
| | **Random Forest (Candidate)** | **23,900** | **34,706** | **+52.9% MAE reduction** |

### 7-Day Out-of-Sample Forecast (`2018-11-05` to `2018-11-11`)
Exported to `data/processed/sales_forecast.csv` with heuristic 80% empirical prediction intervals:
- **Nov 5 (Mon):** 2.64 B [0.00 B – 6.09 B] | 27,216 tickets
- **Nov 6 (Tue):** 2.83 B [0.00 B – 6.27 B] | 54,975 tickets (Promotional volume surge)
- **Nov 7 (Wed):** 3.08 B [0.00 B – 6.52 B] | 45,383 tickets
- **Nov 8 (Thu):** 7.45 B [4.00 B – 10.90 B] | 70,203 tickets (Regional weekend start)
- **Nov 9 (Fri):** 9.12 B [5.67 B – 12.57 B] | 89,784 tickets (Weekly peak)
- **Nov 10 (Sat):** 2.50 B [0.00 B – 5.95 B] | 38,498 tickets
- **Nov 11 (Sun):** 2.53 B [0.00 B – 5.97 B] | 33,886 tickets

---

## ⚠️ Data Quality, Governance & Project Limitations

To maintain academic and professional transparency, users and evaluators should take note of the following explicit project boundaries:

1. **Dataset Temporal Bounds:** The historical transaction dataset begins on February 21, 2018 and strictly terminates on **November 4, 2018**.
2. **Historical Forecast Horizon:** The demand and revenue forecast spans **November 5–11, 2018** (an evaluated out-of-sample historical period). It does not represent an extrapolation to current calendar dates.
3. **Anonymized Entities:** All movie titles and theater halls are represented via numerical identifiers (`film_code`, `cinema_code`) rather than commercial brand names.
4. **Calendar Closure Assumptions:** Four unrecorded operational dates (June 4, June 6, September 19, September 20) are modeled as mandatory entertainment closures based on regional calendar analysis, but are not explicitly tagged in dataset metadata.
5. **No-Show Approximations:** "No-shows" (calculated as `tickets_sold - ticket_use` = 31,523 tickets / 0.16%) are a **derived mathematical proxy**, not an audited turnstile gate count.
6. **Heuristic Prediction Intervals:** The 80% forecast prediction intervals ($\hat{y} \pm 1.2816 \times s_e$) are derived from backtest residual standard deviations. While empirically covering 89.3% in historical backtesting, they are **heuristic bounds** and are not guaranteed to achieve exact 80% coverage under non-stationary market dynamics.
7. **Currency Metadata:** The dataset does not specify official ISO currency symbols. Figures are reported in nominal currency units (scaled to B = billion, M = million, K = thousand) without inventing currency signs.

---

## License

This project is developed as part of the **IBM SkillsBuild Internship** program.
