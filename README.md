# Electricity Bill Predictor

**Singularity Research Lab – AI Division**  
**Academic Machine Learning Research Initiative**

---

## 1. Project Overview
The **Electricity Bill Predictor** project develops, validates, and deploys an end-to-end reproducible machine learning pipeline that forecasts daily household electricity consumption and estimates future electricity bills. 

Using the official [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset (2,075,259 observations spanning ~4 years), the project models daily active energy consumption ($kWh$) using strictly forward-looking chronological splitting, zero target leakage, and empirical validation.

---

## 2. Repository Structure

```text
Electricity-Bill-Predictor/
├── data/
│   ├── household_power_consumption.txt        # Raw UCI dataset (uncompressed, ~127 MB)
│   ├── daily_electricity_consumption.csv      # Clean daily aggregates (1,440 continuous days)
│   └── featured_daily_electricity_consumption.csv # Modeling dataset with lags & calendar features (1,433 days)
├── notebooks/
│   ├── 01_data_understanding.ipynb            # Task 2: Inspection, schema, dtypes, missing values
│   ├── 02_data_preprocessing.ipynb             # Task 3: Ingestion, coverage rules, daily Riemann integration
│   ├── 03_exploratory_data_analysis.ipynb      # Task 4: Statistical EDA, annual seasonality, occupancy patterns
│   ├── 04_feature_engineering.ipynb           # Task 5: Lags, rolling averages, target leakage proofs
│   ├── 05_model_training.ipynb                # Task 6: Chronological split (80/20), Linear Regression, Decision Tree
│   ├── 06_model_evaluation.ipynb              # Task 7: MAE/RMSE/R² metrics, residual diagnostics, error analysis
│   └── 07_future_forecasting_and_billing.ipynb # Task 8: 30-day recursive forecast & bill estimation
├── src/
│   ├── __init__.py                            # Python package initialization
│   ├── preprocessing.py                       # Data cleaning, boundary filtering, daily aggregation
│   ├── feature_engineering.py                 # Leak-free lag and calendar feature generation
│   ├── train.py                               # Chronological splitting, model fitting, and serialization
│   ├── evaluation.py                          # Metric calculation, residual plotting, comparison tables
│   └── prediction.py                          # Multi-step recursive forecasting and bill estimation
├── models/
│   ├── linear_regression.joblib               # Primary production model artifact
│   └── decision_tree.joblib                   # Comparison model artifact
├── results/
│   ├── eda_time_series_daily_energy.png       # 4-year time series with 7d & 30d rolling means
│   ├── eda_monthly_consumption_pattern.png    # Monthly consumption bar chart & boxplots
│   ├── eda_weekly_consumption_pattern.png     # Day-of-week & weekday vs weekend patterns
│   ├── eda_distribution_histogram_kde.png     # Target distribution, KDE, and boxplot
│   ├── eda_outliers_and_anomalies.png         # Winter peaks & summer vacation baseload
│   ├── test_predictions.csv                   # Out-of-time test predictions (287 days)
│   ├── eval_actual_vs_predicted.png           # Test actual vs predicted time series & parity scatter
│   ├── eval_residual_plots.png                # Residuals vs fitted values & error distributions
│   ├── eval_model_comparison.png              # Grouped bar chart comparing Train vs Test metrics
│   ├── model_comparison.csv                   # Machine-readable model metrics comparison (CSV)
│   ├── model_comparison.json                  # Machine-readable model metrics comparison (JSON)
│   ├── future_forecast.csv                    # 30-day recursive forecast & daily charges schedule
│   └── future_forecast_visualization.png      # Historical context, 30-day projection, cumulative bill
├── .gitignore                                 # Excludes large raw dumps, virtual environments, caches
├── README.md                                  # Full system documentation
└── requirements.txt                           # Pinned dependencies
```

---

## 3. End-to-End Pipeline Workflow

### Task 1: Environment & Architecture Setup
- Standardized directory layout for academic and production reproducibility.
- Automated virtual environment configuration and pinned dependencies in `requirements.txt`.

### Task 2: Dataset Acquisition & Inspection
- Downloaded and verified UCI dataset (`household_power_consumption.txt`, 2,075,259 rows, 9 columns).
- Identified that exactly 25,979 rows (1.2518%) have missing power telemetry simultaneously (synchronous sensor dropout).

### Task 3: Data Cleaning & Daily Aggregation
- **Physical Integral:** Aggregated minute-averaged active power ($kW$) to daily energy ($kWh$):
  $$\text{Daily\_Energy\_kWh} = \frac{1}{60} \sum_{m=1}^{1,440} P_m$$
- **Boundary Filtering:** Removed truncated start day (`2006-12-16`, 396 mins) and end day (`2010-11-26`, 1263 mins).
- **Anti-Leakage Missingness Rule:** Days with $<95\%$ coverage are assigned `NaN` and interpolated across a continuous calendar index (`freq='D'`). Missing readings were **never** treated as zero.

### Task 4: Exploratory Data Analysis (EDA)
- **Annual Seasonality:** December peak (mean 35.66 kWh) vs. August trough (mean 13.77 kWh)—a 61.4% drop driven by winter space heating in the Paris region.
- **Weekly Schedule:** Weekend consumption (29.41 kWh) is 18.5% higher than weekdays (24.82 kWh).
- **Vacation Baseload:** Unoccupied days in late August 2008 establish the household's minimum standby load at ~4.2 kWh/day.

### Task 5: Feature Engineering
- Constructed leak-free features:
  - `Lag_1` ($y_{t-1}$): Short-term daily persistence ($r = +0.614$).
  - `Lag_7` ($y_{t-7}$): Weekly periodicity ($r = +0.549$).
  - `Rolling_Mean_7`: Past 7-day average strictly excluding day $t$ ($r = +0.675$).
  - `Day_of_Week` ($0..6$), `Month` ($1..12$), `Is_Weekend` ($0/1$).
- Programmatically verified zero target leakage; removed initial 7 boundary rows ($N=1,433$).

### Task 6: Supervised Model Training
- **Chronological Split (80/20):** Training on earlier 1,146 days (`2006-12-24` to `2010-02-11`); Testing on later 287 days (`2010-02-12` to `2010-11-25`). Zero random shuffling.
- Trained **Linear Regression** (Primary Model) via closed-form OLS.
- Trained **Decision Tree Regressor** (Comparison Model) with depth regularization (`max_depth=5`, `min_samples_leaf=5`, `random_state=42`).

### Task 7: Model Evaluation & Comparison
- Evaluated on identical holdout test observations:

| Model Name | Train MAE | **Test MAE (Primary)** | Train RMSE | **Test RMSE** | Train $R^2$ | **Test $R^2$** | Out-of-Time Degradation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 5.3422 kWh | **4.0539 kWh** | 7.1313 kWh | **5.5803 kWh** | 0.5213 | **0.4166** | $\Delta R^2 = 0.1047$ (Stable) |
| **Decision Tree Regressor** | 4.7922 kWh | **4.1810 kWh** | 6.3973 kWh | **5.7524 kWh** | 0.6148 | **0.3801** | $\Delta R^2 = 0.2347$ (Overfitting) |

- **Linear Regression was selected as the superior production model:** lower Test MAE, lower Test RMSE, higher explained variance, and minimal generalization drop.

### Task 8: Future Forecasting & Financial Bill Estimation
- **30-Day Recursive Forecast (`2010-11-26` to `2010-12-25`):** Dynamically updates lag buffers from prior actuals to model predictions.
- **Projected Consumption:** **842.07 kWh** (Daily Mean: **28.07 kWh/day**).
- **Estimated Electricity Bill:** **₹6,315.53** at an assumed configurable benchmark tariff of **₹7.50 / kWh**.
- Documented key limitations of recursive forecasting: error compounding, variance attenuation, and absence of exogenous weather shock awareness.

---

## 4. Execution Guide

### Run End-to-End Pipelines
```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Run Data Preprocessing Pipeline
python src/preprocessing.py

# 2. Run Feature Engineering Pipeline
python src/feature_engineering.py

# 3. Run Model Training Pipeline
python src/train.py

# 4. Run Future Forecasting & Bill Estimation Pipeline
python src/prediction.py
```
