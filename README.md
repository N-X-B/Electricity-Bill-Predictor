# Electricity Bill Predictor

**Singularity Research Lab – AI Division**  
**Academic Machine Learning Research Initiative**

---

## 1. Project Overview
The objective of this project is to develop and evaluate supervised machine learning models that forecast daily household electricity consumption and accurately estimate future electricity bills. By leveraging granular historical smart meter time-series data, the project demonstrates an end-to-end reproducible machine learning pipeline designed to academic and production standards.

### Primary Goals:
- **Time-Series Aggregation & Preprocessing:** Aggregate minute-level household consumption data to daily totals without forward-looking bias.
- **Model Comparison:** 
  - **Linear Regression (Primary):** Establishes an interpretable, parametric baseline for electricity consumption forecasting.
  - **Decision Tree Regressor (Comparison):** Evaluates non-linear feature interactions and segment-based partitioning.
- **Evaluation Metric:** Mean Absolute Error (MAE) evaluated across chronologically split train and test partitions.
- **Financial Mapping:** Translate predicted kilowatt-hour (kWh) daily usage into estimated bill amounts using tiered and fixed tariff structures.

---

## 2. Dataset
- **Source:** [UCI Machine Learning Repository – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **File:** `household_power_consumption.txt`
- **Description:** Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.

---

## 3. Project Directory Structure

```text
Electricity-Bill-Predictor/
├── data/              # Raw and processed datasets (ignored in VCS except .gitkeep)
├── notebooks/         # Jupyter notebooks for EDA and step-by-step experimentation
├── src/               # Modular Python scripts (data ingestion, preprocessing, modeling, evaluation)
├── models/            # Serialized trained model artifacts (.joblib / .pkl)
├── results/           # Performance metrics, evaluation logs, error analysis, and plots
├── .gitignore         # Ignores bytecode, checkpoints, large datasets, and artifacts
├── README.md          # Project documentation and architectural overview
└── requirements.txt   # Environment dependencies and versions
```

---

## 4. Engineering Principles & Methodology
1. **Chronological Splitting:** Strictly avoid random shuffles. Training data strictly precedes test data in time to prevent lookahead bias.
2. **Leakage Prevention:** Lag feature creation, scaling, and rolling aggregations are computed strictly on past context without peeking into evaluation horizons.
3. **Reproducibility:** Deterministic seeds, modular Python packages under `src/`, and clean virtual environments.
4. **No Fabricated Metrics:** All evaluation results and observations reflect empirical outcomes on holdout sets.

---

## 5. Getting Started

### Prerequisites
- Python 3.10+ recommended

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
