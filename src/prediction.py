"""
Future Consumption Forecasting and Electricity Bill Estimation Pipeline.
Singularity Research Lab – AI Division

This module provides reusable, modular functions to:
1. Generate multi-step out-of-sample daily forecasts using a recursive autoregressive strategy.
2. Maintain strict temporal integrity: past actuals and recursively generated predictions
   are fed into lag buffers without target leakage or future observation peeking.
3. Distinguish one-day-ahead walk-forward evaluation from recursive multi-day planning.
4. Convert forecasted kWh consumption into estimated electricity bills using an explicitly
   configurable assumed tariff rate in INR.
5. Generate publication-quality visualizations comparing historical context with future projections.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def recursive_multi_step_forecast(
    model: Any,
    historical_df: pd.DataFrame,
    forecast_horizon_days: int = 30,
    target_col: str = "Daily_Energy_kWh",
    date_col: str = "Date",
    min_physical_bound_kwh: float = 4.0,
) -> pd.DataFrame:
    """
    Generates multi-step future daily electricity consumption forecasts recursively.
    
    Recursive Forecasting Mechanism:
    ---------------------------------
    At step t = 1 (first future day):
        - Lag_1 is the last known actual daily consumption y_T.
        - Lag_7 is the actual consumption y_{T-6}.
        - Rolling_Mean_7 is the arithmetic mean of actuals [y_{T-6}, ..., y_T].
        - Model generates prediction y_hat_{T+1}.
    At step t = 2:
        - Lag_1 becomes y_hat_{T+1} (recursively generated prediction).
        - Lag_7 is y_{T-5} (actual).
        - Rolling_Mean_7 becomes mean of [y_{T-5}, ..., y_T, y_hat_{T+1}].
        - Model generates prediction y_hat_{T+2}.
    As t advances beyond 7 steps, all lag and rolling features are populated
    exclusively from recursively generated predictions.
    
    Zero Lookahead Guarantee:
    -------------------------
    Future actual consumption values are strictly NEVER accessed. Only historical
    records preceding the forecast start date are utilized.
    
    Parameters
    ----------
    model : Any
        Fitted Scikit-learn estimator (e.g. Linear Regression).
    historical_df : pd.DataFrame
        Historical daily dataset containing date_col and target_col.
    forecast_horizon_days : int
        Number of forward days to forecast (default 30).
    target_col : str
        Target column name (default 'Daily_Energy_kWh').
    date_col : str
        Date column name (default 'Date').
    min_physical_bound_kwh : float
        Lower clamp reflecting standby baseload (~4.0 kWh).
        
    Returns
    -------
    pd.DataFrame
        Table of future dates, lag states, and predicted daily kWh consumption.
    """
    df = historical_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    history_series = list(df[target_col].values)
    last_historical_date = df[date_col].iloc[-1]

    forecast_records = []

    logger.info(
        "Starting recursive forecasting for %d days ahead starting from %s ...",
        forecast_horizon_days,
        (last_historical_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    for step in range(1, forecast_horizon_days + 1):
        fc_date = last_historical_date + pd.Timedelta(days=step)
        day_of_week = fc_date.dayofweek
        month = fc_date.month
        is_weekend = int(day_of_week in [5, 6])

        # Extract lags from history buffer (mix of actuals and prior predictions)
        lag_1 = history_series[-1]
        lag_7 = history_series[-7]
        rolling_7 = float(np.mean(history_series[-7:]))

        # Assemble feature vector matching model training schema
        feature_row = pd.DataFrame([{
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Rolling_Mean_7": rolling_7,
            "Day_of_Week": day_of_week,
            "Month": month,
            "Is_Weekend": is_weekend,
        }])

        pred_kwh = float(model.predict(feature_row)[0])
        # Apply physical minimum baseload clamp
        pred_kwh = max(pred_kwh, min_physical_bound_kwh)

        # Append prediction to history buffer for subsequent recursive steps
        history_series.append(pred_kwh)

        forecast_records.append({
            "Forecast_Step": step,
            "Date": fc_date.strftime("%Y-%m-%d"),
            "Day_Name": fc_date.strftime("%A"),
            "Lag_1": round(lag_1, 4),
            "Lag_7": round(lag_7, 4),
            "Rolling_Mean_7": round(rolling_7, 4),
            "Day_of_Week": day_of_week,
            "Month": month,
            "Is_Weekend": is_weekend,
            "Forecasted_Daily_Energy_kWh": round(pred_kwh, 4),
        })

    forecast_df = pd.DataFrame(forecast_records)
    total_kwh = forecast_df["Forecasted_Daily_Energy_kWh"].sum()
    logger.info(
        "Recursive forecast complete. Horizon: %d days | Cumulative: %.2f kWh | Daily Avg: %.2f kWh.",
        forecast_horizon_days,
        total_kwh,
        total_kwh / forecast_horizon_days,
    )
    return forecast_df


def estimate_electricity_bill(
    forecast_df: pd.DataFrame,
    tariff_rate_inr_per_kwh: float = 7.50,
    fixed_charge_inr: float = 0.0,
    electricity_tax_pct: float = 0.0,
    fuel_adjustment_charge_per_kwh: float = 0.0,
) -> Dict[str, Any]:
    """
    Converts forecasted kWh energy consumption into estimated monetary electricity charges.
    
    Disclaimer & Tariff Assumptions:
    --------------------------------
    The assumed tariff rate (default: INR 7.50 / kWh) is explicitly configurable and
    used solely for academic modeling and simulation. It DOES NOT represent the official
    or actual tariff schedule of any specific electric distribution utility (e.g. BESCOM,
    MSEDCL, TANGEDCO, Tata Power, etc.).
    
    Scope of Calculations:
    ----------------------
    - Energy Charges: Forecasted_kWh * (tariff_rate_inr_per_kwh + fuel_adjustment_charge_per_kwh)
    - Fixed Charges: Fixed meter/sanctioned load charges (default 0.0 unless configured).
    - Taxes: Government electricity duty (default 0.0% unless configured).
    - Slab/Tiered Structures: Excluded in flat rate mode; cumulative bill is calculated uniformly.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Table with 'Forecasted_Daily_Energy_kWh' and 'Date'.
    tariff_rate_inr_per_kwh : float
        Assumed cost per kWh in Indian Rupees (INR).
    fixed_charge_inr : float
        Fixed monthly connection/meter charge in INR.
    electricity_tax_pct : float
        Percentage tax / duty on energy charges (e.g. 5.0 for 5%).
    fuel_adjustment_charge_per_kwh : float
        Additional per-kWh surcharge (default 0.0).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with billing breakdown, cumulative metrics, and augmented daily table.
    """
    df = forecast_df.copy()
    effective_rate = tariff_rate_inr_per_kwh + fuel_adjustment_charge_per_kwh

    df["Estimated_Daily_Charge_INR"] = (df["Forecasted_Daily_Energy_kWh"] * effective_rate).round(2)
    df["Cumulative_Energy_kWh"] = df["Forecasted_Daily_Energy_kWh"].cumsum().round(4)
    df["Cumulative_Charge_INR"] = df["Estimated_Daily_Charge_INR"].cumsum().round(2)

    total_kwh = float(df["Forecasted_Daily_Energy_kWh"].sum())
    total_energy_charges = float(df["Estimated_Daily_Charge_INR"].sum())
    taxes_inr = round(total_energy_charges * (electricity_tax_pct / 100.0), 2)
    total_bill_inr = round(total_energy_charges + fixed_charge_inr + taxes_inr, 2)

    billing_summary = {
        "forecast_period_start": df["Date"].iloc[0],
        "forecast_period_end": df["Date"].iloc[-1],
        "total_forecast_days": len(df),
        "total_consumption_kwh": round(total_kwh, 2),
        "average_daily_consumption_kwh": round(total_kwh / len(df), 2),
        "assumed_tariff_rate_inr_per_kwh": tariff_rate_inr_per_kwh,
        "fuel_adjustment_charge_per_kwh": fuel_adjustment_charge_per_kwh,
        "total_energy_charges_inr": round(total_energy_charges, 2),
        "fixed_charges_inr": fixed_charge_inr,
        "electricity_tax_pct": electricity_tax_pct,
        "electricity_tax_inr": taxes_inr,
        "total_estimated_bill_inr": total_bill_inr,
        "average_cost_per_day_inr": round(total_bill_inr / len(df), 2),
        "disclaimer": (
            "Tariff rate is an assumed benchmark for research simulation. "
            "Excludes provider-specific tiered slab structures, time-of-day (ToD) surcharges, "
            "and municipal duties unless explicitly configured."
        ),
    }

    return {
        "summary": billing_summary,
        "detailed_forecast_df": df,
    }


def plot_future_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: str = "results/future_forecast_visualization.png",
    history_window_days: int = 90,
) -> None:
    """
    Renders and saves a visualization showing recent historical daily consumption
    alongside the 30-day recursive forecast horizon, marking weekends and billing totals.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2, 1]})

    hist_df = historical_df.copy()
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df = hist_df.sort_values("Date").tail(history_window_days).reset_index(drop=True)

    fc_df = forecast_df.copy()
    fc_df["Date"] = pd.to_datetime(fc_df["Date"])

    # Panel 1: Time Series Progression
    ax1.plot(
        hist_df["Date"],
        hist_df["Daily_Energy_kWh"],
        color="#37474f",
        linewidth=1.4,
        label=f"Historical Actual Consumption (Recent {history_window_days} Days)",
    )
    ax1.plot(
        fc_df["Date"],
        fc_df["Forecasted_Daily_Energy_kWh"],
        color="#d32f2f",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Recursive Multi-Step Forecast (Next 30 Days)",
    )

    # Cutoff line
    cutoff_date = hist_df["Date"].iloc[-1]
    ax1.axvline(cutoff_date, color="black", linestyle=":", linewidth=1.5, label=f"Forecast Cutoff ({cutoff_date.strftime('%Y-%m-%d')})")

    # Highlight weekend pulses
    weekend_fc = fc_df[fc_df["Is_Weekend"] == 1]
    ax1.scatter(weekend_fc["Date"], weekend_fc["Forecasted_Daily_Energy_kWh"], color="#e65100", s=60, zorder=5, label="Forecasted Weekend Days")

    ax1.set_title("Historical Electricity Consumption and 30-Day Recursive Forward Projection", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Daily Electricity Consumption (kWh)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Cumulative Daily Bill Projection (INR)
    if "Cumulative_Charge_INR" in fc_df.columns:
        ax2.plot(fc_df["Date"], fc_df["Cumulative_Charge_INR"], color="#2e7d32", linewidth=2.2, marker="s", markersize=4, label="Cumulative Estimated Bill (INR)")
        ax2.fill_between(fc_df["Date"], fc_df["Cumulative_Charge_INR"], color="#81c784", alpha=0.3)
        final_bill = fc_df["Cumulative_Charge_INR"].iloc[-1]
        ax2.annotate(
            f"30-Day Total: INR {final_bill:,.2f}",
            xy=(fc_df["Date"].iloc[-1], final_bill),
            xytext=(fc_df["Date"].iloc[-1] - pd.Timedelta(days=8), final_bill - 600),
            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.5),
            fontsize=11,
            fontweight="bold",
            color="#1b5e20",
        )
        ax2.set_title("Cumulative Electricity Bill Projection (Assumed Tariff: INR 7.50 / kWh)", fontsize=13, fontweight="bold", pad=8)
        ax2.set_xlabel("Forecast Date", fontsize=11)
        ax2.set_ylabel("Cumulative Charges (INR)", fontsize=11)
        ax2.legend(loc="upper left", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved forecast visualization to %s", output_path)


def run_forecasting_pipeline(
    model_path: str = "models/linear_regression.joblib",
    data_path: str = "data/daily_electricity_consumption.csv",
    output_csv_path: str = "results/future_forecast.csv",
    output_viz_path: str = "results/future_forecast_visualization.png",
    forecast_days: int = 30,
    tariff_rate_inr: float = 7.50,
) -> Dict[str, Any]:
    """
    Executes the complete forecasting and billing pipeline.
    """
    model = joblib.load(model_path)
    historical_df = pd.read_csv(data_path)

    # 1. Generate recursive forecasts
    fc_df = recursive_multi_step_forecast(model, historical_df, forecast_horizon_days=forecast_days)

    # 2. Estimate bill
    bill_results = estimate_electricity_bill(fc_df, tariff_rate_inr_per_kwh=tariff_rate_inr)
    detailed_df = bill_results["detailed_forecast_df"]

    # 3. Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    detailed_df.to_csv(output_csv_path, index=False)
    logger.info("Exported future forecasts to %s (%d rows).", output_csv_path, len(detailed_df))

    # 4. Generate plot
    plot_future_forecast(historical_df, detailed_df, output_path=output_viz_path)

    return bill_results


if __name__ == "__main__":
    run_forecasting_pipeline()
