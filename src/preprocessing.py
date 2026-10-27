"""
Preprocessing and Daily Aggregation Pipeline for Electricity Bill Predictor.
Singularity Research Lab – AI Division

This module provides clean, modular, and reproducible functions to:
1. Ingest raw minute-level smart meter power consumption data.
2. Combine Date and Time into a unified Datetime index.
3. Audit missing value structures and quantify sensor outage patterns.
4. Filter out truncated boundary days lacking 24-hour coverage.
5. Handle short-duration telemetry dropouts via linear interpolation without 
   fabricating continuous multi-day outage blocks.
6. Check daily data coverage and aggregate minute-level active power (kW)
   into daily active energy (kWh) using the physical integral formula.
7. Avoid silent zero-filling of missing measurements and maintain calendar continuity.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw UCI Individual Household Electric Power Consumption dataset.
    
    Parameters
    ----------
    filepath : str
        Path to 'household_power_consumption.txt'.
        
    Returns
    -------
    pd.DataFrame
        Raw dataframe with '?' parsed as NaN and numeric columns formatted.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    logger.info("Loading raw dataset from %s ...", filepath)
    df = pd.read_csv(
        filepath,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    logger.info("Loaded %d rows and %d columns.", df.shape[0], df.shape[1])
    return df


def combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines 'Date' (dd/mm/yyyy) and 'Time' (hh:mm:ss) into a single pandas Datetime column,
    verifies chronological ordering, and enforces float types on telemetry columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'Date' and 'Time'.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with sorted 'Datetime' column and validated dtypes.
    """
    logger.info("Combining Date and Time into Datetime series...")
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Ensure numerical telemetry columns are float64
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def audit_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits missing telemetry values across columns, checks synchronous co-occurrence,
    and categorizes consecutive missing gap lengths.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numerical telemetry columns.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with audit metrics and gap distribution summaries.
    """
    null_counts = df[NUMERICAL_COLUMNS].isnull().sum().to_dict()
    null_percentages = (df[NUMERICAL_COLUMNS].isnull().sum() / len(df) * 100).to_dict()

    # Synchronous drop check
    all_null_rows = int(df[NUMERICAL_COLUMNS].isnull().all(axis=1).sum())
    any_null_rows = int(df[NUMERICAL_COLUMNS].isnull().any(axis=1).sum())

    # Gap length analysis on Global_active_power
    is_null = df["Global_active_power"].isnull()
    blocks = (is_null != is_null.shift()).cumsum()[is_null]
    gap_sizes = blocks.value_counts()

    return {
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "all_null_synchronous_rows": all_null_rows,
        "any_null_rows": any_null_rows,
        "total_missing_blocks": len(gap_sizes),
        "gaps_le_5min": int((gap_sizes <= 5).sum()),
        "gaps_6_to_60min": int(((gap_sizes > 5) & (gap_sizes <= 60)).sum()),
        "gaps_gt_60min": int((gap_sizes > 60).sum()),
        "max_gap_minutes": int(gap_sizes.max()) if len(gap_sizes) > 0 else 0,
    }


def filter_incomplete_boundary_days(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies and removes incomplete boundary days (first and last recording days)
    that do not span a full 24-hour cycle (1,440 minutes).
    
    For this dataset:
    - 2006-12-16 starts at 17:24 (only 396 minutes recorded).
    - 2010-11-26 ends at 21:02 (only 1,263 minutes recorded).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'Datetime'.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Filtered dataframe (2006-12-17 to 2010-11-25) and boundary statistics.
    """
    df = df.copy()
    first_date = df["Datetime"].dt.date.min()
    last_date = df["Datetime"].dt.date.max()

    first_day_records = int((df["Datetime"].dt.date == first_date).sum())
    last_day_records = int((df["Datetime"].dt.date == last_date).sum())

    boundary_info = {
        "first_date": str(first_date),
        "first_day_minutes": first_day_records,
        "first_day_complete": first_day_records == 1440,
        "last_date": str(last_date),
        "last_day_minutes": last_day_records,
        "last_day_complete": last_day_records == 1440,
    }

    # Filter strictly to full calendar days
    mask = (df["Datetime"].dt.date > first_date) & (df["Datetime"].dt.date < last_date)
    filtered_df = df[mask].reset_index(drop=True)

    logger.info(
        "Filtered truncated boundary days: %s (%d mins) and %s (%d mins). Kept %d records spanning %d full days.",
        first_date,
        first_day_records,
        last_date,
        last_day_records,
        len(filtered_df),
        filtered_df["Datetime"].dt.date.nunique(),
    )

    return filtered_df, boundary_info


def clean_minute_telemetry(df: pd.DataFrame, max_gap_minutes: int = 60) -> pd.DataFrame:
    """
    Applies time-series linear interpolation to short-duration missing telemetry gaps (<= max_gap_minutes).
    Leaves large multi-hour outage blocks as NaN to prevent artificial flat-line imputation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'Global_active_power'.
    max_gap_minutes : int
        Maximum consecutive minutes of missing data to interpolate linearly. Default is 60.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'clean_active_power' column.
    """
    df = df.copy()
    # Interpolate short telemetry dropouts linearly
    df["clean_active_power"] = df["Global_active_power"].interpolate(
        method="linear",
        limit=max_gap_minutes,
        limit_direction="forward",
    )
    return df


def aggregate_to_daily_energy(
    df: pd.DataFrame,
    coverage_threshold: float = 0.95,
    impute_missing_days: bool = True,
) -> pd.DataFrame:
    """
    Aggregates minute-level active power (kW) into daily energy consumption (kWh).
    
    Mathematical Formulation:
    -------------------------
    Power is instantaneous rate of energy flow: P (kW) = E (kWh) / time (h).
    Over a 1-minute period (dt = 1/60 h), energy consumed is:
        E_minute (kWh) = P_minute (kW) * (1/60) h = P_minute / 60
    For a complete 24-hour day (1,440 minutes):
        E_daily (kWh) = sum_{m=1}^{1440} (P_m / 60) = (sum_{m=1}^{1440} P_m) / 60
    
    Missing Data & Coverage Rules:
    ------------------------------
    1. Daily Coverage: Computes valid minute ratio: coverage = N_valid / 1440.
    2. Zero Avoidance: Days with coverage < coverage_threshold (e.g. 95%) are NOT
       summed to zero; they are assigned NaN because partial sums underestimate true usage.
    3. Calendar Regularity: A full daily calendar index is generated. If impute_missing_days
       is True, NaN days are interpolated using calendar-aware daily linear interpolation,
       preserving strict daily continuity (freq='D') required for true lag feature creation.
       
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned minute-level dataframe with 'Datetime' and 'clean_active_power'.
    coverage_threshold : float
        Minimum fraction of valid minutes required per day (default 0.95, i.e. >=1,368 mins).
    impute_missing_days : bool
        Whether to interpolate NaN days at daily resolution to preserve calendar regularity.
        
    Returns
    -------
    pd.DataFrame
        Clean daily dataframe with columns ['Date', 'Daily_Energy_kWh'].
    """
    df = df.copy()
    df["Date"] = df["Datetime"].dt.date

    # Group by Date and compute metrics
    daily_stats = df.groupby("Date").agg(
        total_minutes=("clean_active_power", "size"),
        valid_minutes=("clean_active_power", "count"),
        power_sum=("clean_active_power", "sum"),
    )

    daily_stats["coverage"] = daily_stats["valid_minutes"] / 1440.0

    # Aggregate using conversion formula: sum(kW) / 60
    # Strictly mask incomplete days with NaN to prevent false zero or deflated values
    daily_stats["Daily_Energy_kWh"] = np.where(
        daily_stats["coverage"] >= coverage_threshold,
        daily_stats["power_sum"] / 60.0,
        np.nan,
    )

    # Reindex to a complete daily calendar range to enforce strict temporal continuity
    min_date = daily_stats.index.min()
    max_date = daily_stats.index.max()
    full_date_idx = pd.date_range(start=min_date, end=max_date, freq="D").date

    daily_df = daily_stats.reindex(full_date_idx)
    daily_df.index.name = "Date"
    daily_df = daily_df.reset_index()

    nan_count = daily_df["Daily_Energy_kWh"].isnull().sum()
    logger.info(
        "Daily aggregation complete. Total calendar days: %d. Days with < %.0f%% coverage (set to NaN): %d.",
        len(daily_df),
        coverage_threshold * 100,
        nan_count,
    )

    if impute_missing_days and nan_count > 0:
        logger.info("Imputing %d missing days via daily time-series linear interpolation...", nan_count)
        daily_df["Daily_Energy_kWh"] = daily_df["Daily_Energy_kWh"].interpolate(
            method="linear",
            limit_direction="both",
        )

    # Return required columns: Date and Daily_Energy_kWh
    output_df = daily_df[["Date", "Daily_Energy_kWh"]].copy()
    output_df["Date"] = pd.to_datetime(output_df["Date"]).dt.strftime("%Y-%m-%d")
    output_df["Daily_Energy_kWh"] = output_df["Daily_Energy_kWh"].round(4)

    return output_df


def save_daily_dataset(daily_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the processed daily dataset to a CSV file.
    
    Parameters
    ----------
    daily_df : pd.DataFrame
        Clean daily energy consumption dataframe.
    output_path : str
        Destination path for CSV export.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_df.to_csv(output_path, index=False)
    logger.info("Successfully exported daily dataset to %s (%d rows).", output_path, len(daily_df))


def run_preprocessing_pipeline(
    raw_path: str = "data/household_power_consumption.txt",
    output_path: str = "data/daily_electricity_consumption.csv",
    coverage_threshold: float = 0.95,
    impute_missing_days: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates the complete data cleaning and daily aggregation pipeline.
    
    Parameters
    ----------
    raw_path : str
        Path to raw dataset.
    output_path : str
        Path to output CSV.
    coverage_threshold : float
        Minimum daily coverage required for valid sum.
    impute_missing_days : bool
        Whether to impute missing days to preserve continuous calendar frequency.
        
    Returns
    -------
    pd.DataFrame
        Processed daily energy dataframe.
    """
    df_raw = load_raw_data(raw_path)
    df_dt = combine_datetime(df_raw)
    df_filtered, _ = filter_incomplete_boundary_days(df_dt)
    df_clean = clean_minute_telemetry(df_filtered, max_gap_minutes=60)
    daily_df = aggregate_to_daily_energy(
        df_clean,
        coverage_threshold=coverage_threshold,
        impute_missing_days=impute_missing_days,
    )
    save_daily_dataset(daily_df, output_path)
    return daily_df


if __name__ == "__main__":
    # Test script execution
    run_preprocessing_pipeline()
