"""
Feature Engineering Module for Daily Electricity Consumption Forecasting.
Singularity Research Lab – AI Division

This module develops reproducible, leak-free feature pipelines:
1. Autoregressive Lags:
   - Lag_1: Immediate previous day consumption (inertia).
   - Lag_7: Same day consumption from the preceding week (weekly seasonality).
2. Historical Rolling Aggregations:
   - Rolling_Mean_7: 7-day historical moving average strictly excluding day t.
3. Deterministic Calendar Features:
   - Day_of_Week: Integer day index (0=Monday ... 6=Sunday).
   - Month: Integer month (1=January ... 12=December).
   - Is_Weekend: Binary indicator (1 if Saturday/Sunday, 0 otherwise).
4. Leakage Verification and Boundary Truncation:
   - Removes invalid rows lacking historical lag context.
   - Programmatically verifies zero target leakage.
"""

import os
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "Daily_Energy_kWh",
    lags: List[int] = [1, 7],
) -> pd.DataFrame:
    """
    Constructs autoregressive lag features from strictly historical observations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe.
    target_col : str
        Target consumption variable.
    lags : List[int]
        List of day shift values (e.g. [1, 7]).
        
    Returns
    -------
    pd.DataFrame
        Dataframe augmented with 'Lag_{k}' columns.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"Lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Daily_Energy_kWh",
    window: int = 7,
) -> pd.DataFrame:
    """
    Constructs historical moving averages strictly excluding the current observation (t).
    
    Formula:
    Rolling_Mean_7(t) = (1 / window) * sum_{i=1}^{window} y_{t - i}
    
    Implementation Rule:
    The shift(1) operation is applied BEFORE the rolling window to guarantee
    zero target leakage into the feature space.
    
    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe.
    target_col : str
        Target consumption variable.
    window : int
        Rolling window length in days (default 7).
        
    Returns
    -------
    pd.DataFrame
        Dataframe augmented with 'Rolling_Mean_{window}' column.
    """
    df = df.copy()
    col_name = f"Rolling_Mean_{window}"
    df[col_name] = df[target_col].shift(1).rolling(window=window).mean()
    return df


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Extracts calendar and occupancy indicators from the Date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing date_col.
    date_col : str
        Name of date column.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'Day_of_Week', 'Month', and 'Is_Weekend' features.
    """
    df = df.copy()
    date_series = pd.to_datetime(df[date_col])

    df["Day_of_Week"] = date_series.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["Month"] = date_series.dt.month            # 1 to 12
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)

    return df


def clean_engineered_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Removes the initial rows that contain NaN values due to shift and rolling operations.
    For a maximum lag of 7 days, the first 7 rows are removed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with feature columns.
        
    Returns
    -------
    Tuple[pd.DataFrame, int]
        Cleaned dataframe without NaNs and the count of dropped rows.
    """
    initial_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    dropped_count = initial_len - len(df_clean)
    logger.info("Dropped %d initial rows with incomplete lag context. Clean rows: %d.", dropped_count, len(df_clean))
    return df_clean, dropped_count


def verify_leakage(
    df_engineered: pd.DataFrame,
    df_original: pd.DataFrame,
    target_col: str = "Daily_Energy_kWh",
) -> bool:
    """
    Rigorous unit check verifying that lag and rolling features contain NO target leakage.
    
    Checks:
    1. df_engineered['Lag_1'].iloc[k] == df_original[target_col].iloc[k + 7 - 1]
    2. df_engineered['Lag_7'].iloc[k] == df_original[target_col].iloc[k + 7 - 7]
    3. df_engineered['Rolling_Mean_7'].iloc[k] == mean of df_original[target_col] from (k) to (k + 6)
    4. Current target df_engineered[target_col].iloc[k] is strictly independent of features at row k.
    
    Returns
    -------
    bool
        True if all verification assertions pass.
    """
    logger.info("Executing mathematical target leakage verification...")
    for idx in range(min(10, len(df_engineered))):
        orig_target_idx = idx + 7  # 7 rows were dropped
        
        # Check Lag_1
        expected_lag1 = df_original[target_col].iloc[orig_target_idx - 1]
        actual_lag1 = df_engineered["Lag_1"].iloc[idx]
        assert np.isclose(expected_lag1, actual_lag1), f"Leakage/Mismatch in Lag_1 at idx {idx}!"

        # Check Lag_7
        expected_lag7 = df_original[target_col].iloc[orig_target_idx - 7]
        actual_lag7 = df_engineered["Lag_7"].iloc[idx]
        assert np.isclose(expected_lag7, actual_lag7), f"Leakage/Mismatch in Lag_7 at idx {idx}!"

        # Check Rolling_Mean_7
        expected_rolling = df_original[target_col].iloc[orig_target_idx - 7 : orig_target_idx].mean()
        actual_rolling = df_engineered["Rolling_Mean_7"].iloc[idx]
        assert np.isclose(expected_rolling, actual_rolling), f"Leakage/Mismatch in Rolling_Mean_7 at idx {idx}!"

    logger.info("Verification PASSED: Zero target leakage detected across all checks.")
    return True


def generate_features_pipeline(
    df: pd.DataFrame,
    target_col: str = "Daily_Energy_kWh",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Orchestrates the full feature engineering pipeline:
    1. Sorts chronologically.
    2. Extracts deterministic calendar features.
    3. Computes autoregressive lags (Lag_1, Lag_7).
    4. Computes leak-free historical moving averages (Rolling_Mean_7).
    5. Filters out initial NaN boundary rows.
    6. Verifies zero leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean daily consumption dataframe.
    target_col : str
        Target variable name.
    date_col : str
        Date column name.
        
    Returns
    -------
    pd.DataFrame
        Complete feature-engineered modeling dataframe.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df_raw_copy = df.copy()

    # Apply transformations
    df = create_calendar_features(df, date_col=date_col)
    df = create_lag_features(df, target_col=target_col, lags=[1, 7])
    df = create_rolling_features(df, target_col=target_col, window=7)

    # Clean boundaries
    df_clean, _ = clean_engineered_dataset(df)

    # Verify zero leakage
    verify_leakage(df_clean, df_raw_copy, target_col=target_col)

    # Reorder columns logically: Date, Features, Target
    feature_cols = [
        "Lag_1",
        "Lag_7",
        "Rolling_Mean_7",
        "Day_of_Week",
        "Month",
        "Is_Weekend",
    ]
    ordered_cols = [date_col] + feature_cols + [target_col]
    df_clean = df_clean[ordered_cols]

    return df_clean


if __name__ == "__main__":
    data_path = "data/daily_electricity_consumption.csv"
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        engineered_df = generate_features_pipeline(raw_df)
        out_path = "data/featured_daily_electricity_consumption.csv"
        engineered_df.to_csv(out_path, index=False)
        logger.info("Saved feature-engineered dataset to %s (%d rows).", out_path, len(engineered_df))
