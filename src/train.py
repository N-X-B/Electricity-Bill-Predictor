"""
Model Training and Serialization Module for Daily Electricity Consumption Forecasting.
Singularity Research Lab – AI Division

This module provides modular, reproducible functions to:
1. Split feature-engineered data chronologically into training and testing partitions.
2. Train Model 1: Ordinary Least Squares (OLS) Linear Regression (Primary).
3. Train Model 2: Decision Tree Regressor with tree-depth regularization (Comparison).
4. Persist trained estimators in models/ using joblib.
5. Generate predictions on the holdout test set without data leakage.
"""

import os
import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Lag_1",
    "Lag_7",
    "Rolling_Mean_7",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
]
TARGET_COLUMN = "Daily_Energy_kWh"


def split_chronological(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    date_col: str = "Date",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Performs a strictly chronological train-test split without data shuffling.
    
    The earlier (1 - test_size) portion of the time series is assigned to training;
    the subsequent test_size portion is reserved for out-of-time evaluation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted feature-engineered dataframe.
    feature_cols : List[str]
        List of input feature column names.
    target_col : str
        Target variable name.
    date_col : str
        Date column name.
    test_size : float
        Fraction of observations to allocate to testing (default 0.2 = 20%).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing X_train, X_test, y_train, y_test, train_dates, test_dates, and metadata.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    n_total = len(df)
    split_idx = int(n_total * (1 - test_size))

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    split_info = {
        "n_total": n_total,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_ratio": len(train_df) / n_total,
        "test_ratio": len(test_df) / n_total,
        "train_start_date": str(train_df[date_col].min().date()),
        "train_end_date": str(train_df[date_col].max().date()),
        "test_start_date": str(test_df[date_col].min().date()),
        "test_end_date": str(test_df[date_col].max().date()),
    }

    logger.info(
        "Chronological split complete: Train=%d days (%s to %s) | Test=%d days (%s to %s).",
        split_info["n_train"],
        split_info["train_start_date"],
        split_info["train_end_date"],
        split_info["n_test"],
        split_info["test_start_date"],
        split_info["test_end_date"],
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_dates": train_df[date_col],
        "test_dates": test_df[date_col],
        "split_info": split_info,
    }


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fit_intercept: bool = True,
) -> LinearRegression:
    """
    Fits an Ordinary Least Squares (OLS) Linear Regression model.
    
    Mathematical Formulation:
    -------------------------
    Hypothesis:
        y_hat = w_0 + sum_{j=1}^p w_j * x_j = X * w
    Objective:
        min_w || y - X * w ||_2^2
    Analytical Solution (Normal Equation):
        w* = (X^T * X)^{-1} * X^T * y
        
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    fit_intercept : bool
        Whether to calculate intercept w_0 (default True).
        
    Returns
    -------
    LinearRegression
        Fitted linear regression model.
    """
    logger.info("Fitting Linear Regression model (OLS)...")
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    logger.info("Linear Regression fitted successfully. Intercept: %.4f", model.intercept_)
    return model


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """
    Fits a regularized Decision Tree Regressor.
    
    Mathematical Formulation:
    -------------------------
    Partitions feature space into M disjoint hyper-rectangles R_1, ..., R_M.
    Prediction in region R_m:
        y_hat = (1 / |R_m|) * sum_{i in R_m} y_i
    Splitting Criterion:
        Greedy minimization of mean squared error (variance reduction):
        Cost(j, s) = sum_{i in R_1(j,s)} (y_i - y_bar_R1)^2 + sum_{i in R_2(j,s)} (y_i - y_bar_R2)^2
        
    Regularization Hyperparameters:
    -------------------------------
    max_depth (5): Restricts tree depth to control variance and prevent overfitting.
    min_samples_split (10): Requires >=10 samples to consider an internal split.
    min_samples_leaf (5): Requires >=5 samples per terminal leaf node for robust estimation.
    random_state (42): Deterministic split selection when feature ties occur.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    max_depth : int
        Maximum tree depth.
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required at a terminal leaf node.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    DecisionTreeRegressor
        Fitted decision tree estimator.
    """
    logger.info(
        "Fitting Decision Tree Regressor (max_depth=%d, min_samples_split=%d, min_samples_leaf=%d)...",
        max_depth,
        min_samples_split,
        min_samples_leaf,
    )
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("Decision Tree Regressor fitted successfully.")
    return model


def save_model(model: Any, filepath: str) -> None:
    """
    Serializes a trained model artifact using joblib.
    
    Parameters
    ----------
    model : Any
        Fitted Scikit-learn estimator.
    filepath : str
        Target file path (.joblib).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Saved model artifact to %s", filepath)


def load_model(filepath: str) -> Any:
    """
    Loads a serialized model artifact from disk.
    
    Parameters
    ----------
    filepath : str
        Path to .joblib file.
        
    Returns
    -------
    Any
        Loaded Scikit-learn estimator.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def generate_predictions(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Generates point predictions for a feature matrix.
    
    Parameters
    ----------
    model : Any
        Fitted estimator with predict() method.
    X : pd.DataFrame
        Feature matrix.
        
    Returns
    -------
    np.ndarray
        Predicted target values.
    """
    return model.predict(X)


def run_training_pipeline(
    data_path: str = "data/featured_daily_electricity_consumption.csv",
    models_dir: str = "models",
    results_dir: str = "results",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Orchestrates chronological splitting, model fitting, serialization, and test predictions.
    
    Parameters
    ----------
    data_path : str
        Path to feature-engineered dataset.
    models_dir : str
        Directory to save .joblib models.
    results_dir : str
        Directory to save prediction tables.
    test_size : float
        Holdout test ratio.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing trained models, predictions, and split info.
    """
    df = pd.read_csv(data_path)
    split_data = split_chronological(df, test_size=test_size)

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    test_dates = split_data["test_dates"]

    # 1. Fit Linear Regression
    lr_model = train_linear_regression(X_train, y_train, fit_intercept=True)
    save_model(lr_model, os.path.join(models_dir, "linear_regression.joblib"))

    # 2. Fit Decision Tree Regressor
    dt_model = train_decision_tree(
        X_train,
        y_train,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    save_model(dt_model, os.path.join(models_dir, "decision_tree.joblib"))

    # 3. Generate test predictions
    y_pred_lr = generate_predictions(lr_model, X_test)
    y_pred_dt = generate_predictions(dt_model, X_test)

    # Save test predictions for evaluation
    pred_df = pd.DataFrame({
        "Date": test_dates.dt.strftime("%Y-%m-%d"),
        "Actual_Daily_Energy_kWh": y_test.values,
        "Pred_Linear_Regression": y_pred_lr,
        "Pred_Decision_Tree": y_pred_dt,
    })
    os.makedirs(results_dir, exist_ok=True)
    pred_path = os.path.join(results_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved test predictions table to %s (%d rows).", pred_path, len(pred_df))

    return {
        "lr_model": lr_model,
        "dt_model": dt_model,
        "pred_df": pred_df,
        "split_info": split_data["split_info"],
    }


if __name__ == "__main__":
    run_training_pipeline()
