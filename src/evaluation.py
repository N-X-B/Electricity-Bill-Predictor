"""
Model Evaluation and Error Diagnostics Module for Electricity Bill Predictor.
Singularity Research Lab – AI Division

This module provides modular, mathematically grounded evaluation functions:
1. Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2).
2. Performs out-of-time chronological error diagnostics and residual analysis.
3. Generates publication-grade evaluation plots saved directly to results/.
4. Exports machine-readable model comparison tables (CSV and JSON).
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics.
    
    Mathematical Definitions:
    -------------------------
    1. Mean Absolute Error (MAE):
       MAE = (1 / n) * sum_{i=1}^n |y_i - y_hat_i|
       Measures average magnitude of prediction errors in physical units (kWh),
       linear penalty without disproportionately weighting outliers.
       
    2. Root Mean Squared Error (RMSE):
       RMSE = sqrt( (1 / n) * sum_{i=1}^n (y_i - y_hat_i)^2 )
       Penalizes larger errors quadratically; sensitive to extreme peak spikes.
       
    3. Coefficient of Determination (R^2):
       R^2 = 1 - (sum_{i=1}^n (y_i - y_hat_i)^2) / (sum_{i=1}^n (y_i - y_bar)^2)
       Proportion of target variance explained by the model compared to baseline mean.
       
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth daily consumption values (kWh).
    y_pred : np.ndarray
        Model predicted daily consumption values (kWh).
        
    Returns
    -------
    Dict[str, float]
        Dictionary with 'MAE', 'RMSE', 'R2', and 'Max_Error'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(np.max(np.abs(y_true - y_pred)))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Max_Error": round(max_err, 4),
    }


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates an estimator on both train and test partitions to assess bias and variance.
    
    Parameters
    ----------
    model : Any
        Fitted Scikit-learn estimator.
    X_train, y_train : Features and target for training set.
    X_test, y_test : Features and target for test set.
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary of train and test metrics.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    return {
        "train": train_metrics,
        "test": test_metrics,
        "predictions": {
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
        },
    }


def build_comparison_table(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Builds a structured model comparison table across Train and Test sets.
    
    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary mapping model names to fitted estimators.
    X_train, y_train, X_test, y_test : Data splits.
    
    Returns
    -------
    pd.DataFrame
        Formatted comparison DataFrame.
    """
    rows = []
    for name, model in models_dict.items():
        eval_res = evaluate_model(model, X_train, y_train, X_test, y_test)
        train_m = eval_res["train"]
        test_m = eval_res["test"]

        rows.append({
            "Model": name,
            "Train_MAE": train_m["MAE"],
            "Test_MAE": test_m["MAE"],
            "Train_RMSE": train_m["RMSE"],
            "Test_RMSE": test_m["RMSE"],
            "Train_R2": train_m["R2"],
            "Test_R2": test_m["R2"],
            "Test_Max_Error": test_m["Max_Error"],
        })

    return pd.DataFrame(rows)


def save_comparison_table(
    comparison_df: pd.DataFrame,
    output_csv: str = "results/model_comparison.csv",
    output_json: str = "results/model_comparison.json",
) -> None:
    """
    Exports machine-readable model comparison tables to CSV and JSON formats.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    
    # Export to JSON
    json_records = comparison_df.to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=4)
        
    logger.info("Saved comparison tables to %s and %s", output_csv, output_json)


def analyze_top_errors(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Extracts and ranks the largest absolute prediction errors on the test set.
    """
    residuals = y_true.values - y_pred
    abs_errors = np.abs(residuals)

    df_err = pd.DataFrame({
        "Date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
        "Actual_kWh": y_true.values,
        "Predicted_kWh": y_pred,
        "Residual_kWh": residuals,
        "Absolute_Error_kWh": abs_errors,
        "Model": model_name,
    })

    return df_err.sort_values("Absolute_Error_kWh", ascending=False).head(n_top).reset_index(drop=True)


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred_lr: np.ndarray,
    y_pred_dt: np.ndarray,
    output_path: str = "results/eval_actual_vs_predicted.png",
) -> None:
    """
    Generates a dual-panel figure:
    1. Chronological time-series overlay of test actuals vs. model predictions.
    2. Parity scatter plots (Actual vs Predicted) with 1:1 ideal line.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    ax_ts = fig.add_subplot(gs[0, :])
    ax_sc_lr = fig.add_subplot(gs[1, 0])
    ax_sc_dt = fig.add_subplot(gs[1, 1])

    dates_dt = pd.to_datetime(dates)

    # Panel 1: Time Series Overlay
    ax_ts.plot(dates_dt, y_true, color="black", linewidth=1.5, label="Actual Consumption", alpha=0.85)
    ax_ts.plot(dates_dt, y_pred_lr, color="#1976d2", linewidth=1.3, linestyle="--", label="Linear Regression (OLS)")
    ax_ts.plot(dates_dt, y_pred_dt, color="#388e3c", linewidth=1.3, linestyle=":", label="Decision Tree Regressor")
    ax_ts.set_title("Actual vs. Predicted Daily Electricity Consumption (Test Horizon: Feb 2010 – Nov 2010)", fontsize=14, fontweight="bold", pad=10)
    ax_ts.set_xlabel("Date", fontsize=11)
    ax_ts.set_ylabel("Electricity Consumption (kWh)", fontsize=11)
    ax_ts.legend(loc="upper right", frameon=True)
    ax_ts.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Scatter Parity Plot for Linear Regression
    min_val = min(y_true.min(), y_pred_lr.min(), y_pred_dt.min()) - 2
    max_val = max(y_true.max(), y_pred_lr.max(), y_pred_dt.max()) + 2

    ax_sc_lr.scatter(y_true, y_pred_lr, alpha=0.45, color="#1976d2", s=25, edgecolors="none")
    ax_sc_lr.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="Ideal 1:1 Parity")
    ax_sc_lr.set_title("Linear Regression: Actual vs. Predicted", fontsize=12, fontweight="bold")
    ax_sc_lr.set_xlabel("Actual Consumption (kWh)", fontsize=11)
    ax_sc_lr.set_ylabel("Predicted Consumption (kWh)", fontsize=11)
    ax_sc_lr.set_xlim(min_val, max_val)
    ax_sc_lr.set_ylim(min_val, max_val)
    ax_sc_lr.legend(loc="upper left")
    ax_sc_lr.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Scatter Parity Plot for Decision Tree
    ax_sc_dt.scatter(y_true, y_pred_dt, alpha=0.45, color="#388e3c", s=25, edgecolors="none")
    ax_sc_dt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="Ideal 1:1 Parity")
    ax_sc_dt.set_title("Decision Tree: Actual vs. Predicted", fontsize=12, fontweight="bold")
    ax_sc_dt.set_xlabel("Actual Consumption (kWh)", fontsize=11)
    ax_sc_dt.set_ylabel("Predicted Consumption (kWh)", fontsize=11)
    ax_sc_dt.set_xlim(min_val, max_val)
    ax_sc_dt.set_ylim(min_val, max_val)
    ax_sc_dt.legend(loc="upper left")
    ax_sc_dt.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved actual vs predicted figure to %s", output_path)


def plot_residuals(
    y_true: pd.Series,
    y_pred_lr: np.ndarray,
    y_pred_dt: np.ndarray,
    output_path: str = "results/eval_residual_plots.png",
) -> None:
    """
    Generates a 2x2 grid of residual diagnostics:
    - Top: Residuals vs Predicted values (heteroscedasticity & non-linear bias check).
    - Bottom: Residual error distribution histograms with Gaussian KDE.
    """
    res_lr = y_true.values - y_pred_lr
    res_dt = y_true.values - y_pred_dt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Top-Left: LR Residuals vs Predicted
    axes[0, 0].scatter(y_pred_lr, res_lr, color="#1976d2", alpha=0.45, s=25)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Linear Regression: Residuals vs. Fitted", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Fitted Values (kWh)", fontsize=11)
    axes[0, 0].set_ylabel("Residual (Actual - Fitted, kWh)", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Top-Right: DT Residuals vs Predicted
    axes[0, 1].scatter(y_pred_dt, res_dt, color="#388e3c", alpha=0.45, s=25)
    axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("Decision Tree: Residuals vs. Fitted", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Fitted Values (kWh)", fontsize=11)
    axes[0, 1].set_ylabel("Residual (Actual - Fitted, kWh)", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Bottom-Left: LR Residual Distribution
    sns.histplot(res_lr, kde=True, color="#1976d2", ax=axes[1, 0], bins=30, alpha=0.6)
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 0].axvline(np.mean(res_lr), color="orange", linestyle=":", label=f"Mean: {np.mean(res_lr):.2f}")
    axes[1, 0].set_title("Linear Regression: Residual Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Residual (kWh)", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Bottom-Right: DT Residual Distribution
    sns.histplot(res_dt, kde=True, color="#388e3c", ax=axes[1, 1], bins=30, alpha=0.6)
    axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 1].axvline(np.mean(res_dt), color="orange", linestyle=":", label=f"Mean: {np.mean(res_dt):.2f}")
    axes[1, 1].set_title("Decision Tree: Residual Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Residual (kWh)", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved residual plots figure to %s", output_path)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = "results/eval_model_comparison.png",
) -> None:
    """
    Generates a grouped bar chart comparing MAE, RMSE, and R2 across Train and Test sets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = comparison_df["Model"].values
    x = np.arange(len(models))
    width = 0.35

    # 1. MAE Comparison
    axes[0].bar(x - width/2, comparison_df["Train_MAE"], width, label="Train MAE", color="#90caf9", edgecolor="black")
    axes[0].bar(x + width/2, comparison_df["Test_MAE"], width, label="Test MAE", color="#1976d2", edgecolor="black")
    axes[0].set_title("Mean Absolute Error (MAE - Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MAE (kWh)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    for i in range(len(models)):
        axes[0].text(i - width/2, comparison_df["Train_MAE"].iloc[i] + 0.1, f"{comparison_df['Train_MAE'].iloc[i]:.2f}", ha="center", fontsize=9)
        axes[0].text(i + width/2, comparison_df["Test_MAE"].iloc[i] + 0.1, f"{comparison_df['Test_MAE'].iloc[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    # 2. RMSE Comparison
    axes[1].bar(x - width/2, comparison_df["Train_RMSE"], width, label="Train RMSE", color="#ffcc80", edgecolor="black")
    axes[1].bar(x + width/2, comparison_df["Test_RMSE"], width, label="Test RMSE", color="#f57c00", edgecolor="black")
    axes[1].set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("RMSE (kWh)", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    for i in range(len(models)):
        axes[1].text(i - width/2, comparison_df["Train_RMSE"].iloc[i] + 0.15, f"{comparison_df['Train_RMSE'].iloc[i]:.2f}", ha="center", fontsize=9)
        axes[1].text(i + width/2, comparison_df["Test_RMSE"].iloc[i] + 0.15, f"{comparison_df['Test_RMSE'].iloc[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    # 3. R2 Comparison
    axes[2].bar(x - width/2, comparison_df["Train_R2"], width, label="Train R²", color="#a5d6a7", edgecolor="black")
    axes[2].bar(x + width/2, comparison_df["Test_R2"], width, label="Test R²", color="#388e3c", edgecolor="black")
    axes[2].set_title("R-squared (R² - Higher is Better)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Variance Explained (R²)", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    for i in range(len(models)):
        axes[2].text(i - width/2, comparison_df["Train_R2"].iloc[i] + 0.02, f"{comparison_df['Train_R2'].iloc[i]:.3f}", ha="center", fontsize=9)
        axes[2].text(i + width/2, comparison_df["Test_R2"].iloc[i] + 0.02, f"{comparison_df['Test_R2'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved model comparison figure to %s", output_path)
