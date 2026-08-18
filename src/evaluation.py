"""Evaluation helpers for project models.

Classification models (Ride Outcome, Customer Risk, Driver Risk) are scored
with accuracy/F1/AUC/confusion matrix; the regression model (Fare Prediction)
uses RMSE/MAE/R^2 -- these metric families are deliberately kept separate
(see PROJECT_IMPLEMENTATION_GUIDE.md section 6.3).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

CLASSIFICATION_ACCURACY_BENCHMARK = (0.85, 0.90)
REGRESSION_RMSE_BENCHMARK_PCT = 0.10


def evaluate_classification(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Return accuracy/F1/AUC/confusion-matrix metrics for a classifier."""

    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
                )
        except ValueError:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    low, high = CLASSIFICATION_ACCURACY_BENCHMARK
    metrics["meets_accuracy_benchmark"] = bool(low <= metrics["accuracy"])
    metrics["accuracy_benchmark_note"] = (
        f"Accuracy {metrics['accuracy']:.1%} vs the guide's {low:.0%}-{high:.0%} industry benchmark."
    )

    return metrics


def evaluate_regression(y_true, y_pred) -> dict:
    """Return RMSE/MAE/R^2 metrics for a regressor, plus a benchmark check."""

    rmse = float(root_mean_squared_error(y_true, y_pred))
    mean_actual = float(np.mean(y_true))
    rmse_pct_of_mean = (rmse / mean_actual) if mean_actual else None

    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_pct_of_mean": rmse_pct_of_mean,
        "meets_rmse_benchmark": bool(rmse_pct_of_mean is not None and rmse_pct_of_mean <= REGRESSION_RMSE_BENCHMARK_PCT),
        "rmse_benchmark_note": (
            f"RMSE is {rmse_pct_of_mean:.1%} of the mean fare vs the guide's "
            f"±{REGRESSION_RMSE_BENCHMARK_PCT:.0%} benchmark."
            if rmse_pct_of_mean is not None
            else "Unable to compute RMSE-as-percent-of-mean-fare."
        ),
    }
