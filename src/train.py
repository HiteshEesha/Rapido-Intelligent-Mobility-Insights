"""End-to-end training script.

Run with:  python -m src.train

Loads the CSVs, validates + cleans them, engineers features, trains all four
models, evaluates them against the guide's benchmarks, persists the trained
pipelines + metrics to models/, and loads the cleaned tables into SQLite.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .config import (
    CUSTOMER_RISK_CATEGORICAL,
    CUSTOMER_RISK_NUMERIC,
    CUSTOMER_RISK_TARGET,
    DATA_DIR,
    DRIVER_RISK_CATEGORICAL,
    DRIVER_RISK_NUMERIC,
    DRIVER_RISK_TARGET,
    FARE_CATEGORICAL,
    FARE_NUMERIC,
    FARE_TARGET,
    MODELS_DIR,
    RIDE_OUTCOME_CATEGORICAL,
    RIDE_OUTCOME_NUMERIC,
    RIDE_OUTCOME_TARGET,
)
from .data_loader import load_csv_files
from .db import persist_datasets
from .evaluation import evaluate_classification, evaluate_regression
from .feature_engineering import add_customer_loyalty_score, add_driver_reliability_score, build_booking_features
from .modeling import train_binary_risk_model, train_fare_model, train_ride_outcome_model
from .preprocessing import clean_bookings, clean_customers, clean_drivers
from .validation import run_validation


def prepare_datasets(data_dir: Path = DATA_DIR) -> dict:
    """Load, validate, clean, and feature-engineer all datasets."""

    datasets = load_csv_files(data_dir)
    validation_report = run_validation(datasets)

    bookings = clean_bookings(datasets["bookings"])
    customers = clean_customers(datasets["customers"])
    drivers = clean_drivers(datasets["drivers"])

    drivers = add_driver_reliability_score(drivers)
    customers = add_customer_loyalty_score(customers)
    bookings = build_booking_features(bookings, datasets["time_features"], customers, drivers)

    return {
        "bookings": bookings,
        "customers": customers,
        "drivers": drivers,
        "location_demand": datasets["location_demand"],
        "time_features": datasets["time_features"],
        "validation_report": validation_report,
    }


def run_full_pipeline(data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR, persist_db: bool = True) -> dict:
    prepared = prepare_datasets(data_dir)
    bookings, customers, drivers = prepared["bookings"], prepared["customers"], prepared["drivers"]

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    ride_outcome = train_ride_outcome_model(
        bookings, RIDE_OUTCOME_CATEGORICAL, RIDE_OUTCOME_NUMERIC, RIDE_OUTCOME_TARGET
    )
    metrics["ride_outcome"] = {
        **evaluate_classification(ride_outcome.y_test, ride_outcome.y_pred, ride_outcome.y_proba, labels=sorted(bookings[RIDE_OUTCOME_TARGET].unique())),
        "best_params": ride_outcome.best_params,
    }
    joblib.dump(ride_outcome.pipeline, models_dir / "ride_outcome_model.joblib")

    fare = train_fare_model(bookings, FARE_CATEGORICAL, FARE_NUMERIC, FARE_TARGET)
    metrics["fare"] = {**evaluate_regression(fare.y_test, fare.y_pred), "best_params": fare.best_params}
    joblib.dump(fare.pipeline, models_dir / "fare_model.joblib")

    customer_risk = train_binary_risk_model(
        customers, CUSTOMER_RISK_CATEGORICAL, CUSTOMER_RISK_NUMERIC, CUSTOMER_RISK_TARGET, name="customer_risk"
    )
    metrics["customer_risk"] = {
        **evaluate_classification(customer_risk.y_test, customer_risk.y_pred, customer_risk.y_proba, labels=[0, 1]),
        "best_params": customer_risk.best_params,
    }
    joblib.dump(customer_risk.pipeline, models_dir / "customer_risk_model.joblib")

    driver_risk = train_binary_risk_model(
        drivers, DRIVER_RISK_CATEGORICAL, DRIVER_RISK_NUMERIC, DRIVER_RISK_TARGET, name="driver_risk"
    )
    metrics["driver_risk"] = {
        **evaluate_classification(driver_risk.y_test, driver_risk.y_pred, driver_risk.y_proba, labels=[0, 1]),
        "best_params": driver_risk.best_params,
    }
    joblib.dump(driver_risk.pipeline, models_dir / "driver_risk_model.joblib")

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    if persist_db:
        persist_datasets(
            {
                "bookings": bookings,
                "customers": customers,
                "drivers": drivers,
                "time_features": prepared["time_features"],
                "location_demand": prepared["location_demand"],
            }
        )

    return {"validation_report": prepared["validation_report"], "metrics": metrics}


if __name__ == "__main__":
    result = run_full_pipeline()
    print(json.dumps(result["metrics"], indent=2, default=str))
