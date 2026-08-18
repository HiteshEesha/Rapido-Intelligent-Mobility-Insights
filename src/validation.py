"""Data quality validation for the Rapido datasets.

Implements the checks documented in PROJECT_IMPLEMENTATION_GUIDE.md section 4:
structural, referential integrity, range/domain, and business-rule checks.
Checks are non-fatal by design -- they report issues so they can be surfaced
in the Streamlit "data quality" view, rather than crashing the pipeline.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

REQUIRED_COLUMNS = {
    "bookings": [
        "booking_id", "booking_date", "booking_time", "day_of_week", "is_weekend",
        "hour_of_day", "city", "pickup_location", "drop_location", "vehicle_type",
        "ride_distance_km", "estimated_ride_time_min", "actual_ride_time_min",
        "traffic_level", "weather_condition", "base_fare", "surge_multiplier",
        "booking_value", "booking_status", "incomplete_ride_reason",
        "customer_id", "driver_id",
    ],
    "customers": [
        "customer_id", "customer_gender", "customer_age", "customer_city",
        "customer_signup_days_ago", "preferred_vehicle_type", "total_bookings",
        "completed_rides", "cancelled_rides", "incomplete_rides",
        "cancellation_rate", "avg_customer_rating", "customer_cancel_flag",
    ],
    "drivers": [
        "driver_id", "driver_age", "driver_city", "vehicle_type",
        "driver_experience_years", "total_assigned_rides", "accepted_rides",
        "incomplete_rides", "delay_count", "acceptance_rate", "delay_rate",
        "avg_driver_rating", "avg_pickup_delay_min", "driver_delay_flag",
    ],
    "location_demand": [
        "city", "pickup_location", "hour_of_day", "vehicle_type", "total_requests",
        "completed_rides", "cancelled_rides", "avg_wait_time_min",
        "avg_surge_multiplier", "demand_level",
    ],
    "time_features": [
        "datetime", "hour_of_day", "day_of_week", "is_weekend", "is_holiday",
        "peak_time_flag", "season",
    ],
}


def _add(report: list[dict], check: str, passed: bool, detail: str, severity: str = "error") -> None:
    report.append({"check": check, "passed": bool(passed), "detail": detail, "severity": severity})


def check_required_columns(name: str, df: pd.DataFrame, report: list[dict]) -> None:
    expected = REQUIRED_COLUMNS.get(name, [])
    missing = [c for c in expected if c not in df.columns]
    _add(
        report,
        f"{name}: required columns present",
        not missing,
        "all expected columns found" if not missing else f"missing columns: {missing}",
    )


def check_primary_key_unique(name: str, df: pd.DataFrame, key: str, report: list[dict]) -> None:
    if key not in df.columns:
        return
    duplicate_count = int(df[key].duplicated().sum())
    _add(
        report,
        f"{name}: {key} is unique",
        duplicate_count == 0,
        "no duplicates" if duplicate_count == 0 else f"{duplicate_count} duplicate {key} values",
    )


def check_referential_integrity(bookings: pd.DataFrame, customers: pd.DataFrame, drivers: pd.DataFrame, report: list[dict]) -> None:
    if {"customer_id"}.issubset(bookings.columns) and "customer_id" in customers.columns:
        orphaned = int((~bookings["customer_id"].isin(customers["customer_id"])).sum())
        _add(report, "bookings.customer_id -> customers.customer_id", orphaned == 0,
             "all customer_id values resolve" if orphaned == 0 else f"{orphaned} bookings reference unknown customers")

    if {"driver_id"}.issubset(bookings.columns) and "driver_id" in drivers.columns:
        orphaned = int((~bookings["driver_id"].isin(drivers["driver_id"])).sum())
        _add(report, "bookings.driver_id -> drivers.driver_id", orphaned == 0,
             "all driver_id values resolve" if orphaned == 0 else f"{orphaned} bookings reference unknown drivers")


def check_ranges(bookings: pd.DataFrame, report: list[dict]) -> None:
    range_checks = [
        ("ride_distance_km", 0, None),
        ("base_fare", 0, None),
        ("booking_value", 0, None),
        ("surge_multiplier", 1.0, None),
        ("hour_of_day", 0, 23),
    ]
    for column, low, high in range_checks:
        if column not in bookings.columns:
            continue
        series = bookings[column]
        if not is_numeric_dtype(series):
            continue
        below = series.lt(low).sum() if low is not None else 0
        above = series.gt(high).sum() if high is not None else 0
        violations = int(below + above)
        _add(report, f"bookings.{column} within expected range", violations == 0,
             "within range" if violations == 0 else f"{violations} rows out of range")


def check_rate_columns_0_1(name: str, df: pd.DataFrame, columns: list[str], report: list[dict]) -> None:
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        violations = int(((series < 0) | (series > 1)).sum())
        _add(report, f"{name}.{column} in [0, 1]", violations == 0,
             "within range" if violations == 0 else f"{violations} rows out of range")


def check_rating_columns(name: str, df: pd.DataFrame, columns: list[str], report: list[dict]) -> None:
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column]
        violations = int(((series < 1) | (series > 5)).sum())
        _add(report, f"{name}.{column} in [1, 5]", violations == 0,
             "within range" if violations == 0 else f"{violations} rows out of range")


def check_business_rules(bookings: pd.DataFrame, report: list[dict]) -> None:
    if {"booking_status", "actual_ride_time_min"}.issubset(bookings.columns):
        completed = bookings["booking_status"] == "Completed"
        violations = int(((completed) & bookings["actual_ride_time_min"].isna()).sum())
        violations += int(((~completed) & bookings["actual_ride_time_min"].notna()).sum())
        _add(report, "actual_ride_time_min populated iff Completed", violations == 0,
             "consistent" if violations == 0 else f"{violations} inconsistent rows")

    if {"booking_status", "incomplete_ride_reason"}.issubset(bookings.columns):
        incomplete = bookings["booking_status"] == "Incomplete"
        violations = int(((incomplete) & bookings["incomplete_ride_reason"].isna()).sum())
        violations += int(((~incomplete) & bookings["incomplete_ride_reason"].notna()).sum())
        _add(report, "incomplete_ride_reason populated iff Incomplete", violations == 0,
             "consistent" if violations == 0 else f"{violations} inconsistent rows")


def run_validation(datasets: dict[str, pd.DataFrame]) -> list[dict]:
    """Run all validation checks and return a flat report (list of check results)."""

    report: list[dict] = []
    bookings = datasets.get("bookings", pd.DataFrame())
    customers = datasets.get("customers", pd.DataFrame())
    drivers = datasets.get("drivers", pd.DataFrame())

    for name, df in datasets.items():
        check_required_columns(name, df, report)

    check_primary_key_unique("bookings", bookings, "booking_id", report)
    check_primary_key_unique("customers", customers, "customer_id", report)
    check_primary_key_unique("drivers", drivers, "driver_id", report)

    check_referential_integrity(bookings, customers, drivers, report)
    check_ranges(bookings, report)
    check_rate_columns_0_1("customers", customers, ["cancellation_rate"], report)
    check_rate_columns_0_1("drivers", drivers, ["acceptance_rate", "delay_rate"], report)
    check_rating_columns("customers", customers, ["avg_customer_rating"], report)
    check_rating_columns("drivers", drivers, ["avg_driver_rating"], report)
    check_business_rules(bookings, report)

    return report
