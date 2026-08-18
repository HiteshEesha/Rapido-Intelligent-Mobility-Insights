"""Feature engineering for ride analytics.

Maps the guide's requested features (Fare_per_KM, Rush_Hour_Flag, City_Pair,
Driver_Reliability_Score, Customer_Loyalty_Score, ...) onto the real dataset
columns. See PROJECT_IMPLEMENTATION_GUIDE.md section 5 for the reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG


def add_season_feature(bookings: pd.DataFrame, time_features: pd.DataFrame) -> pd.DataFrame:
    """Join `season` from time_features by calendar date.

    `is_holiday` is constant (always 0 in this dataset) and `peak_time_flag`
    is a deterministic function of hour_of_day alone -- both are computed
    directly in add_core_booking_features instead of merged, so this join
    only needs to bring in `season` (one row per date, not per hour).
    """

    if "datetime" not in time_features.columns or "season" not in time_features.columns:
        result = bookings.copy()
        result["season"] = "Unknown"
        return result

    daily_season = (
        time_features.assign(date=pd.to_datetime(time_features["datetime"]).dt.date)
        .drop_duplicates(subset="date")[["date", "season"]]
    )

    merged = bookings.copy()
    merged["_date"] = merged["booking_date"].dt.date
    merged = merged.merge(daily_season, left_on="_date", right_on="date", how="left")
    merged["season"] = merged["season"].fillna("Unknown")
    merged = merged.drop(columns=["_date", "date"])
    return merged


def add_core_booking_features(bookings: pd.DataFrame, config=DEFAULT_CONFIG) -> pd.DataFrame:
    """Create the guide's requested per-booking features."""

    features = bookings.copy()

    features["fare_per_km"] = features["booking_value"] / features["ride_distance_km"].replace(0, np.nan)
    features["fare_per_min"] = features["booking_value"] / features["actual_ride_time_min"].replace(0, np.nan)

    features["rush_hour_flag"] = features["hour_of_day"].isin(config.rush_hours).astype(int)
    features["peak_time_flag"] = features["hour_of_day"].isin(config.peak_hours).astype(int)
    features["long_distance_flag"] = (features["ride_distance_km"] >= config.long_distance_km).astype(int)

    features["city_pair"] = (
        features["city"].astype(str) + ": " + features["pickup_location"].astype(str)
        + " -> " + features["drop_location"].astype(str)
    )

    features["ride_completed_flag"] = (features["booking_status"] == "Completed").astype(int)
    features["is_cancelled_flag"] = (features["booking_status"] == "Cancelled").astype(int)
    features["is_incomplete_flag"] = (features["booking_status"] == "Incomplete").astype(int)

    return features


def add_driver_reliability_score(drivers: pd.DataFrame) -> pd.DataFrame:
    """Combine acceptance/delay/rating into a single 0-1 reliability score."""

    scored = drivers.copy()
    scored["driver_reliability_score"] = (
        0.4 * scored["acceptance_rate"]
        + 0.4 * (1 - scored["delay_rate"])
        + 0.2 * (scored["avg_driver_rating"] / 5.0)
    ).clip(0, 1)
    return scored


def add_customer_loyalty_score(customers: pd.DataFrame) -> pd.DataFrame:
    """Combine tenure, volume, rating, and cancellation history into a 0-1 loyalty score."""

    scored = customers.copy()
    tenure_norm = (scored["customer_signup_days_ago"] / scored["customer_signup_days_ago"].max()).clip(0, 1)
    volume_norm = (scored["total_bookings"] / scored["total_bookings"].max()).clip(0, 1)
    scored["customer_loyalty_score"] = (
        0.3 * tenure_norm
        + 0.3 * volume_norm
        + 0.2 * (scored["avg_customer_rating"] / 5.0)
        + 0.2 * (1 - scored["cancellation_rate"])
    ).clip(0, 1)
    return scored


def add_customer_driver_signals(bookings: pd.DataFrame, customers: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Join each booking's customer/driver historical reliability signals.

    These are lifetime aggregates computed across a customer/driver's *other*
    bookings too, not derived from this booking's own outcome -- using them as
    Ride Outcome features is a legitimate historical-reliability signal, not
    target leakage. See PROJECT_IMPLEMENTATION_GUIDE.md section 6.4: this was
    added after finding trip-only features plateau around the majority-class
    baseline (~68% accuracy) for this dataset.
    """

    customer_cols = customers[["customer_id", "cancellation_rate", "avg_customer_rating"]].rename(
        columns={"cancellation_rate": "customer_cancellation_rate", "avg_customer_rating": "customer_avg_rating"}
    )
    driver_cols = drivers[["driver_id", "delay_rate", "acceptance_rate", "avg_driver_rating"]].rename(
        columns={"delay_rate": "driver_delay_rate", "acceptance_rate": "driver_acceptance_rate", "avg_driver_rating": "driver_avg_rating"}
    )

    merged = bookings.merge(customer_cols, on="customer_id", how="left").merge(driver_cols, on="driver_id", how="left")
    return merged


def build_booking_features(
    bookings: pd.DataFrame, time_features: pd.DataFrame, customers: pd.DataFrame, drivers: pd.DataFrame, config=DEFAULT_CONFIG
) -> pd.DataFrame:
    """Full booking-level feature pipeline: season join + core features + customer/driver signals."""

    with_season = add_season_feature(bookings, time_features)
    with_core = add_core_booking_features(with_season, config=config)
    return add_customer_driver_signals(with_core, customers, drivers)
