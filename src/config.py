"""Paths and shared configuration for the Rapido project.

See PROJECT_IMPLEMENTATION_GUIDE.md for the reasoning behind these choices
(feature lists per model, why certain columns are excluded, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "rapido.sqlite3"

DATA_FILES = {
    "bookings": "bookings.csv",
    "customers": "customers.csv",
    "drivers": "drivers.csv",
    "location_demand": "location_demand.csv",
    "time_features": "time_features.csv",
}


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int = 42
    test_size: float = 0.2
    # Hours treated as "rush hour" for Rush_Hour_Flag (guide-requested feature).
    rush_hours: tuple[int, ...] = (7, 8, 9, 17, 18, 19)
    # Hours where location_demand/time_features show elevated demand (data-derived, see guide).
    peak_hours: tuple[int, ...] = (8, 9, 10, 17, 18, 19, 20)
    long_distance_km: float = 10.0


DEFAULT_CONFIG = ProjectConfig()

# --- Targets -----------------------------------------------------------
RIDE_OUTCOME_TARGET = "booking_status"
FARE_TARGET = "booking_value"
CUSTOMER_RISK_TARGET = "customer_cancel_flag"
DRIVER_RISK_TARGET = "driver_delay_flag"

# --- Feature lists -------------------------------------------------------
# Ride Outcome and Fare models only use signals known *before* a trip starts
# (city/vehicle/time/weather/traffic/distance-estimate/pricing inputs) --
# never actual_ride_time_min or incomplete_ride_reason, which are only known
# after the outcome (would leak the target).
RIDE_OUTCOME_CATEGORICAL = [
    "city",
    "vehicle_type",
    "traffic_level",
    "weather_condition",
    "day_of_week",
    "season",
]
RIDE_OUTCOME_NUMERIC = [
    "hour_of_day",
    "is_weekend",
    "ride_distance_km",
    "estimated_ride_time_min",
    "base_fare",
    "surge_multiplier",
    "rush_hour_flag",
    "long_distance_flag",
    "peak_time_flag",
    # Historical reliability of the specific customer/driver on this booking --
    # a legitimate feature (aggregated from *past* bookings, not this one's
    # outcome) that turned out to matter far more than trip conditions alone;
    # see PROJECT_IMPLEMENTATION_GUIDE.md section 6.4.
    "customer_cancellation_rate",
    "customer_avg_rating",
    "driver_delay_rate",
    "driver_acceptance_rate",
    "driver_avg_rating",
]

FARE_CATEGORICAL = ["city", "vehicle_type", "traffic_level", "weather_condition", "season"]
FARE_NUMERIC = [
    "ride_distance_km",
    "estimated_ride_time_min",
    "hour_of_day",
    "is_weekend",
    "rush_hour_flag",
    "long_distance_flag",
    "surge_multiplier",
]

# Customer/Driver risk models score the *entity* from lifetime aggregates in
# customers.csv / drivers.csv (see guide section 5.1 leakage note) -- not a
# specific upcoming booking.
CUSTOMER_RISK_CATEGORICAL = ["customer_gender", "customer_city", "preferred_vehicle_type"]
CUSTOMER_RISK_NUMERIC = [
    "customer_age",
    "customer_signup_days_ago",
    "total_bookings",
    "cancellation_rate",
    "avg_customer_rating",
]

DRIVER_RISK_CATEGORICAL = ["driver_city", "vehicle_type"]
DRIVER_RISK_NUMERIC = [
    "driver_age",
    "driver_experience_years",
    "total_assigned_rides",
    "acceptance_rate",
    "delay_rate",
    "avg_driver_rating",
    "avg_pickup_delay_min",
]
