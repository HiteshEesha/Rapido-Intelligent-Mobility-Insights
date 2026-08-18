"""Data cleaning for the Rapido datasets.

Cleaning intentionally runs *after* validation (src/validation.py) so issues
are reported before they're silently fixed. See
PROJECT_IMPLEMENTATION_GUIDE.md section 4.5 for the missing-value strategy.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def clean_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
    """Parse booking datetimes and fill only genuinely-missing values.

    `actual_ride_time_min` is null exactly when a ride wasn't completed --
    that's expected structure, not missing data, so it is left as NaN rather
    than imputed (imputing it would fabricate a duration for rides that never
    finished). `incomplete_ride_reason` is left null for the same reason.
    """

    cleaned = bookings.copy()

    cleaned["booking_date"] = pd.to_datetime(cleaned["booking_date"], errors="coerce")
    cleaned["booking_datetime"] = pd.to_datetime(
        cleaned["booking_date"].dt.strftime("%Y-%m-%d") + " " + cleaned["booking_time"].astype(str),
        errors="coerce",
    )

    protected_columns = {"actual_ride_time_min", "incomplete_ride_reason"}
    for column in cleaned.columns:
        if column in protected_columns:
            continue
        if is_numeric_dtype(cleaned[column]):
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def _fill_generic(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if is_numeric_dtype(cleaned[column]):
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            if cleaned[column].isna().any():
                cleaned[column] = cleaned[column].fillna("Unknown")
    return cleaned


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing values in the customer aggregates table."""

    return _fill_generic(customers)


def clean_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing values in the driver aggregates table."""

    return _fill_generic(drivers)


def parse_datetime_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to datetime when they are present."""

    parsed = frame.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed
