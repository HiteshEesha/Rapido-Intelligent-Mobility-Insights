"""SQL persistence for the cleaned Rapido datasets.

Uses a local SQLite file by default (zero setup, matches `sql/schema.sql`'s
normalized structure) via SQLAlchemy, so it can be pointed at MySQL later by
passing a different connection string -- see sql/schema.sql for the
normalized reference DDL (bookings/customers/drivers/locations/time_features).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, create_engine

from .config import DB_PATH


def get_engine(db_path=DB_PATH) -> Engine:
    return create_engine(f"sqlite:///{db_path}")


def build_locations_dim(bookings: pd.DataFrame) -> pd.DataFrame:
    """Normalize the repeated (city, location_code) pairs into a lookup dimension."""

    pickup = bookings[["city", "pickup_location"]].rename(columns={"pickup_location": "location_code"})
    drop = bookings[["city", "drop_location"]].rename(columns={"drop_location": "location_code"})
    locations = pd.concat([pickup, drop], ignore_index=True).drop_duplicates().reset_index(drop=True)
    locations.insert(0, "location_id", locations.index + 1)
    return locations


def persist_datasets(datasets: dict[str, pd.DataFrame], engine: Engine | None = None) -> None:
    """Write the cleaned datasets to SQL as normalized tables."""

    engine = engine or get_engine()

    bookings = datasets["bookings"]
    locations = build_locations_dim(bookings)

    bookings_with_ids = bookings.merge(
        locations.rename(columns={"location_code": "pickup_location", "location_id": "pickup_location_id"}),
        on=["city", "pickup_location"],
        how="left",
    ).merge(
        locations.rename(columns={"location_code": "drop_location", "location_id": "drop_location_id"}),
        on=["city", "drop_location"],
        how="left",
    )

    locations[["location_id", "city", "location_code"]].to_sql("locations", engine, if_exists="replace", index=False)
    bookings_with_ids.drop(columns=["pickup_location", "drop_location"]).to_sql(
        "bookings", engine, if_exists="replace", index=False
    )
    datasets["customers"].to_sql("customers", engine, if_exists="replace", index=False)
    datasets["drivers"].to_sql("drivers", engine, if_exists="replace", index=False)
    datasets["time_features"].to_sql("time_features", engine, if_exists="replace", index=False)
    datasets["location_demand"].to_sql("location_demand", engine, if_exists="replace", index=False)
