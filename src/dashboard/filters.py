"""Sidebar filters shared across dashboard pages.

Widgets use stable `key=` values so Streamlit's own session state keeps the
selection intact as the user switches pages (each page re-renders the same
sidebar widgets on rerun).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_sidebar_filters(bookings: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    cities = sorted(bookings["city"].unique())
    selected_cities = st.sidebar.multiselect("City", cities, default=cities, key="filter_city")

    vehicle_types = sorted(bookings["vehicle_type"].unique())
    selected_vehicles = st.sidebar.multiselect("Vehicle type", vehicle_types, default=vehicle_types, key="filter_vehicle")

    statuses = sorted(bookings["booking_status"].unique())
    selected_statuses = st.sidebar.multiselect("Booking status", statuses, default=statuses, key="filter_status")

    min_date = bookings["booking_date"].min().date()
    max_date = bookings["booking_date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="filter_dates"
    )

    filtered = bookings[
        bookings["city"].isin(selected_cities)
        & bookings["vehicle_type"].isin(selected_vehicles)
        & bookings["booking_status"].isin(selected_statuses)
    ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["booking_date"].dt.date >= start) & (filtered["booking_date"].dt.date <= end)
        ]

    st.sidebar.caption(f"{len(filtered):,} of {len(bookings):,} bookings match these filters.")
    return filtered
