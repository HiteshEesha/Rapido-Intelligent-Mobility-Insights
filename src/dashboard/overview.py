"""Overview page: top-line KPIs and headline trends."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from . import palette


def render(bookings: pd.DataFrame) -> None:
    st.subheader("Overview")

    total = len(bookings)
    if total == 0:
        st.warning("No bookings match the current filters.")
        return

    completed = int((bookings["booking_status"] == "Completed").sum())
    cancelled = int((bookings["booking_status"] == "Cancelled").sum())
    incomplete = int((bookings["booking_status"] == "Incomplete").sum())

    cols = st.columns(5)
    cols[0].metric("Total bookings", f"{total:,}")
    cols[1].metric("Completion rate", f"{completed / total:.1%}")
    cols[2].metric("Cancellation rate", f"{cancelled / total:.1%}")
    cols[3].metric("Incomplete rate", f"{incomplete / total:.1%}")
    cols[4].metric("Avg booking value", f"Rs {bookings['booking_value'].mean():,.0f}")

    st.divider()

    daily = (
        bookings.assign(date=bookings["booking_date"].dt.date)
        .groupby(["date", "booking_status"])
        .size()
        .reset_index(name="bookings")
    )
    fig = px.line(
        daily,
        x="date",
        y="bookings",
        color="booking_status",
        color_discrete_map=palette.BOOKING_STATUS_COLORS,
        title="Daily bookings by outcome",
    )
    fig.update_layout(legend_title_text="Outcome")
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        by_city = (
            bookings.groupby("city").size().reset_index(name="bookings").sort_values("bookings", ascending=False)
        )
        fig_city = px.bar(
            by_city, x="city", y="bookings", color="city",
            color_discrete_map=palette.CITY_COLORS, title="Bookings by city",
        )
        fig_city.update_layout(showlegend=False)
        st.plotly_chart(fig_city, width='stretch')
    with col_b:
        by_vehicle = bookings.groupby("vehicle_type").size().reset_index(name="bookings")
        fig_vehicle = px.bar(
            by_vehicle, x="vehicle_type", y="bookings", color="vehicle_type",
            color_discrete_map=palette.VEHICLE_TYPE_COLORS, title="Bookings by vehicle type",
        )
        fig_vehicle.update_layout(showlegend=False)
        st.plotly_chart(fig_vehicle, width='stretch')
