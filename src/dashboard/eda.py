"""EDA & Trends page -- the exploratory visuals named in the project guide,
mapped onto the real dataset columns (see PROJECT_IMPLEMENTATION_GUIDE.md 8.2).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from . import palette

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render(bookings: pd.DataFrame, customers: pd.DataFrame, drivers: pd.DataFrame) -> None:
    st.subheader("EDA & Trends")
    if bookings.empty:
        st.warning("No bookings match the current filters.")
        return

    tabs = st.tabs(
        [
            "Ride volume", "Cancellations", "Fare & distance", "Ratings",
            "Customer vs driver", "Traffic & weather", "Vehicle usage",
        ]
    )

    with tabs[0]:
        st.markdown("**Ride volume by hour, weekday, and city**")
        col1, col2 = st.columns(2)
        with col1:
            by_hour = bookings.groupby("hour_of_day").size().reset_index(name="bookings")
            fig = px.bar(by_hour, x="hour_of_day", y="bookings", title="Bookings by hour of day")
            fig.update_traces(marker_color=palette.CATEGORICAL[0])
            st.plotly_chart(fig, width='stretch')
        with col2:
            by_day = (
                bookings.groupby("day_of_week").size().reindex(WEEKDAY_ORDER).reset_index(name="bookings")
            )
            fig = px.bar(by_day, x="day_of_week", y="bookings", title="Bookings by weekday")
            fig.update_traces(marker_color=palette.CATEGORICAL[0])
            st.plotly_chart(fig, width='stretch')

        volume_pivot = bookings.pivot_table(index="city", columns="hour_of_day", values="booking_id", aggfunc="count", fill_value=0)
        fig = px.imshow(
            volume_pivot, color_continuous_scale=palette.SEQUENTIAL_BLUE, aspect="auto",
            labels=dict(x="Hour of day", y="City", color="Bookings"), title="Ride volume: city x hour",
        )
        st.plotly_chart(fig, width='stretch')

    with tabs[1]:
        st.markdown("**Cancellation heatmap across cities**")
        cancel_pivot = (
            bookings.assign(is_cancelled=(bookings["booking_status"] == "Cancelled").astype(int))
            .pivot_table(index="city", columns="hour_of_day", values="is_cancelled", aggfunc="mean", fill_value=0)
        )
        fig = px.imshow(
            cancel_pivot, color_continuous_scale=palette.SEQUENTIAL_BLUE, aspect="auto",
            labels=dict(x="Hour of day", y="City", color="Cancellation rate"),
            title="Cancellation rate by city and hour",
        )
        st.plotly_chart(fig, width='stretch')

    with tabs[2]:
        st.markdown("**Distance vs fare correlation**")
        sample = bookings.sample(min(5000, len(bookings)), random_state=42)
        fig = px.scatter(
            sample, x="ride_distance_km", y="booking_value", color="vehicle_type",
            color_discrete_map=palette.VEHICLE_TYPE_COLORS, opacity=0.5,
            title="Ride distance vs booking value (sampled)",
        )
        st.plotly_chart(fig, width='stretch')
        corr = bookings["ride_distance_km"].corr(bookings["booking_value"])
        st.caption(f"Correlation coefficient: {corr:.2f}")

    with tabs[3]:
        st.markdown("**Rating distribution**")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(customers, x="avg_customer_rating", nbins=20, title="Customer rating distribution")
            fig.update_traces(marker_color=palette.CATEGORICAL[0])
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.histogram(drivers, x="avg_driver_rating", nbins=20, title="Driver rating distribution")
            fig.update_traces(marker_color=palette.CATEGORICAL[1])
            st.plotly_chart(fig, width='stretch')

    with tabs[4]:
        st.markdown("**Customer vs driver behavior comparison**")
        comparison = pd.DataFrame(
            {
                "group": ["Customers", "Drivers"],
                "rate": [customers["cancellation_rate"].mean(), drivers["delay_rate"].mean()],
                "metric": ["Avg cancellation rate", "Avg delay rate"],
            }
        )
        fig = px.bar(
            comparison, x="group", y="rate", color="group", text="metric",
            color_discrete_map={"Customers": palette.CATEGORICAL[0], "Drivers": palette.CATEGORICAL[1]},
            title="Customers' cancellation rate vs drivers' delay rate",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with tabs[5]:
        st.markdown("**Traffic & weather vs cancellation**")
        grouped = (
            bookings.assign(is_cancelled=(bookings["booking_status"] == "Cancelled").astype(int))
            .groupby(["traffic_level", "weather_condition"])["is_cancelled"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            grouped, x="traffic_level", y="is_cancelled", color="weather_condition", barmode="group",
            category_orders={"traffic_level": ["Low", "Medium", "High"]},
            color_discrete_map=palette.WEATHER_COLORS,
            title="Cancellation rate by traffic level and weather",
        )
        st.plotly_chart(fig, width='stretch')

    with tabs[6]:
        st.markdown(
            "**Vehicle type usage patterns.** The guide also asks for payment-method "
            "usage patterns, but this dataset has no `payment_method` column -- vehicle "
            "type usage is shown instead (see PROJECT_IMPLEMENTATION_GUIDE.md section 2.6)."
        )
        by_vehicle_city = bookings.groupby(["city", "vehicle_type"]).size().reset_index(name="bookings")
        fig = px.bar(
            by_vehicle_city, x="city", y="bookings", color="vehicle_type", barmode="stack",
            color_discrete_map=palette.VEHICLE_TYPE_COLORS, title="Vehicle type usage by city",
        )
        st.plotly_chart(fig, width='stretch')
