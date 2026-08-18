"""Trip Predictor page: Ride Outcome (multi-class) + Fare (regression) on one
shared trip-scenario form -- both models draw from the same booking-level
inputs, so one form avoids asking the user to enter the same trip twice.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import DEFAULT_CONFIG
from . import palette

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]
VEHICLE_TYPES = ["Bike", "Auto", "Cab"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Heavy Rain"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASONS = ["Winter", "Summer", "Monsoon"]


def _build_scenario_row(
    city, vehicle_type, traffic_level, weather_condition, day_of_week, season,
    hour_of_day, ride_distance_km, estimated_ride_time_min, base_fare, surge_multiplier,
    customer_cancellation_rate, customer_avg_rating,
    driver_delay_rate, driver_acceptance_rate, driver_avg_rating,
    config=DEFAULT_CONFIG,
) -> pd.DataFrame:
    is_weekend = int(day_of_week in {"Saturday", "Sunday"})
    rush_hour_flag = int(hour_of_day in config.rush_hours)
    peak_time_flag = int(hour_of_day in config.peak_hours)
    long_distance_flag = int(ride_distance_km >= config.long_distance_km)

    return pd.DataFrame(
        [
            {
                "city": city,
                "vehicle_type": vehicle_type,
                "traffic_level": traffic_level,
                "weather_condition": weather_condition,
                "day_of_week": day_of_week,
                "season": season,
                "hour_of_day": hour_of_day,
                "is_weekend": is_weekend,
                "ride_distance_km": ride_distance_km,
                "estimated_ride_time_min": estimated_ride_time_min,
                "base_fare": base_fare,
                "surge_multiplier": surge_multiplier,
                "rush_hour_flag": rush_hour_flag,
                "peak_time_flag": peak_time_flag,
                "long_distance_flag": long_distance_flag,
                "customer_cancellation_rate": customer_cancellation_rate,
                "customer_avg_rating": customer_avg_rating,
                "driver_delay_rate": driver_delay_rate,
                "driver_acceptance_rate": driver_acceptance_rate,
                "driver_avg_rating": driver_avg_rating,
            }
        ]
    )


def render(ride_outcome_model, fare_model, fare_rmse: float | None) -> None:
    st.subheader("Trip Predictor")
    st.caption(
        "Enter a hypothetical trip's details to predict its outcome and estimated fare "
        "before the ride starts -- exactly the 'before trip start' scenario the guide describes."
    )

    if ride_outcome_model is None or fare_model is None:
        st.info("Models aren't trained yet. Go to **Model Monitoring** and click **Train / retrain all models**.")
        return

    with st.form("trip_scenario_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City", CITIES)
            vehicle_type = st.selectbox("Vehicle type", VEHICLE_TYPES)
            day_of_week = st.selectbox("Day of week", WEEKDAYS)
        with col2:
            traffic_level = st.selectbox("Traffic level", TRAFFIC_LEVELS)
            weather_condition = st.selectbox("Weather condition", WEATHER_CONDITIONS)
            season = st.selectbox("Season", SEASONS)
        with col3:
            hour_of_day = st.slider("Hour of day", 0, 23, 9)
            ride_distance_km = st.number_input("Ride distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
            estimated_ride_time_min = st.number_input(
                "Estimated ride time (min)", min_value=1.0, max_value=180.0, value=20.0, step=1.0
            )

        col4, col5 = st.columns(2)
        with col4:
            base_fare = st.number_input("Base fare (Rs)", min_value=0.0, max_value=2000.0, value=100.0, step=5.0)
        with col5:
            surge_multiplier = st.slider("Surge multiplier", 1.0, 3.0, 1.0, step=0.1)

        st.markdown(
            "**Customer & driver history** -- the single strongest predictors of ride "
            "outcome for this dataset (see PROJECT_IMPLEMENTATION_GUIDE.md section 6.4). "
            "Defaults are the dataset averages; adjust to simulate a specific customer/driver."
        )
        col6, col7 = st.columns(2)
        with col6:
            customer_cancellation_rate = st.slider("Customer's historical cancellation rate", 0.0, 1.0, 0.23, step=0.01)
            customer_avg_rating = st.slider("Customer's average rating", 1.0, 5.0, 4.0, step=0.1)
        with col7:
            driver_delay_rate = st.slider("Driver's historical delay rate", 0.0, 1.0, 0.13, step=0.01)
            driver_acceptance_rate = st.slider("Driver's acceptance rate", 0.0, 1.0, 0.75, step=0.01)
            driver_avg_rating = st.slider("Driver's average rating", 1.0, 5.0, 4.0, step=0.1)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    if ride_distance_km <= 0 or estimated_ride_time_min <= 0:
        st.error("Distance and estimated ride time must be positive.")
        return

    scenario = _build_scenario_row(
        city, vehicle_type, traffic_level, weather_condition, day_of_week, season,
        hour_of_day, ride_distance_km, estimated_ride_time_min, base_fare, surge_multiplier,
        customer_cancellation_rate, customer_avg_rating,
        driver_delay_rate, driver_acceptance_rate, driver_avg_rating,
    )

    st.divider()
    result_col, fare_col = st.columns(2)

    with result_col:
        st.markdown("### Ride outcome prediction")
        proba = ride_outcome_model.predict_proba(scenario)[0]
        classes = ride_outcome_model.classes_
        proba_df = pd.DataFrame({"outcome": classes, "probability": proba}).sort_values(
            "probability", ascending=False
        )
        predicted = proba_df.iloc[0]["outcome"]
        st.metric("Most likely outcome", predicted)
        fig = px.bar(
            proba_df, x="probability", y="outcome", orientation="h", color="outcome",
            color_discrete_map=palette.BOOKING_STATUS_COLORS, title="Outcome probabilities",
        )
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    with fare_col:
        st.markdown("### Fare estimate")
        predicted_fare = float(fare_model.predict(scenario)[0])
        st.metric("Estimated booking value", f"Rs {predicted_fare:,.0f}")
        if fare_rmse:
            st.caption(
                f"Typical model error (RMSE): +/- Rs {fare_rmse:,.0f} "
                f"(roughly Rs {predicted_fare - fare_rmse:,.0f} - Rs {predicted_fare + fare_rmse:,.0f})"
            )
