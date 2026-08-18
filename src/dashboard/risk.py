"""Risk Scoring page: customer cancellation risk + driver delay risk, both
scored at the entity level (see guide section 5.1 leakage note) rather than
per upcoming booking.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import CUSTOMER_RISK_CATEGORICAL, CUSTOMER_RISK_NUMERIC, DRIVER_RISK_CATEGORICAL, DRIVER_RISK_NUMERIC
from . import palette


def render(customers: pd.DataFrame, drivers: pd.DataFrame, customer_risk_model, driver_risk_model) -> None:
    st.subheader("Risk Scoring")
    tabs = st.tabs(["Customer cancellation risk", "Driver delay risk"])

    with tabs[0]:
        _render_customer_risk(customers, customer_risk_model)
    with tabs[1]:
        _render_driver_risk(drivers, driver_risk_model)


def _render_customer_risk(customers: pd.DataFrame, model) -> None:
    if model is None:
        st.info("Model not trained yet -- see Model Monitoring.")
        return

    feature_columns = CUSTOMER_RISK_CATEGORICAL + CUSTOMER_RISK_NUMERIC
    scored = customers.copy()
    scored["cancellation_risk"] = model.predict_proba(scored[feature_columns])[:, 1]

    top_n = st.slider("Show top N highest-risk customers", 5, 50, 15, key="customer_top_n")
    top = scored.sort_values("cancellation_risk", ascending=False).head(top_n)
    st.dataframe(
        top[
            ["customer_id", "customer_city", "preferred_vehicle_type", "avg_customer_rating",
             "cancellation_rate", "customer_loyalty_score", "cancellation_risk"]
        ],
        width='stretch',
        hide_index=True,
    )

    fig = px.histogram(scored, x="cancellation_risk", nbins=30, title="Distribution of predicted cancellation risk")
    fig.update_traces(marker_color=palette.CATEGORICAL[0])
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Look up a specific customer")
    customer_id = st.selectbox("Customer ID", scored["customer_id"], key="customer_lookup")
    if customer_id:
        row = scored[scored["customer_id"] == customer_id].iloc[0]
        st.metric("Predicted cancellation risk", f"{row['cancellation_risk']:.1%}")
        st.json(
            {
                "cancellation_rate": float(row["cancellation_rate"]),
                "avg_customer_rating": float(row["avg_customer_rating"]),
                "total_bookings": int(row["total_bookings"]),
                "customer_loyalty_score": round(float(row["customer_loyalty_score"]), 2),
            }
        )


def _render_driver_risk(drivers: pd.DataFrame, model) -> None:
    if model is None:
        st.info("Model not trained yet -- see Model Monitoring.")
        return

    feature_columns = DRIVER_RISK_CATEGORICAL + DRIVER_RISK_NUMERIC
    scored = drivers.copy()
    scored["delay_risk"] = model.predict_proba(scored[feature_columns])[:, 1]

    top_n = st.slider("Show top N highest-risk drivers", 5, 50, 15, key="driver_top_n")
    top = scored.sort_values("delay_risk", ascending=False).head(top_n)
    st.dataframe(
        top[
            ["driver_id", "driver_city", "vehicle_type", "avg_driver_rating",
             "delay_rate", "driver_reliability_score", "delay_risk"]
        ],
        width='stretch',
        hide_index=True,
    )

    fig = px.histogram(scored, x="delay_risk", nbins=30, title="Distribution of predicted delay risk")
    fig.update_traces(marker_color=palette.CATEGORICAL[1])
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Look up a specific driver")
    driver_id = st.selectbox("Driver ID", scored["driver_id"], key="driver_lookup")
    if driver_id:
        row = scored[scored["driver_id"] == driver_id].iloc[0]
        st.metric("Predicted delay risk", f"{row['delay_risk']:.1%}")
        st.json(
            {
                "delay_rate": float(row["delay_rate"]),
                "acceptance_rate": float(row["acceptance_rate"]),
                "avg_driver_rating": float(row["avg_driver_rating"]),
                "driver_reliability_score": round(float(row["driver_reliability_score"]), 2),
            }
        )
