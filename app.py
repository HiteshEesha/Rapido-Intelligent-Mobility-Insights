"""Streamlit entry point for Rapido Intelligent Mobility Insights.

Run with: streamlit run app.py
Train models first with: python -m src.train (or use the in-app button on
the Model Monitoring page).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATA_DIR
from src.dashboard import eda, filters, monitoring, overview, predictors, risk
from src.dashboard.data import load_metrics, load_models, load_prepared_data, models_are_missing

st.set_page_config(page_title="Rapido Mobility Insights", page_icon="🚕", layout="wide")

st.title("Rapido: Intelligent Mobility Insights")
st.caption("Ride outcome prediction, fare estimation, and risk scoring for a ride-hailing platform.")

prepared = load_prepared_data(str(DATA_DIR))
bookings, customers, drivers = prepared["bookings"], prepared["customers"], prepared["drivers"]

models = load_models()
metrics = load_metrics()

if models_are_missing(models):
    st.warning(
        "Some models aren't trained yet. Visit **Model Monitoring** and click "
        "**Train / retrain all models** before using the Trip Predictor or Risk Scoring pages."
    )

page = st.sidebar.radio(
    "Page",
    ["Overview", "EDA & Trends", "Trip Predictor", "Risk Scoring", "Model Monitoring", "Data Quality"],
)

if page in ("Overview", "EDA & Trends"):
    filtered_bookings = filters.render_sidebar_filters(bookings)
else:
    filtered_bookings = bookings

if page == "Overview":
    overview.render(filtered_bookings)
elif page == "EDA & Trends":
    eda.render(filtered_bookings, customers, drivers)
elif page == "Trip Predictor":
    fare_rmse = metrics["fare"]["rmse"] if metrics else None
    predictors.render(models.get("ride_outcome"), models.get("fare"), fare_rmse)
elif page == "Risk Scoring":
    risk.render(customers, drivers, models.get("customer_risk"), models.get("driver_risk"))
elif page == "Model Monitoring":
    monitoring.render()
elif page == "Data Quality":
    st.subheader("Data Quality Report")
    st.caption(
        "Validation checks run against the raw CSVs before cleaning "
        "(see PROJECT_IMPLEMENTATION_GUIDE.md section 4)."
    )
    report_df = pd.DataFrame(prepared["validation_report"])
    failed = report_df[~report_df["passed"]]
    if failed.empty:
        st.success("All validation checks passed.")
    else:
        st.warning(f"{len(failed)} check(s) failed.")
    st.dataframe(report_df, width='stretch', hide_index=True)
