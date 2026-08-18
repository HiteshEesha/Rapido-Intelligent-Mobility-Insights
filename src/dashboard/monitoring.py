"""Model Monitoring page: metrics vs the guide's benchmarks, confusion
matrices, feature importances, and an on-demand retrain button.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..modeling import get_feature_importance
from . import palette
from .data import load_metrics, load_models


def render() -> None:
    st.subheader("Model Monitoring")

    if st.button("Train / retrain all models", type="primary"):
        with st.spinner("Training all 4 models -- this can take a couple of minutes..."):
            from ..train import run_full_pipeline

            run_full_pipeline()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Training complete.")
        st.rerun()

    metrics = load_metrics()
    if metrics is None:
        st.info("No trained models found yet. Click the button above to train them.")
        return

    models = load_models()

    _render_classification_card("Ride Outcome (multi-class)", metrics["ride_outcome"], models.get("ride_outcome"))
    _render_regression_card("Fare Prediction (regression)", metrics["fare"], models.get("fare"))
    _render_classification_card(
        "Customer Cancellation Risk (binary)", metrics["customer_risk"], models.get("customer_risk")
    )
    _render_classification_card("Driver Delay Risk (binary)", metrics["driver_risk"], models.get("driver_risk"))


def _render_classification_card(title: str, metrics: dict, pipeline) -> None:
    st.markdown(f"### {title}")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    cols[1].metric("F1 (weighted)", f"{metrics['f1_weighted']:.3f}")
    cols[2].metric("AUC", f"{metrics['auc']:.3f}" if metrics.get("auc") is not None else "-")
    cols[3].metric("Meets 85-90% benchmark?", "Yes" if metrics["meets_accuracy_benchmark"] else "No")
    st.caption(metrics["accuracy_benchmark_note"])

    col_a, col_b = st.columns(2)
    with col_a:
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale=palette.SEQUENTIAL_BLUE,
            labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix",
        )
        st.plotly_chart(fig, width='stretch')
    with col_b:
        _render_feature_importance(pipeline)
    st.divider()


def _render_regression_card(title: str, metrics: dict, pipeline) -> None:
    st.markdown(f"### {title}")
    cols = st.columns(4)
    cols[0].metric("RMSE", f"Rs {metrics['rmse']:,.2f}")
    cols[1].metric("MAE", f"Rs {metrics['mae']:,.2f}")
    cols[2].metric("R-squared", f"{metrics['r2']:.3f}")
    cols[3].metric("Meets +/-10% benchmark?", "Yes" if metrics["meets_rmse_benchmark"] else "No")
    st.caption(metrics["rmse_benchmark_note"])
    _render_feature_importance(pipeline)
    st.divider()


def _render_feature_importance(pipeline) -> None:
    if pipeline is None:
        return
    importance = get_feature_importance(pipeline)
    if importance is None:
        return
    top = importance.head(10).reset_index()
    top.columns = ["feature", "importance"]
    fig = px.bar(top, x="importance", y="feature", orientation="h", title="Top 10 feature importances")
    fig.update_traces(marker_color=palette.CATEGORICAL[0])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width='stretch')
