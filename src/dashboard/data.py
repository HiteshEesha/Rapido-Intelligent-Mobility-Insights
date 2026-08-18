"""Cached data/model loaders shared by every dashboard page."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from ..config import DATA_DIR, MODELS_DIR
from ..pipeline import run_pipeline

MODEL_FILES = {
    "ride_outcome": "ride_outcome_model.joblib",
    "fare": "fare_model.joblib",
    "customer_risk": "customer_risk_model.joblib",
    "driver_risk": "driver_risk_model.joblib",
}


@st.cache_data(show_spinner="Loading and preparing datasets...")
def load_prepared_data(data_dir: str = str(DATA_DIR)) -> dict:
    return run_pipeline(Path(data_dir))["prepared"]


@st.cache_resource(show_spinner="Loading trained models...")
def load_models(models_dir: str = str(MODELS_DIR)) -> dict:
    base = Path(models_dir)
    models = {}
    for key, filename in MODEL_FILES.items():
        path = base / filename
        models[key] = joblib.load(path) if path.exists() else None
    return models


def load_metrics(models_dir: str = str(MODELS_DIR)) -> dict | None:
    path = Path(models_dir) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def models_are_missing(models: dict) -> bool:
    return any(model is None for model in models.values())
