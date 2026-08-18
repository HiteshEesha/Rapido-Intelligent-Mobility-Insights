"""Data-preparation pipeline used by the Streamlit dashboard.

This does not train models (see src/train.py for that, run offline via
`python -m src.train`) -- it only loads, validates, cleans, and
feature-engineers the datasets so the dashboard can render EDA and feed the
persisted model pipelines.
"""

from __future__ import annotations

from pathlib import Path

from .train import prepare_datasets


def run_pipeline(data_dir: Path) -> dict:
    """Run the data-prep pipeline and return a small summary (used by the
    'Run data pipeline' preview) plus the prepared datasets."""

    prepared = prepare_datasets(data_dir)
    bookings = prepared["bookings"]

    return {
        "status": "pipeline ready",
        "rows": int(bookings.shape[0]),
        "columns": list(bookings.columns[:20]),
        "validation_issues": sum(1 for item in prepared["validation_report"] if not item["passed"]),
        "prepared": prepared,
    }
