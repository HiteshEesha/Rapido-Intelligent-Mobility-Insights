"""Load the CSV inputs described in the ProjectGuide."""

from pathlib import Path

import pandas as pd


def load_csv_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all expected project datasets from the data folder."""

    dataset_files = {
        "bookings": "bookings.csv",
        "customers": "customers.csv",
        "drivers": "drivers.csv",
        "location_demand": "location_demand.csv",
        "time_features": "time_features.csv",
    }
    datasets: dict[str, pd.DataFrame] = {}

    for name, filename in dataset_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            datasets[name] = pd.read_csv(file_path)
        else:
            datasets[name] = pd.DataFrame()

    return datasets