# Rapido: Intelligent Mobility Insights

Ride outcome prediction, fare estimation, and cancellation/delay risk scoring
for a ride-hailing platform, built from the brief in `data/ProjectGuide.txt`.
See `PROJECT_IMPLEMENTATION_GUIDE.md` for the full design rationale (data
dictionary, validation rules, feature engineering, modeling choices, and
Streamlit UI/UX plan) -- this README only covers setup and how to run it.

## Setup

```powershell
# from the repo root, with the provided .venv (or create your own)
.venv\Scripts\pip install -r requirements.txt
```

## Train the models

```powershell
.venv\Scripts\python -m src.train
```

This loads the CSVs from `data/`, validates and cleans them, engineers
features, trains all four models, evaluates them, and writes:

- `models/*.joblib` -- the four trained sklearn pipelines (preprocessing +
  estimator bundled together)
- `models/metrics.json` -- accuracy/F1/AUC/confusion matrix (classifiers) and
  RMSE/MAE/R^2 (fare regressor), each checked against the guide's benchmarks
- `rapido.sqlite3` -- the cleaned datasets loaded as normalized SQL tables
  (see `sql/schema.sql` for the reference DDL)

You can also train from inside the app: **Model Monitoring -> Train / retrain
all models**.

## Run the dashboard

```powershell
.venv\Scripts\streamlit run app.py
```

Pages:

- **Overview** -- KPI tiles and headline trends, filterable by city/vehicle
  type/status/date range.
- **EDA & Trends** -- the guide's required exploratory visuals (ride volume,
  cancellation heatmap, distance-vs-fare, ratings, customer-vs-driver
  comparison, traffic/weather vs cancellation, vehicle usage).
- **Trip Predictor** -- enter a hypothetical trip to get a Ride Outcome
  prediction (Completed/Cancelled/Incomplete, with probabilities) and a Fare
  estimate, side by side.
- **Risk Scoring** -- customer cancellation risk and driver delay risk,
  ranked tables plus per-ID lookup.
- **Model Monitoring** -- metrics vs benchmarks, confusion matrices, feature
  importances, and the retrain button.
- **Data Quality** -- the validation report (structural, referential
  integrity, range, and business-rule checks) run against the raw CSVs.

## Project layout

- `data/` -- source CSVs (`bookings`, `customers`, `drivers`,
  `location_demand`, `time_features`) plus the original project brief.
- `src/config.py` -- paths and the feature lists each model uses.
- `src/validation.py` -- data quality checks (non-fatal, reported not raised).
- `src/preprocessing.py` -- cleaning (datetime parsing, missing-value rules).
- `src/feature_engineering.py` -- the guide's requested derived features
  (Fare_per_KM, Rush_Hour_Flag, City_Pair, reliability/loyalty scores, ...).
- `src/modeling.py` -- the four model-training functions (multi-class,
  regression, two binary risk models) as single sklearn Pipelines.
- `src/evaluation.py` -- classification and regression metrics, checked
  against the guide's benchmarks.
- `src/db.py` / `sql/schema.sql` -- SQL persistence (SQLite by default,
  normalized schema matching the real columns).
- `src/train.py` -- the training/evaluation/persistence orchestration script.
- `src/pipeline.py` -- the lighter data-prep path the dashboard uses (no
  training).
- `src/dashboard/` -- one module per Streamlit page, plus shared
  filters/palette/data-loading helpers.
- `app.py` -- the Streamlit entry point that wires the pages together.

## Known data limitations

- No `payment_method` column exists anywhere in the dataset, so the guide's
  "payment method usage patterns" EDA bullet isn't producible -- the EDA page
  shows vehicle-type usage patterns instead and says so explicitly.
- No GPS coordinates -- `pickup_location`/`drop_location` are anonymized
  per-city location codes, so location-based visuals are by-category, not on
  a literal map.
- The Ride Outcome model's accuracy falls short of the guide's 85-90%
  benchmark even after adding customer/driver historical reliability
  features (see `PROJECT_IMPLEMENTATION_GUIDE.md` section 6.4 for the full
  investigation) -- the `Incomplete` outcome class in particular has very
  little signal in the available features.
