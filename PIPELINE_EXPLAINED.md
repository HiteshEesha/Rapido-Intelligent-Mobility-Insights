# The Rapido Pipeline, Explained Like You're New to ML

This is a plain-English walkthrough of what actually happens when this project runs, why each
step exists, and why the specific models were chosen. It follows the real code (`src/*.py`), not
just the plan — where the code and the original plan disagreed (mostly in Section 6), it says so.

For the original requirements/reasoning doc this project was built from, see
`PROJECT_IMPLEMENTATION_GUIDE.md`. This file is the "explain it to me simply" companion to that.

---

## 1. The big picture

Five raw CSV files come in. Four trained models and a SQLite database come out. In between:

```
bookings.csv  ─┐
customers.csv ─┤
drivers.csv   ─┼──▶  LOAD  ──▶  VALIDATE  ──▶  CLEAN  ──▶  ENGINEER FEATURES ──▶  TRAIN & TUNE ──▶ EVALUATE ──▶  SAVE (models/*.joblib, rapido.sqlite3)
location_demand.csv ─┤            │                                                                  │
time_features.csv ──┘             ▼                                                                  ▼
                        data-quality report                                              metrics.json + Streamlit dashboard
```

Two entry points run this:
- `python -m src.train` — the **real** pipeline. Loads → validates → cleans → engineers features →
  trains all 4 models → evaluates them → saves the trained models to `models/` and the cleaned
  tables to `rapido.sqlite3`. This is the one that produces the `.joblib` files the dashboard loads.
- `src/pipeline.py` (`run_pipeline`) — a **lighter** version used by the Streamlit "Run data
  pipeline" button. It does everything *except* training a model — load, validate, clean, engineer
  features — so the dashboard can show fresh EDA/data-quality numbers without waiting for a full
  model-training run every time someone clicks a button.

Why split it this way: training 4 models with hyperparameter search takes real time. A dashboard
button that a user might click repeatedly shouldn't re-train models every time — it should just
reuse the models already saved on disk and only redo the cheap data-prep work.

---

## 2. Step 1 — Data Loading (`src/data_loader.py`)

**What it does:** reads the 5 CSVs (`bookings`, `customers`, `drivers`, `location_demand`,
`time_features`) from the `data/` folder into pandas DataFrames.

**Why it's this simple:** there's no transformation here on purpose. Loading is kept completely
separate from cleaning/validating so that:
- Validation (next step) checks the data **exactly as it arrived**, not a version you've already
  quietly patched — otherwise you'd never know the raw data had a problem.
- If a file is missing, it loads an empty table instead of crashing, so one missing CSV doesn't
  take down the whole pipeline before you even get to see a report saying *which* file is missing.

Think of this step as just "open the boxes and put the contents on the table" — no judgment calls
about what's good or bad data yet.

---

## 3. Step 2 — Validation (`src/validation.py`)

**What it does:** runs a checklist over the raw data and produces a report (a list of
pass/fail items) — it never fixes or crashes on bad data, it just tells you what's wrong.

**Why validate *before* cleaning:** if you clean first, you silently paper over problems (nulls
disappear, weird values get replaced) before anyone — including you — gets to see how bad the raw
data actually was. Validating first means the "Data Quality" page in the dashboard shows the truth
about the source files, and cleaning becomes a documented, deliberate fix rather than an invisible
one.

**Why it never crashes the pipeline** (`severity` is tracked but nothing raises an exception): a
real production data pipeline that hard-stops on the first data problem is fragile — one bad row
in a 20,000-row file would block the whole team. Instead, issues are surfaced as a report so a
human can decide if they're serious.

The checklist, in plain terms:

| Check | Plain-English question it answers |
|---|---|
| Required columns present | "Did the file we expected actually show up with all its columns?" |
| Primary key uniqueness (`booking_id`, `customer_id`, `driver_id`) | "Is any booking/customer/driver accidentally listed twice?" |
| Referential integrity | "Does every booking point to a customer and driver that actually exist?" (catches broken foreign keys) |
| Range checks (distance ≥ 0, fare ≥ 0, surge ≥ 1.0, hour 0–23) | "Are there physically impossible values, like a negative fare or a 27th hour?" |
| Rate columns in [0, 1] | "Is `cancellation_rate`/`acceptance_rate`/`delay_rate` actually a valid percentage, not e.g. 150%?" |
| Rating columns in [1, 5] | "Is every rating within the actual rating scale?" |
| Business rules | "Does `actual_ride_time_min` only exist for *Completed* rides, and does `incomplete_ride_reason` only exist for *Incomplete* ones?" — catches internally-contradictory rows |

That last one matters a lot for the next step, because it's the reasoning behind a key cleaning
decision.

---

## 4. Step 3 — Preprocessing / Cleaning (`src/preprocessing.py`)

**What it does:** parses dates, and fills missing values — but only where a missing value is
actually *missing data*, not where it's a structurally expected null.

**The key idea — "missing" vs "expected empty":**

`actual_ride_time_min` is blank for every ride that isn't `Completed` — because a cancelled ride
never had an actual duration to record. That's not a data-quality problem, it's just how reality
works. So the code deliberately **does not fill it in**. Filling it with, say, the median ride time
would invent a duration for a ride that never happened — a subtle bug that would make the data look
cleaner while actually making it wrong. Same logic applies to `incomplete_ride_reason` — a
completed ride was never incomplete, so it shouldn't have a reason.

Everything else gets filled:
- **Numeric columns** → filled with the column's **median**. Median (not average) is used because
  it isn't dragged around by a few extreme outliers — a handful of very long rides won't distort
  what "typical" looks like the way an average would.
- **Text/category columns** → filled with the literal string `"Unknown"`, so the row is kept (not
  dropped) and it's still obvious downstream that this value wasn't originally known.

**Why keep the row instead of dropping it:** dropping any row with a missing value can silently
throw away a large chunk of real data if a column has scattered gaps. Filling with a sensible
default keeps every row usable for the many other columns that *are* complete.

---

## 5. Step 4 — Feature Engineering (`src/feature_engineering.py`)

Feature engineering means: take the raw columns and build new, more *useful* signals from them —
things a model (or a human reading a dashboard) can act on more directly than the raw numbers.

| Feature | How it's built | Why it exists |
|---|---|---|
| `fare_per_km` | `booking_value / ride_distance_km` | Normalizes fare by trip length, so you can spot "this ride was expensive *for its distance*" rather than just "this ride was expensive." |
| `fare_per_min` | `booking_value / actual_ride_time_min` | Same idea, but by time instead of distance — useful for spotting traffic-driven pricing patterns. |
| `rush_hour_flag` | 1 if `hour_of_day` is in `{7,8,9,17,18,19}`, else 0 | A simple yes/no a model can use directly, instead of making it re-learn "which specific hours count as rush hour" from scratch. |
| `peak_time_flag` | 1 if `hour_of_day` is in the dataset's empirically busiest hours `{8,9,10,17,18,19,20}` | Same idea as rush hour, but based on when demand in `location_demand`/`time_features` is actually elevated, not just a generic commute assumption. |
| `long_distance_flag` | 1 if `ride_distance_km ≥ 10` | Long rides tend to behave differently (pricing, cancellation risk) — flagging this lets the model treat "long trip" as a distinct case instead of just a bigger number on a continuous scale. |
| `city_pair` | `"{city}: {pickup} -> {drop}"` | Groups rides by their actual route, useful for EDA ("which routes are busiest / most cancelled") — not fed into the models themselves. |
| `season` | Joined in from `time_features` by calendar date | Adds a seasonal signal without having to re-derive it from a raw date. |
| `ride_completed_flag` / `is_cancelled_flag` / `is_incomplete_flag` | One-hot of `booking_status` | Convenience columns for charts and regression-style analysis — **never** fed into the Ride Outcome classifier itself, since that model's whole job is to predict `booking_status`, and feeding it a disguised copy of the answer would be cheating (this is called "target leakage" — see the callout box below). |
| `driver_reliability_score` | `0.4·acceptance_rate + 0.4·(1 − delay_rate) + 0.2·(rating/5)`, clipped to [0,1] | A single 0–1 number that blends "does this driver accept rides," "are they on time," and "are they rated well" — easier to rank and threshold than three separate numbers. |
| `customer_loyalty_score` | `0.3·tenure + 0.3·booking-volume + 0.2·(rating/5) + 0.2·(1 − cancellation_rate)`, clipped to [0,1] | Same idea for customers: one score blending how long they've been around, how much they book, how they're rated, and how often they cancel. |
| `customer_cancellation_rate`, `customer_avg_rating`, `driver_delay_rate`, `driver_acceptance_rate`, `driver_avg_rating` | Joined from `customers.csv`/`drivers.csv` onto each booking | The single most important addition in this project — see the callout box below. |

> **What is "target leakage" and why does it matter here?**
> A model looks impressively accurate if you accidentally give it a feature that already contains
> (or gives away) the answer. It'll ace the test but be useless in the real world, because at
> prediction time — *before* the ride actually happens — you won't have that information yet. This
> is why `actual_ride_time_min` and `incomplete_ride_reason` are explicitly excluded from the Ride
> Outcome and Fare models (`src/config.py`'s feature lists): both are only known *after* a ride
> finishes, so a real customer app could never supply them when asking "will this ride succeed?"

> **Why join in customer/driver history at all — isn't that also risky?**
> It's a different situation. `customer_cancellation_rate` and `driver_delay_rate` are *lifetime*
> averages computed from a customer/driver's *other* past bookings — they exist and are knowable
> the moment a new booking is created, before that specific ride has any outcome. That's a
> legitimate "this customer/driver has a track record" signal, not a leak. It also turned out to
> matter a lot in practice: without it, the Ride Outcome model plateaued at ~68% accuracy — barely
> better than just always guessing "Completed" (the majority class). Adding these joined columns
> pushed accuracy to ~72.5% and made them the single most important features in the model. Full
> story of that investigation is in `PROJECT_IMPLEMENTATION_GUIDE.md` §6.4.

---

## 6. Step 5 — Modelling (`src/modeling.py`)

There are **4 models**, because there are 4 different business questions, and each question has a
different "shape" of answer:

| # | Model | Business question | Answer shape | Target column |
|---|---|---|---|---|
| 1 | Ride Outcome | "Will this ride complete, get cancelled, or go incomplete?" | 3 categories (multi-class) | `booking_status` |
| 2 | Fare Prediction | "What will this ride cost?" | A number (regression) | `booking_value` |
| 3 | Customer Cancellation Risk | "Is this customer likely to cancel?" | Yes/no (binary) | `customer_cancel_flag` |
| 4 | Driver Delay Risk | "Is this driver likely to run late?" | Yes/no (binary) | `driver_delay_flag` |

### 6.1 Why these specific algorithms

**Random Forest — used for Ride Outcome and Fare Prediction.**
Imagine one decision tree as a single flowchart of yes/no questions ("Is distance > 10km? → Is
traffic High? → ..."). One flowchart tends to memorize the quirks of the exact data it was shown,
rather than the general pattern. A Random Forest builds *hundreds* of slightly different flowcharts
— each one only sees a random slice of the rows and a random slice of the columns — and then lets
them vote (majority vote for a category, average for a number). It's like polling hundreds of
independent analysts instead of trusting one: individual mistakes cancel out, so it's much more
reliable, while still training fast enough to be practical here.

**Logistic Regression vs. Random Forest — a bake-off, for the two risk models.**
For Customer Cancellation Risk and Driver Delay Risk, the code doesn't just pick one algorithm — it
trains *both* and keeps whichever scores higher on AUC (explained below):

```python
candidates = {
    "logistic_regression": LogisticRegression(...),
    "random_forest": RandomForestClassifier(...),
}
# whichever gets the higher ROC-AUC on the test set wins
```

Why start with Logistic Regression at all: think of it as a scorecard, like a bank loan officer's
checklist — every factor (rating, cancellation history, etc.) gets a weight, the weights add up,
and the total becomes a 0–1 probability. It's simple and its weights are directly explainable
("a 1-point rating drop raises cancellation odds by X%"), which matters when the answer needs to be
justified to a business stakeholder, not just be accurate. Random Forest is only kept as the final
model *if it actually beats* that simple baseline on this project's data — added complexity has to
earn its keep, not be assumed to be better. In the current trained run, Logistic Regression won for
Customer Risk and Random Forest won for Driver Risk (see `models/metrics.json`'s
`best_params.chosen_algorithm`).

### 6.2 Why hyperparameter tuning (`GridSearchCV`)

Every model has a few "knobs" — for Random Forest, how many trees to build (`n_estimators`) and how
deep each tree is allowed to grow (`max_depth`); for Logistic Regression, how strongly to penalize
overly-confident weights (`C`). `GridSearchCV` tries every combination of knob-values you give it,
scores each one with cross-validation, and keeps the best-scoring combination automatically —
removing the guesswork of picking these by hand.

### 6.3 Why an 80/20 split with `stratify=y`, and a fixed `random_state`

- **80/20 split:** the model trains on 80% of the data and is judged on the remaining 20% it never
  saw during training — otherwise you'd be grading a model on questions it already memorized the
  answers to.
- **`stratify=y`:** for the classification models, this forces the 80/20 split to keep the same
  *proportion* of each class in both halves. This matters most for the two binary risk models,
  where "at risk" customers/drivers are a minority — a plain random split could accidentally put
  almost none of them in the test set, making the evaluation meaningless.
- **`random_state=42`:** a fixed seed so the split (and therefore the reported metrics) is
  reproducible — running training again gives the same test set and comparable numbers, instead of
  a new random split every time making it look like the model changed when it didn't.

### 6.4 Why every model is one `sklearn.Pipeline` (preprocessing bundled with the estimator)

```python
Pipeline([("preprocess", ColumnTransformer(...)), ("model", RandomForestClassifier(...))])
```

The `ColumnTransformer` one-hot encodes category columns (turns e.g. `city="Mumbai"` into a set of
0/1 columns the model can use — models need numbers, not text) and passes numeric columns straight
through. Bundling this *inside* the saved pipeline means anyone using the trained model later (the
Streamlit dashboard, in this case) can call `.predict(raw_dataframe)` directly on ordinary,
human-readable columns — they never have to remember or re-implement "oh right, I need to one-hot
encode `city` the exact same way training did" themselves. That consistency is important: encoding
raw input slightly differently at prediction time than at training time is a common, hard-to-spot
bug this design avoids entirely.

### 6.5 Why Ride Outcome doesn't use `class_weight="balanced"` (but the risk models do)

`class_weight="balanced"` tells the model "pay extra attention to the rare class" — useful when you
care about catching rare events even at the cost of overall accuracy. The two risk models use it
because "at risk" customers/drivers are a small minority and missing them (a false negative) is
the costly mistake to avoid. The Ride Outcome model does **not** use it, because this project's
target benchmark is *raw accuracy* (85–90%), and balancing trades some of that away in exchange for
better minority-class detection — the wrong trade for this specific goal. (`f1_macro` — a metric
that weighs all classes equally regardless of size — is still reported alongside accuracy, so that
trade-off stays visible even though it isn't what's being optimized.)

---

## 7. Step 6 — Evaluation (`src/evaluation.py`)

Classification and regression get graded with different rulers, because "right or wrong" doesn't
apply the same way to a category as it does to a number:

**Classification (Ride Outcome, Customer Risk, Driver Risk):**
- **Accuracy** — % of predictions that exactly matched reality. Simple, but misleading alone if one
  class dominates (e.g. if 90% of rides complete, a model that *always* guesses "Completed" scores
  90% while being useless).
- **F1 (macro/weighted)** — a blend of precision ("when I said cancel, was I right?") and recall
  ("of all the actual cancels, how many did I catch?"). *Macro* treats every class equally
  regardless of size — a fairer check on whether rare classes are being ignored, which raw accuracy
  hides.
- **AUC (ROC-AUC)** — how well the model's probability scores rank actual positives above actual
  negatives, across every possible decision threshold — not just the default 0.5 cutoff. A model
  can have a mediocre accuracy at threshold 0.5 but still have genuinely useful, well-ranked risk
  scores; AUC captures that.
- **Confusion matrix** — a small table of exactly which classes get confused for which (e.g. "how
  many actually-Incomplete rides did the model call Completed?"), more diagnostic than one number.

**Regression (Fare Prediction):**
- **RMSE** (root mean squared error) — the typical size of the model's ₹ error, with big misses
  penalized extra heavily.
- **MAE** (mean absolute error) — the typical size of the error, without the extra penalty on big
  misses — easier to read as "on average, off by about ₹X."
- **R²** — what fraction of the fare's variation the model actually explains (1.0 = perfect, 0 = no
  better than always guessing the average fare).

Every evaluation is also checked against a plain-language benchmark from the project guide:
classification should hit **85–90% accuracy**, and regression RMSE should be within **±10%** of the
mean fare. `evaluate_classification`/`evaluate_regression` compute this automatically
(`meets_accuracy_benchmark` / `meets_rmse_benchmark`) so it's a visible pass/fail, not something you
have to eyeball from a raw metric.

### 7.1 What the current trained run (`models/metrics.json`) actually shows

| Model | Result | Meets benchmark? | Honest read |
|---|---|---|---|
| Ride Outcome | 72.5% accuracy, AUC 0.76 | ❌ (below 85–90%) | Below target, and documented as a known limitation rather than papered over — see the callout below. |
| Fare Prediction | RMSE = 3.5% of mean fare, R² = 0.997 | ✅ | Comfortably beats the ±10% target — fare is mostly explainable from distance/vehicle/surge. |
| Customer Cancellation Risk | 100% accuracy, AUC 1.0 | ✅ (suspiciously so) | See caution below. |
| Driver Delay Risk | 100% accuracy, AUC 1.0 | ✅ (suspiciously so) | See caution below. |

> **Why is the Ride Outcome model "only" 72.5%, and is that a bug?**
> No — it's an honestly-reported limitation, not a bug. Three attempts were made (documented in
> `PROJECT_IMPLEMENTATION_GUIDE.md` §6.4): the first two versions plateaued at or below the trivial
> "always guess the majority class" baseline (~68%). Adding the customer/driver historical
> reliability features (Section 5 above) pushed it to 72.5% and made those columns the most
> important features in the model — but it still falls short of the 85–90% target. The likely
> reason: `Incomplete` rides are driven by one-off events (`incomplete_ride_reason` values like
> "App Issue," "Vehicle Issue," "Customer No-show") that genuinely aren't predictable in advance
> from any column in this dataset. Chasing the benchmark further by overfitting would produce a
> model that looks better on paper but is actually less trustworthy — so the honest, below-target
> number was kept and documented instead.

> **Why do the two risk models score a perfect 100%/AUC 1.0 — and should that be trusted at face
> value?**
> A perfect score on real-world behavioral data is unusual enough to be worth questioning rather
> than celebrating. The likely explanation: `customer_cancellation_rate`/`driver_delay_rate` (fed
> in as features) and `customer_cancel_flag`/`driver_delay_flag` (the targets) are both derived
> from the *same underlying lifetime counts* in the source CSVs — e.g. if `customer_cancel_flag` is
> essentially "1 if `cancellation_rate` is above some cutoff," the model doesn't need to learn a
> risk pattern at all, it just needs to (re-)discover that cutoff, which is nearly trivial for it to
> do perfectly. This would make the "risk model" closer to reconstructing a rule that already
> exists in the data than genuinely predicting *new* risk from independent signals. Worth
> confirming against how `customer_cancel_flag`/`driver_delay_flag` were originally generated before
> presenting these two models' accuracy to a business audience as-is.

---

## 8. Step 7 — Saving results

- **`models/*.joblib`** — each trained model (the *entire* pipeline: encoding + estimator) is saved
  with `joblib.dump`, so the Streamlit dashboard can load it back and call `.predict()` without
  retraining anything.
- **`models/metrics.json`** — every model's evaluation metrics, in one file, so the dashboard's
  "Model Monitoring" page can display them without re-running evaluation.
- **`rapido.sqlite3`** (`src/db.py`) — the *cleaned, feature-engineered* tables (not the raw CSVs)
  get written to a local SQLite database, normalized per `sql/schema.sql` (bookings/customers/
  drivers/locations/time_features, with `pickup_location`/`drop_location` pulled out into a shared
  `locations` lookup table since city+location pairs repeat heavily across bookings). Using SQLite
  means zero setup for local development, while going through SQLAlchemy means swapping in a real
  MySQL/Postgres connection string later requires no code changes — `src/db.py`'s docstring calls
  this out explicitly.

---

## 9. Where the dashboard fits in

`app.py` + `src/dashboard/` is the Streamlit UI that sits on top of everything above — it doesn't
do any data prep or training itself. It loads the already-trained `.joblib` models and the
already-prepared tables, and gives a human:
- an Overview/EDA view of the cleaned data,
- a "Trip Predictor" form that feeds a hypothetical trip through the Ride Outcome + Fare models
  together (since they share the same trip-level inputs, one form avoids asking twice),
- a Risk Scoring view for customer/driver lookups,
- a Model Monitoring page showing the metrics above plus feature importance,
- a "Run data pipeline" button that re-runs the *light* pipeline (`src/pipeline.py`) — load,
  validate, clean, feature-engineer — without re-training, for the reasons explained in Section 1.

---

## 10. File-to-responsibility quick reference

| File | Responsibility |
|---|---|
| `src/data_loader.py` | Read the 5 raw CSVs into DataFrames, no transformation |
| `src/validation.py` | Report data-quality issues without fixing or crashing on them |
| `src/preprocessing.py` | Fill genuinely-missing values; preserve intentionally-empty ones |
| `src/feature_engineering.py` | Build derived features (flags, scores, joins) from cleaned data |
| `src/modeling.py` | Define, train, and tune the 4 model pipelines |
| `src/evaluation.py` | Score trained models against the guide's benchmarks |
| `src/train.py` | Orchestrates all of the above end-to-end; the script you actually run |
| `src/pipeline.py` | The lighter, no-training version `app.py` calls for live data refreshes |
| `src/db.py` | Persist cleaned tables to SQLite in normalized form |
| `src/config.py` | Central place for paths, feature lists per model, and tunable constants |
| `src/dashboard/*`, `app.py` | The Streamlit UI consuming everything above |
