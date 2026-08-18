"""Model training for the Rapido project.

Four models, three "shapes" (see PROJECT_IMPLEMENTATION_GUIDE.md section 6.1):
- Ride Outcome        -> multi-class classification (RandomForest)
- Fare Prediction     -> regression (RandomForest)
- Customer/Driver Risk -> binary classification, Logistic Regression vs
  RandomForest compared by AUC and the better one kept (see section 6.2.1's
  "start simple, only add complexity if it earns its keep" rule).

Every trained model is a single sklearn Pipeline (preprocessing + estimator),
so callers (the training script, the Streamlit app) can call `.predict(df)`
on raw, un-encoded feature columns without duplicating encoding logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_CONFIG


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: Any
    y_proba: Any
    best_params: dict
    feature_columns: list[str] = field(default_factory=list)


def _build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


def _fit_grid_search(estimator, param_grid: dict, preprocessor: ColumnTransformer, X_train, y_train, scoring: str, cv: int = 3):
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    search = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_ride_outcome_model(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    target: str,
    config=DEFAULT_CONFIG,
) -> TrainedModel:
    """Multi-class classification: predict booking_status before trip start."""

    feature_columns = categorical_features + numeric_features
    data = frame.dropna(subset=[target] + feature_columns).copy()
    X, y = data[feature_columns], data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    preprocessor = _build_preprocessor(categorical_features, numeric_features)
    param_grid = {"model__n_estimators": [200, 300], "model__max_depth": [12, None]}
    # No class_weight="balanced" here: the guide's benchmark is raw accuracy
    # (85-90%), and balancing trades accuracy for minority-class recall --
    # f1_macro is still reported in evaluation.py so that trade-off stays visible.
    estimator = RandomForestClassifier(random_state=config.random_state)

    best_pipeline, best_params = _fit_grid_search(
        estimator, param_grid, preprocessor, X_train, y_train, scoring="accuracy"
    )

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)

    return TrainedModel("ride_outcome", best_pipeline, X_test, y_test, y_pred, y_proba, best_params, feature_columns)


def train_fare_model(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    target: str,
    config=DEFAULT_CONFIG,
) -> TrainedModel:
    """Regression: predict booking_value before trip confirmation."""

    feature_columns = categorical_features + numeric_features
    data = frame.dropna(subset=[target] + feature_columns).copy()
    X, y = data[feature_columns], data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    preprocessor = _build_preprocessor(categorical_features, numeric_features)
    param_grid = {"model__n_estimators": [200, 300], "model__max_depth": [12, None]}
    estimator = RandomForestRegressor(random_state=config.random_state)

    best_pipeline, best_params = _fit_grid_search(
        estimator, param_grid, preprocessor, X_train, y_train, scoring="neg_root_mean_squared_error"
    )

    y_pred = best_pipeline.predict(X_test)

    return TrainedModel("fare_prediction", best_pipeline, X_test, y_test, y_pred, None, best_params, feature_columns)


def train_binary_risk_model(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    target: str,
    name: str,
    config=DEFAULT_CONFIG,
) -> TrainedModel:
    """Binary classification: try Logistic Regression first, keep RandomForest
    only if it beats the baseline on AUC (see guide section 6.2.1)."""

    feature_columns = categorical_features + numeric_features
    data = frame.dropna(subset=[target] + feature_columns).copy()
    X, y = data[feature_columns], data[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    preprocessor = _build_preprocessor(categorical_features, numeric_features)

    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.random_state, class_weight="balanced"),
            {"model__n_estimators": [200, 300], "model__max_depth": [8, None]},
        ),
    }

    best = None
    for candidate_name, (estimator, grid) in candidates.items():
        pipeline, params = _fit_grid_search(estimator, grid, preprocessor, X_train, y_train, scoring="roc_auc")
        proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        if best is None or auc > best["auc"]:
            best = {"algorithm": candidate_name, "pipeline": pipeline, "params": params, "auc": auc}

    y_pred = best["pipeline"].predict(X_test)
    y_proba = best["pipeline"].predict_proba(X_test)
    best_params = {**best["params"], "chosen_algorithm": best["algorithm"]}

    return TrainedModel(name, best["pipeline"], X_test, y_test, y_pred, y_proba, best_params, feature_columns)


def get_feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Return a feature-importance/coefficient Series indexed by encoded feature name, or None."""

    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
    return None
