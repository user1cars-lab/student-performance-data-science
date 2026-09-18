from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FEATURES, MODEL_DIR, RANDOM_STATE


def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in FEATURES if df[c].dtype == "object"]
    numeric = [c for c in FEATURES if c not in categorical]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def train_models(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    X, y_reg = df[FEATURES], df["g3"]
    y_cls = df["performance_level"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
    )
    reg_specs = {
        "baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=2),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=150, max_depth=2),
    }
    cls_specs = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
    }
    reg_models, cls_models = {}, {}
    reg_rows, cls_rows = [], []
    for name, estimator in reg_specs.items():
        model = Pipeline([("preprocess", make_preprocessor(X)), ("model", estimator)])
        model.fit(X_train, yr_train)
        pred = model.predict(X_test)
        cv = cross_validate(model, X_train, yr_train, cv=5, scoring="neg_root_mean_squared_error")
        reg_rows.append({"model": name, "mae": mean_absolute_error(yr_test, pred), "rmse": mean_squared_error(yr_test, pred) ** 0.5, "r2": r2_score(yr_test, pred), "cv_rmse_mean": -cv["test_score"].mean()})
        reg_models[name] = model
    for name, estimator in cls_specs.items():
        model = Pipeline([("preprocess", make_preprocessor(X)), ("model", estimator)])
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        cls_rows.append({"model": name, "accuracy": accuracy_score(yc_test, pred), "precision_macro": precision_score(yc_test, pred, average="macro", zero_division=0), "recall_macro": recall_score(yc_test, pred, average="macro", zero_division=0), "f1_macro": f1_score(yc_test, pred, average="macro", zero_division=0)})
        cls_models[name] = model
    best_reg = min(reg_rows, key=lambda row: row["rmse"])["model"]
    best_cls = max(cls_rows, key=lambda row: row["f1_macro"])["model"]
    joblib.dump(reg_models[best_reg], MODEL_DIR / "regression_model.joblib")
    joblib.dump(cls_models[best_cls], MODEL_DIR / "classification_model.joblib")
    (MODEL_DIR / "metadata.json").write_text(json.dumps({"best_regression": best_reg, "best_classification": best_cls}, indent=2))
    return reg_models, cls_models, pd.DataFrame(reg_rows), pd.DataFrame(cls_rows)
