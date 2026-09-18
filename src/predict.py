from __future__ import annotations

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_DIR

RANGES = {"age": (15, 22), "studytime": (1, 4), "absences": (0, 93), "traveltime": (1, 4), "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5), "dalc": (1, 5), "walc": (1, 5), "health": (1, 5)}


def validate_input(values: dict) -> None:
    missing = set(FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    for name, (low, high) in RANGES.items():
        value = values[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")


def predict(values: dict) -> tuple[float, str]:
    validate_input(values)
    model = joblib.load(MODEL_DIR / "regression_model.joblib")
    grade = float(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    level = "Low" if grade < 10 else "Medium" if grade < 15 else "High"
    return max(0.0, min(20.0, grade)), level
