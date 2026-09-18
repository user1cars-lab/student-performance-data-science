import pytest

from src.config import FEATURES, MODEL_DIR
from src.predict import predict


def valid_values():
    values = {feature: 1 for feature in FEATURES}
    values.update({"school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "pstatus": "A", "mjob": "other", "fjob": "other", "reason": "course", "guardian": "mother", "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no", "age": 17, "absences": 2})
    return values


def test_saved_regression_model_exists():
    assert (MODEL_DIR / "regression_model.joblib").exists()


def test_prediction_is_bounded_and_labeled():
    grade, level = predict(valid_values())
    assert 0 <= grade <= 20
    assert level in {"Low", "Medium", "High"}


def test_missing_feature_is_rejected():
    values = valid_values(); values.pop("school")
    with pytest.raises(ValueError, match="school"):
        predict(values)
