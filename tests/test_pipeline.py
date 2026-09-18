import pandas as pd
import pytest

from src.data_pipeline import clean_data, build_features
from src.predict import validate_input


def sample():
    return pd.DataFrame({"school":["GP"], "sex":["F"], "age":[17], "address":["U"], "famsize":["GT3"], "Pstatus":["A"], "Medu":[4], "Fedu":[4], "Mjob":["at_home"], "Fjob":["teacher"], "reason":["course"], "guardian":["mother"], "traveltime":[1], "studytime":[2], "failures":[0], "schoolsup":["no"], "famsup":["yes"], "paid":["no"], "activities":["no"], "nursery":["yes"], "higher":["yes"], "internet":["yes"], "romantic":["no"], "famrel":[4], "freetime":[3], "goout":[3], "Dalc":[1], "Walc":[1], "health":[5], "absences":[2], "G1":[10], "G2":[11], "G3":[12]})


def test_cleaning_creates_target_label():
    result = clean_data(sample())
    assert result.loc[0, "performance_level"] == "Medium"


def test_feature_engineering_excludes_leakage_columns():
    result = build_features(clean_data(sample()))
    assert "g1" not in result.columns and "g2" not in result.columns
    assert "study_engagement" in result.columns


def test_invalid_prediction_input_is_rejected():
    values = {c: 1 for c in __import__("src.config", fromlist=["FEATURES"]).FEATURES}
    values["age"] = 30
    with pytest.raises(ValueError, match="age"):
        validate_input(values)
