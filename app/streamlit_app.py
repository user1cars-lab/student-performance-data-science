import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import FEATURES, PROCESSED_DIR
from src.predict import predict

st.set_page_config(page_title="Student Performance", page_icon="📊", layout="wide")
st.title("Student Performance Prediction")
st.caption("Educational prototype. Predictions describe model behavior and are not causal or high-stakes decisions.")

data_path = PROCESSED_DIR / "student_performance_clean.csv"
if not data_path.exists():
    st.error("Run `python scripts/train.py` first to prepare data and train models.")
    st.stop()

df = pd.read_csv(data_path)
col1, col2, col3 = st.columns(3)
col1.metric("Students", len(df))
col2.metric("Mean final grade", f"{df.g3.mean():.2f}/20")
col3.metric("Mean absences", f"{df.absences.mean():.1f}")
st.subheader("Grade distribution")
st.bar_chart(df["performance_level"].value_counts())
st.subheader("Predict a student grade")
with st.form("prediction"):
    values = {}
    for feature in FEATURES:
        if df[feature].dtype == "object":
            values[feature] = st.selectbox(feature, sorted(df[feature].dropna().unique()))
        else:
            values[feature] = st.number_input(feature, value=float(df[feature].median()), step=1.0)
    submitted = st.form_submit_button("Predict")
if submitted:
    try:
        grade, level = predict(values)
        st.success(f"Predicted grade: {grade:.2f}/20 — {level}")
    except ValueError as exc:
        st.error(str(exc))
