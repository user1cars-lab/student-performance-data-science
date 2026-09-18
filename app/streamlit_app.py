from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src.config import FEATURES, MODEL_DIR, PROCESSED_DIR
from src.predict import predict

st.set_page_config(page_title="Student Performance", page_icon="📊", layout="wide")
st.title("Student Performance Prediction")
st.caption("Educational prototype. Predictions describe model behavior and are not causal or high-stakes decisions.")

path = PROCESSED_DIR / "student_performance_clean.csv"
if not path.exists():
    st.error("Run `python scripts/train.py` first.")
    st.stop()
df = pd.read_csv(path)

pages = st.sidebar.radio("Page", ["Overview", "Exploration", "Prediction", "Explainability", "Evaluation"])
if pages == "Overview":
    st.header("Overview")
    a, b, c, d = st.columns(4)
    a.metric("Students", len(df)); b.metric("Mean grade", f"{df.g3.mean():.2f}/20"); c.metric("Mean absences", f"{df.absences.mean():.1f}"); d.metric("Missing values", int(df.isna().sum().sum()))
    st.subheader("Performance levels")
    st.bar_chart(df["performance_level"].value_counts())
    st.info("The dataset is public and anonymized. This system is for educational analysis, not institutional decisions.")
elif pages == "Exploration":
    st.header("Exploratory analysis")
    numeric = st.multiselect("Numeric variables", df.select_dtypes("number").columns.tolist(), default=["g3", "absences", "studytime", "failures"])
    if numeric: st.line_chart(df[numeric].sort_values("g3").reset_index(drop=True))
    st.dataframe(df.groupby("performance_level")["g3"].agg(["count", "mean", "median", "std"]).round(2))
elif pages == "Prediction":
    st.header("Individual prediction")
    st.warning("A prediction explains model behavior and does not establish causality or determine a student's future.")
    with st.form("prediction"):
        values = {}
        for feature in FEATURES:
            if df[feature].dtype == "object": values[feature] = st.selectbox(feature, sorted(df[feature].dropna().unique()))
            else: values[feature] = st.number_input(feature, value=float(df[feature].median()), step=1.0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            grade, level = predict(values)
            st.success(f"Predicted grade: {grade:.2f}/20 — {level}")
        except ValueError as exc: st.error(str(exc))
elif pages == "Explainability":
    st.header("Model explainability")
    importance_path = PROCESSED_DIR / "permutation_importance.csv"
    shap_path = PROCESSED_DIR / "local_shap_explanation.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(10).set_index("feature")
        st.subheader("Global permutation importance"); st.bar_chart(importance["importance_mean"])
    if shap_path.exists():
        st.subheader("Example local SHAP contributions"); st.dataframe(pd.read_csv(shap_path))
    st.caption("Importance and SHAP values describe the fitted model; they are not causal effects.")
elif pages == "Evaluation":
    st.header("Evaluation")
    for filename in ["tuned_regression_results.csv", "tuned_classification_results.csv", "classification_report.csv", "confusion_matrix.csv"]:
        file = PROCESSED_DIR / filename
        if file.exists():
            st.subheader(filename); st.dataframe(pd.read_csv(file))
