from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import FEATURES, FIGURES_DIR, MODEL_DIR, PROCESSED_DIR, RANDOM_STATE
from src.data_pipeline import prepare_dataset
from src.modeling import make_preprocessor


def save_eda(df: pd.DataFrame) -> dict:
    """Create reproducible univariate and bivariate EDA figures and summary."""
    sns.set_theme(style="whitegrid")
    summary = {"rows": int(len(df)), "columns": int(df.shape[1]), "missing_values": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())}
    plt.figure(figsize=(8, 5)); sns.histplot(df["g3"], bins=21, kde=True); plt.xlabel("Final grade (G3)"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "grade_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 5)); sns.scatterplot(data=df, x="absences", y="g3", hue="performance_level", alpha=.75); plt.tight_layout(); plt.savefig(FIGURES_DIR / "absences_vs_grade.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x="studytime", y="g3"); plt.xlabel("Weekly study-time category"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "studytime_vs_grade.png", dpi=160); plt.close()
    numeric = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 8)); sns.heatmap(numeric.corr(), cmap="vlag", center=0); plt.tight_layout(); plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=160); plt.close()
    return summary


def statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Test pre-registered associations without treating association as causation."""
    rows = []
    for feature in ("absences", "studytime", "failures"):
        pearson_r, pearson_p = pearsonr(df[feature], df["g3"])
        spearman_r, spearman_p = spearmanr(df[feature], df["g3"])
        rows.append({"hypothesis": f"{feature} is associated with final grade", "test": "Pearson and Spearman correlation", "feature": feature, "statistic": pearson_r, "p_value": pearson_p, "spearman_statistic": spearman_r, "spearman_p_value": spearman_p, "alpha": 0.05, "decision": "reject H0" if pearson_p < 0.05 else "do not reject H0"})
    yes = df.loc[df["higher"] == "yes", "g3"]
    no = df.loc[df["higher"] == "no", "g3"]
    statistic, p_value = ttest_ind(yes, no, equal_var=False)
    rows.append({"hypothesis": "Final grade differs by higher-education intention", "test": "Welch independent-samples t-test", "feature": "higher", "statistic": statistic, "p_value": p_value, "spearman_statistic": np.nan, "spearman_p_value": np.nan, "alpha": 0.05, "decision": "reject H0" if p_value < 0.05 else "do not reject H0"})
    result = pd.DataFrame(rows); result.to_csv(PROCESSED_DIR / "statistical_tests.csv", index=False); return result


def tune_models(df: pd.DataFrame) -> tuple[Pipeline, Pipeline, pd.DataFrame, pd.DataFrame]:
    """Tune random forest models using only training folds."""
    X, yr, yc = df[FEATURES], df["g3"], df["performance_level"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(X, yr, yc, test_size=.2, random_state=RANDOM_STATE, stratify=yc)
    reg = Pipeline([("preprocess", make_preprocessor(X)), ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
    cls = Pipeline([("preprocess", make_preprocessor(X)), ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))])
    reg_grid = GridSearchCV(reg, {"model__n_estimators": [150, 300], "model__max_depth": [None, 8], "model__min_samples_leaf": [1, 2]}, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
    cls_grid = GridSearchCV(cls, {"model__n_estimators": [150, 300], "model__max_depth": [None, 8], "model__min_samples_leaf": [1, 2]}, scoring="f1_macro", cv=5, n_jobs=-1)
    reg_grid.fit(X_train, yr_train); cls_grid.fit(X_train, yc_train)
    reg_pred, cls_pred = reg_grid.predict(X_test), cls_grid.predict(X_test)
    reg_eval = pd.DataFrame([{"model": "tuned_random_forest", "best_params": json.dumps(reg_grid.best_params_), "test_rmse": float(np.sqrt(np.mean((yr_test - reg_pred) ** 2))), "test_mae": float(np.mean(np.abs(yr_test - reg_pred)))}])
    cls_eval = pd.DataFrame([{"model": "tuned_random_forest", "best_params": json.dumps(cls_grid.best_params_), "test_f1_macro": float(__import__('sklearn.metrics', fromlist=['f1_score']).f1_score(yc_test, cls_pred, average='macro')), "test_accuracy": float(__import__('sklearn.metrics', fromlist=['accuracy_score']).accuracy_score(yc_test, cls_pred))}])
    reg_eval.to_csv(PROCESSED_DIR / "tuned_regression_results.csv", index=False); cls_eval.to_csv(PROCESSED_DIR / "tuned_classification_results.csv", index=False)
    joblib.dump(reg_grid.best_estimator_, MODEL_DIR / "tuned_regression_model.joblib"); joblib.dump(cls_grid.best_estimator_, MODEL_DIR / "tuned_classification_model.joblib")
    return reg_grid.best_estimator_, cls_grid.best_estimator_, reg_eval, cls_eval


def explain_and_errors(df: pd.DataFrame, reg_model: Pipeline, cls_model: Pipeline) -> dict:
    """Generate permutation importance, SHAP plots, residuals, and classification errors."""
    X, yr, yc = df[FEATURES], df["g3"], df["performance_level"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(X, yr, yc, test_size=.2, random_state=RANDOM_STATE, stratify=yc)
    reg_pred = reg_model.predict(X_test); cls_pred = cls_model.predict(X_test)
    errors = X_test.copy(); errors["actual_g3"] = yr_test; errors["predicted_g3"] = reg_pred; errors["residual"] = yr_test - reg_pred; errors["absolute_error"] = np.abs(errors["residual"]); errors.to_csv(PROCESSED_DIR / "regression_error_analysis.csv", index=False)
    pd.DataFrame(classification_report(yc_test, cls_pred, output_dict=True, zero_division=0)).T.to_csv(PROCESSED_DIR / "classification_report.csv")
    pd.DataFrame(confusion_matrix(yc_test, cls_pred, labels=["Low", "Medium", "High"]), index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"]).to_csv(PROCESSED_DIR / "confusion_matrix.csv")
    plt.figure(figsize=(7, 5)); sns.scatterplot(x=yr_test, y=reg_pred); plt.plot([0, 20], [0, 20], "r--"); plt.xlabel("Actual G3"); plt.ylabel("Predicted G3"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 5)); sns.histplot(errors["residual"], kde=True); plt.axvline(0, color="red", linestyle="--"); plt.xlabel("Residual (actual - predicted)"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=160); plt.close()
    perm = permutation_importance(reg_model, X_test, yr_test, n_repeats=10, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error")
    importance = pd.DataFrame({"feature": FEATURES, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False); importance.to_csv(PROCESSED_DIR / "permutation_importance.csv", index=False)
    transformed = reg_model.named_steps["preprocess"].transform(X_test)
    names = reg_model.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(reg_model.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    shap.summary_plot(shap_values, transformed, feature_names=names, show=False, max_display=15); plt.tight_layout(); plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
    local = pd.DataFrame({"feature": names, "shap_value": shap_values[0], "absolute_value": np.abs(shap_values[0])}).sort_values("absolute_value", ascending=False).head(15); local.to_csv(PROCESSED_DIR / "local_shap_explanation.csv", index=False)
    return {"mean_absolute_error": float(errors["absolute_error"].mean()), "largest_error": float(errors["absolute_error"].max()), "top_features": importance.head(10)["feature"].tolist()}


def run_full_analysis() -> None:
    df = pd.read_csv(PROCESSED_DIR / "student_performance_clean.csv") if (PROCESSED_DIR / "student_performance_clean.csv").exists() else prepare_dataset()
    summary = save_eda(df); stats = statistical_analysis(df); reg, cls, reg_eval, cls_eval = tune_models(df); error_summary = explain_and_errors(df, reg, cls)
    (PROCESSED_DIR / "analysis_summary.json").write_text(json.dumps({"data": summary, "statistics": stats.to_dict(orient="records"), "tuned_regression": reg_eval.to_dict(orient="records"), "tuned_classification": cls_eval.to_dict(orient="records"), "error_analysis": error_summary}, indent=2, default=str))


if __name__ == "__main__":
    run_full_analysis()
