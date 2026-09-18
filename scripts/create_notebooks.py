import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

sections = {
"01_data_understanding.ipynb": ("Data understanding", "Load the processed UCI Student Performance data, inspect dimensions, types, missing values, duplicates, and target distribution."),
"02_data_cleaning.ipynb": ("Data cleaning", "Document column normalization, duplicate removal, validation, target labels, and the decision to exclude G1/G2 from predictive features."),
"03_eda.ipynb": ("Exploratory data analysis", "Reproduce the saved grade distribution, absence relationship, study-time comparison, and correlation matrix."),
"04_statistical_analysis.ipynb": ("Statistical analysis", "Run the pre-registered Pearson/Spearman tests and Welch t-test. Interpret p-values without claiming causality."),
"05_feature_engineering.ipynb": ("Feature engineering", "Review the engineered study_engagement and absence_rate_proxy features and verify that target and leakage fields are excluded from X."),
"06_modeling.ipynb": ("Modeling", "Train baseline and candidate regression/classification pipelines with fixed random state."),
"07_model_evaluation.ipynb": ("Model evaluation", "Inspect model comparison tables, cross-validation RMSE, tuned results, confusion matrix, and residual analysis."),
"08_explainability.ipynb": ("Explainability", "Inspect permutation importance, global SHAP summary, and a local SHAP explanation. Explanations are not causal."),
}

for filename, (title, description) in sections.items():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n", f"{description}\n", "\n", "This notebook is part of a reproducible project. Run from the repository root after `python scripts/train.py` and `python scripts/run_analysis.py`.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\n", "import sys\n", "import pandas as pd\n", "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n", "sys.path.insert(0, str(ROOT))\n", "DATA = ROOT / 'data' / 'processed'\n", "df = pd.read_csv(DATA / 'student_performance_clean.csv')\n", "df.head()\n"]},
    ]
    if filename == "03_eda.ipynb":
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["from IPython.display import Image, display\n", "for name in ['grade_distribution.png', 'absences_vs_grade.png', 'studytime_vs_grade.png', 'correlation_matrix.png']:\n", "    display(Image(filename=str(ROOT / 'reports' / 'figures' / name)))\n"]})
    elif filename == "04_statistical_analysis.ipynb":
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["pd.read_csv(DATA / 'statistical_tests.csv')\n"]})
    elif filename == "07_model_evaluation.ipynb":
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["for name in ['tuned_regression_results.csv', 'tuned_classification_results.csv', 'classification_report.csv', 'confusion_matrix.csv']:\n", "    print(name); display(pd.read_csv(DATA / name))\n"]})
    elif filename == "08_explainability.ipynb":
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["display(pd.read_csv(DATA / 'permutation_importance.csv').head(15))\n", "display(pd.read_csv(DATA / 'local_shap_explanation.csv'))\n"]})
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    (NOTEBOOKS / filename).write_text(json.dumps(notebook, indent=2))
