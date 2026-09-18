from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42
TARGET = "G3"
CLASS_TARGET = "performance_level"
FEATURES = [
    "school", "sex", "age", "address", "famsize", "pstatus", "medu", "fedu",
    "mjob", "fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "dalc", "walc", "health", "absences"
]

for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)
