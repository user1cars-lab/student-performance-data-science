from __future__ import annotations

from pathlib import Path
import zipfile
import urllib.request

import pandas as pd

from src.config import FEATURES, RAW_DIR, PROCESSED_DIR

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
RAW_FILE = RAW_DIR / "student-mat.csv"


def download_dataset(destination: Path = RAW_FILE) -> Path:
    """Download and extract the official UCI Student Performance dataset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    archive = destination.parent / "student-performance.zip"
    urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        nested_name = next(name for name in zf.namelist() if name.endswith("student.zip"))
        nested_archive = zf.read(nested_name)
    with zipfile.ZipFile(__import__("io").BytesIO(nested_archive)) as nested:
        member = next(name for name in nested.namelist() if name.endswith("student-mat.csv"))
        with nested.open(member) as source, destination.open("wb") as target:
            target.write(source.read())
    archive.unlink(missing_ok=True)
    return destination


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the semicolon-delimited mathematics data."""
    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, remove duplicate rows, and create documented target labels."""
    result = df.copy()
    result.columns = result.columns.str.strip().str.lower()
    required = set(FEATURES + ["g1", "g2", "g3"])
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = result.drop_duplicates().reset_index(drop=True)
    result["performance_level"] = pd.cut(
        result["g3"], bins=[-1, 9, 14, 20], labels=["Low", "Medium", "High"]
    ).astype(str)
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features available before the final grade; G1/G2 are intentionally excluded."""
    result = df[FEATURES + ["g3", "performance_level"]].copy()
    result["study_engagement"] = result["studytime"] * (result["higher"] == "yes").astype(int)
    result["absence_rate_proxy"] = result["absences"].clip(lower=0)
    return result


def prepare_dataset() -> pd.DataFrame:
    """Download, clean, engineer, and persist the modeling dataset."""
    data = build_features(clean_data(load_raw(download_dataset())))
    data.to_csv(PROCESSED_DIR / "student_performance_clean.csv", index=False)
    return data
