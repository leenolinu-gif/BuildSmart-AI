```python
"""Shared constants and helper utilities for BuildSmart AI."""

from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# File paths: match files currently in the repository root
DATASET_PATH = PROJECT_ROOT / "construction_dataset.xlsx"
MODEL_PATH = PROJECT_ROOT / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model_metrics.json"

# Feature names must match preprocessing.py and the dataset
NUMERIC_FEATURES = [
    "Building Area",
    "Floors",
    "Construction Duration",
    "Inflation Rate",
    "Material Price Index",
    "Labour Rate",
    "Equipment Cost Input",
]

CATEGORICAL_FEATURES = [
    "Building Type",
    "Material Quality",
    "Location",
    "Foundation Type",
    "Roof Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = [
    "Material Cost",
    "Labour Cost",
    "Equipment Cost",
    "Miscellaneous Cost",
]

BUDGET_CATEGORIES = ["Low", "Medium", "High"]


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def money(value, currency="₹"):
    return f"{currency}{value:,.0f}"


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def budget_category(total_cost, area):
    if area <= 0:
        return "High"

    rate = total_cost / area

    if rate < 1800:
        return "Low"
    elif rate < 3200:
        return "Medium"

    return "High"


def timestamp():
    return datetime.now().astimezone().strftime(
        "%d %b %Y, %I:%M %p %Z"
    )


def ensure_directories():
    for path in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_dataframe(df):
    df = df.copy().drop_duplicates()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            median = df[column].median()
            df[column] = df[column].fillna(
                0 if pd.isna(median) else median
            )
        else:
            mode = df[column].mode(dropna=True)
            df[column] = df[column].fillna(
                mode.iloc[0] if not mode.empty else "Unknown"
            )

    return df
```
