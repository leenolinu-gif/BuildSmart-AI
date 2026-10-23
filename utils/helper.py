
"""Shared constants and helper utilities for BuildSmart AI."""

from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd

# Project root: parent directory of the utils folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths: files currently stored in the repository root
DATASET_PATH = PROJECT_ROOT / "construction_dataset.xlsx"
MODEL_PATH = PROJECT_ROOT / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model_metrics.json"

# Feature definitions used by the current project
FEATURES = [
    "Area_sqft",
    "Location",
    "Building_Type",
    "Quality",
]

TARGETS = [
    "Material_Cost",
    "Labor_Cost",
    "Equipment_Cost",
    "Total_Cost",
]

# Compatibility constants for modules that use the newer schema
NUMERIC_FEATURES = ["Area_sqft"]
CATEGORICAL_FEATURES = [
    "Location",
    "Building_Type",
    "Quality",
]

BUDGET_CATEGORIES = ["Low Budget", "Medium Budget", "High Budget"]


def configure_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def budget_category(cost, area=None):
    """Classify a construction project by budget or cost per area."""
    if area is not None:
        if area <= 0:
            return "High Budget"
        rate = cost / area

        if rate < 1800:
            return "Low Budget"
        elif rate < 3200:
            return "Medium Budget"
        return "High Budget"

    if cost < 1000000:
        return "Low Budget"
    elif cost < 5000000:
        return "Medium Budget"
    return "High Budget"


def money(value, currency="₹"):
    """Format a value as Indian currency."""
    return f"{currency}{value:,.2f}"


def timestamp():
    """Return the current local timestamp."""
    return datetime.now().astimezone().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


def safe_float(value, default=0.0):
    """Convert a value to a finite float safely."""
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def ensure_directories():
    """Create runtime directories if required."""
    return None


def clean_dataframe(df):
    """Remove duplicates and fill missing values."""
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
