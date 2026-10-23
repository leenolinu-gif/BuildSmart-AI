"""Shared constants and helper utilities for BuildSmart AI."""

from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths: matches files in your repository's root folder
DATASET_PATH = PROJECT_ROOT / "construction_dataset.xlsx"
MODEL_PATH = PROJECT_ROOT / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "model_metrics.json"

# Features and targets
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


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def budget_category(cost):
    if cost < 1000000:
        return "Low Budget"
    elif cost < 5000000:
        return "Medium Budget"
    return "High Budget"


def money(value):
    return f"₹{value:,.2f}"


def timestamp():
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def ensure_directories():
    return None


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
