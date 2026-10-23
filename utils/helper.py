
"""Shared constants and helper utilities for BuildSmart AI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
DATASET_PATH = DATA_DIR / "construction_dataset.xlsx"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

CATEGORICAL_FEATURES = [
    "Building Type", "Material Quality", "Location", "Foundation Type", "Roof Type"
]
NUMERIC_FEATURES = [
    "Building Area", "Floors", "Construction Duration", "Inflation Rate",
    "Material Price Index", "Labour Rate", "Equipment Cost Input"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["Material Cost", "Labour Cost", "Equipment Cost", "Miscellaneous Cost"]
BUDGET_CATEGORIES = ["Low", "Medium", "High"]

def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def money(value: float, currency: str = "₹") -> str:
    """Format a numeric amount as Indian currency."""
    return f"{currency}{value:,.0f}"

def safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float without raising an exception."""
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default

def budget_category(total_cost: float, area: float) -> str:
    """Classify a project's cost intensity using an area-normalized benchmark."""
    if area <= 0:
        return "High"
    rate = total_cost / area
    if rate < 1800:
        return "Low"
    if rate < 3200:
        return "Medium"
    return "High"

def timestamp() -> str:
    """Return a readable local timestamp."""
    return datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z")

def ensure_directories() -> None:
    """Create expected runtime directories."""
    for path in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and fill missing values using robust defaults."""
    df = df.copy().drop_duplicates()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            median = df[column].median()
            df[column] = df[column].fillna(0 if pd.isna(median) else median)
        else:
            mode = df[column].mode(dropna=True)
            df[column] = df[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df
