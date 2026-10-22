"""Dataset creation and preprocessing helpers."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.helper import (
    CATEGORICAL_FEATURES, DATASET_PATH, FEATURES, NUMERIC_FEATURES, TARGETS,
    clean_dataframe, ensure_directories,
)

LOGGER = logging.getLogger(__name__)

def generate_synthetic_dataset(rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible, internally consistent synthetic project dataset.

    Values are illustrative and should not be treated as a substitute for local
    quantity surveying, contractor quotations, or a validated market database.
    """
    rng = np.random.default_rng(seed)
    area = rng.integers(650, 18001, rows).astype(float)
    floors = rng.integers(1, 16, rows)
    building_type = rng.choice(["Residential", "Commercial", "Industrial"], rows, p=[.62, .25, .13])
    quality = rng.choice(["Basic", "Standard", "Premium", "Luxury"], rows, p=[.18, .45, .27, .10])
    location = rng.choice(["Urban", "Semi Urban", "Rural"], rows, p=[.45, .35, .20])
    foundation = rng.choice(["Concrete", "Pile", "Raft"], rows, p=[.62, .22, .16])
    roof = rng.choice(["Flat", "Sloped", "Metal"], rows, p=[.58, .27, .15])
    duration = np.clip(np.round(area / 1400 + floors * 1.7 + rng.normal(3, 2.5, rows)), 3, 48).astype(int)
    inflation = np.round(rng.uniform(2.0, 9.0, rows), 2)
    material_index = np.round(rng.uniform(90, 155, rows), 2)
    labour_rate = np.round(rng.uniform(650, 1600, rows), 2)
    equipment_input = np.round(rng.uniform(250000, 3500000, rows), 2)

    type_factor = pd.Series(building_type).map({"Residential": 1.0, "Commercial": 1.22, "Industrial": 1.12}).to_numpy()
    quality_factor = pd.Series(quality).map({"Basic": .82, "Standard": 1.0, "Premium": 1.28, "Luxury": 1.62}).to_numpy()
    location_factor = pd.Series(location).map({"Urban": 1.20, "Semi Urban": 1.0, "Rural": .86}).to_numpy()
    foundation_factor = pd.Series(foundation).map({"Concrete": 1.0, "Pile": 1.18, "Raft": 1.12}).to_numpy()
    roof_factor = pd.Series(roof).map({"Flat": 1.0, "Sloped": 1.06, "Metal": .94}).to_numpy()

    base_rate = 1050 * type_factor * quality_factor * location_factor * foundation_factor * roof_factor
    inflation_factor = 1 + inflation / 100 * (duration / 12) * .45
    material_cost = area * base_rate * (material_index / 115) * .48 * inflation_factor
    labour_cost = area * base_rate * .29 * (labour_rate / 1000) * location_factor * inflation_factor
    equipment_cost_target = equipment_input * (1 + floors * .035) + area * base_rate * .075
    miscellaneous_cost = (material_cost + labour_cost + equipment_cost_target) * rng.uniform(.075, .14, rows)
    noise = rng.normal(1.0, .045, rows)
    material_cost = np.maximum(100000, material_cost * noise).round(2)
    labour_cost = np.maximum(60000, labour_cost * noise).round(2)
    equipment_cost_target = np.maximum(25000, equipment_cost_target * rng.normal(1, .06, rows)).round(2)
    miscellaneous_cost = np.maximum(20000, miscellaneous_cost * noise).round(2)
    total = material_cost + labour_cost + equipment_cost_target + miscellaneous_cost
    rate = total / area
    category = np.where(rate < 1800, "Low", np.where(rate < 3200, "Medium", "High"))

    df = pd.DataFrame({
        "Project ID": [f"BS-{i:05d}" for i in range(1, rows + 1)],
        "Building Area": area.round(0), "Floors": floors,
        "Building Type": building_type, "Material Quality": quality,
        "Location": location, "Foundation Type": foundation, "Roof Type": roof,
        "Construction Duration": duration, "Inflation Rate": inflation,
        "Material Price Index": material_index, "Labour Rate": labour_rate,
        "Equipment Cost Input": equipment_input, "Material Cost": material_cost,
        "Labour Cost": labour_cost, "Equipment Cost": equipment_cost_target,
        "Miscellaneous Cost": miscellaneous_cost, "Total Cost": total.round(2),
        "Budget Category": category,
    })
    return df

def save_dataset(path=DATASET_PATH, rows: int = 10000) -> pd.DataFrame:
    """Create and save the synthetic dataset as an Excel workbook."""
    ensure_directories()
    df = generate_synthetic_dataset(rows=rows)
    df.to_excel(path, index=False, engine="openpyxl")
    LOGGER.info("Saved %s rows to %s", len(df), path)
    return df

def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing pipeline with imputation, scaling, and one-hot encoding."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])

def load_or_create_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Load the Excel dataset or create it automatically when missing."""
    ensure_directories()
    if path.exists():
        try:
            df = pd.read_excel(path, engine="openpyxl")
            if len(df) >= 10000:
                return clean_dataframe(df)
            LOGGER.warning("Dataset contains fewer than 10,000 rows; regenerating it.")
        except (ValueError, OSError, ImportError) as exc:
            LOGGER.warning("Could not read dataset (%s); regenerating it.", exc)
    return clean_dataframe(save_dataset(path=path, rows=10000))
