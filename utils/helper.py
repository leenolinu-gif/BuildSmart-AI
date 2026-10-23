```python
from pathlib import Path
import logging
from datetime import datetime

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
DATASET_PATH = BASE_DIR / "construction_dataset.xlsx"
METRICS_PATH = BASE_DIR / "model_metrics.json"
MODEL_PATH = BASE_DIR / "best_model.pkl"

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
```
