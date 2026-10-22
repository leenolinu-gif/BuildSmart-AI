"""Model loading, prediction, economic indicators, and recommendations."""
from __future__ import annotations

import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from utils.helper import MODEL_PATH, FEATURES, TARGETS, budget_category, money

LOGGER = logging.getLogger(__name__)

def load_model(model_path: Path = MODEL_PATH):
    """Load the trained model bundle, returning None if it is unavailable."""
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except (OSError, ValueError, EOFError) as exc:
        LOGGER.exception("Unable to load model: %s", exc)
        return None

def predict_project(model_bundle: dict, inputs: dict) -> dict:
    """Predict cost components and calculate decision-support indicators."""
    row = pd.DataFrame([{key: inputs[key] for key in FEATURES}])
    predictions = model_bundle["model"].predict(row)[0]
    costs = {target: max(0.0, float(value)) for target, value in zip(TARGETS, predictions)}
    total = sum(costs.values())
    area = max(float(inputs["Building Area"]), 1.0)
    expected_budget = max(float(inputs.get("Expected Budget", 0)), 0.0)
    cost_per_sqft = total / area
    utilization = total / expected_budget * 100 if expected_budget else None
    category = budget_category(total, area)
    material_pct = costs["Material Cost"] / total * 100 if total else 0
    labour_pct = costs["Labour Cost"] / total * 100 if total else 0
    equipment_pct = costs["Equipment Cost"] / total * 100 if total else 0
    misc_pct = costs["Miscellaneous Cost"] / total * 100 if total else 0
    inflation_impact = costs["Material Cost"] * (float(inputs["Inflation Rate"]) / 100) * (float(inputs["Construction Duration"]) / 12) * .45
    efficiency = int(np.clip(100 - (cost_per_sqft / 50) - max(0, float(inputs["Floors"]) - 3) * 1.3, 0, 100))
    risk = "High" if (utilization is not None and utilization > 100) or float(inputs["Inflation Rate"]) >= 7 else (
        "Medium" if (utilization is not None and utilization > 85) or float(inputs["Construction Duration"]) > 24 else "Low"
    )
    return {
        **costs, "Total Cost": total, "Recommended Project Budget": total * 1.10,
        "Budget Category": category, "Cost Per Square Foot": cost_per_sqft,
        "Budget Utilization %": utilization, "Material Percentage": material_pct,
        "Labour Percentage": labour_pct, "Equipment Percentage": equipment_pct,
        "Miscellaneous Percentage": misc_pct, "Inflation Impact Estimate": inflation_impact,
        "Cost Efficiency Score": efficiency, "Risk Level": risk,
        "Model Name": model_bundle.get("best_model_name", "Trained regression model"),
    }

def generate_recommendations(inputs: dict, results: dict) -> list[str]:
    """Generate transparent rule-based recommendations for the cost estimate."""
    recs = []
    if inputs["Material Quality"] in ("Premium", "Luxury"):
        recs.append("Premium or luxury specifications raise the estimate. Compare lifecycle performance before downgrading critical materials.")
    if inputs["Location"] == "Urban":
        recs.append("Urban labour, logistics, and site-access costs can be higher. Confirm local contractor rates and delivery constraints.")
    if inputs["Foundation Type"] == "Pile":
        recs.append("Pile foundations can materially affect cost. Validate the selection using a geotechnical investigation and structural design.")
    if float(inputs["Inflation Rate"]) >= 6:
        recs.append("The assumed inflation rate is elevated. Include escalation clauses and review long-lead material quotations.")
    if float(inputs["Construction Duration"]) > 24:
        recs.append("A long schedule increases exposure to escalation and overheads. Review critical-path activities and procurement packages.")
    if results.get("Budget Utilization %") is not None:
        if results["Budget Utilization %"] > 100:
            recs.append("The current expected budget is below the model estimate. Re-scope, reprice, or secure contingency before committing.")
        elif results["Budget Utilization %"] > 85:
            recs.append("The budget has limited headroom. Check contingency, taxes, approvals, and site-specific costs.")
        else:
            recs.append("The expected budget appears to cover the estimate. Keep a controlled contingency and update the estimate at each design stage.")
    if inputs["Material Quality"] in ("Luxury", "Premium"):
        recs.append("Compare alternative specifications and whole-life costs; do not compromise structural safety or applicable codes.")
    else:
        recs.append("Obtain at least three comparable quotations and check that material specifications, wastage, and scope are aligned.")
    recs.append("This is an indicative synthetic-data estimate, not a bill of quantities or contractor quotation. Have a qualified estimator validate it.")
    return recs
