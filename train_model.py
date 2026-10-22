"""Train and evaluate candidate construction cost regression models.

Run from the project root:
    python train_model.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from utils.helper import (
    FEATURES, TARGETS, MODEL_PATH, METRICS_PATH, ensure_directories, configure_logging,
)
from utils.preprocessing import build_preprocessor, load_or_create_dataset

LOGGER = logging.getLogger(__name__)

def main() -> None:
    """Train candidate models, compare held-out metrics, and persist the winner."""
    configure_logging()
    ensure_directories()
    df = load_or_create_dataset()
    df = df.dropna(subset=FEATURES + TARGETS).copy()
    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=18, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=22, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=80, learning_rate=.06, max_depth=3, random_state=42)
        ),
    }
    results = []
    trained = {}
    for name, estimator in candidates.items():
        started = time.perf_counter()
        pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, prediction, multioutput="uniform_average")
        elapsed = time.perf_counter() - started
        results.append({
            "Model": name, "MAE": float(mae), "MSE": float(mse),
            "RMSE": rmse, "R2 Score": float(r2), "Training Time (s)": round(elapsed, 2),
        })
        trained[name] = pipeline
        LOGGER.info("%s | MAE %.2f | RMSE %.2f | R2 %.4f", name, mae, rmse, r2)

    metrics = pd.DataFrame(results).sort_values(["RMSE", "MAE"])
    best_name = str(metrics.iloc[0]["Model"])
    bundle = {
        "model": trained[best_name],
        "best_model_name": best_name,
        "features": FEATURES,
        "targets": TARGETS,
        "metrics": metrics.to_dict(orient="records"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": int(len(df)),
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    LOGGER.info("Best model: %s. Saved to %s", best_name, MODEL_PATH)
    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nBest model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
