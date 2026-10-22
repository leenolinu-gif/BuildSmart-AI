"""Reusable Plotly chart builders."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#2563EB"
NAVY = "#102A43"
TEAL = "#0F9D9A"

def cost_breakdown_chart(results: dict):
    """Return a donut chart of predicted cost components."""
    labels = ["Materials", "Labour", "Equipment", "Miscellaneous"]
    values = [results["Material Cost"], results["Labour Cost"], results["Equipment Cost"], results["Miscellaneous Cost"]]
    fig = px.pie(names=labels, values=values, hole=.58, title="Estimated Cost Breakdown")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=55, b=10, l=10, r=10), legend_title_text="")
    return fig

def model_comparison_chart(metrics: pd.DataFrame):
    """Return a model comparison chart using RMSE."""
    if metrics.empty or "RMSE" not in metrics:
        return go.Figure().update_layout(title="Model metrics are not available")
    fig = px.bar(metrics.sort_values("RMSE"), x="Model", y="RMSE", title="Model Comparison — RMSE (lower is better)")
    fig.update_layout(xaxis_title="", yaxis_title="RMSE (₹)", margin=dict(t=55, b=10))
    return fig

def dashboard_charts(df: pd.DataFrame) -> dict:
    """Build the collection of charts used by the analytics dashboard."""
    charts = {}
    charts["budget"] = px.pie(df, names="Budget Category", title="Budget Category Distribution", hole=.45)
    charts["type"] = px.box(df, x="Building Type", y="Total Cost", color="Building Type", title="Cost by Building Type")
    charts["location"] = px.bar(df.groupby("Location", as_index=False)["Total Cost"].mean(),
                                x="Location", y="Total Cost", title="Average Cost by Location")
    charts["quality"] = px.bar(df.groupby("Material Quality", as_index=False)["Total Cost"].mean(),
                               x="Material Quality", y="Total Cost", title="Average Cost by Material Quality")
    charts["hist"] = px.histogram(df, x="Total Cost", nbins=40, title="Construction Cost Distribution")
    charts["sqft"] = px.box(df, x="Building Type", y=df["Total Cost"] / df["Building Area"],
                            title="Cost per Square Foot by Building Type")
    charts["trend"] = px.line(df.head(300).assign(Project_Number=range(1, min(len(df), 300) + 1)),
                              x="Project_Number", y="Total Cost", title="Synthetic Project Cost Trend")
    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    charts["corr"] = px.imshow(corr, text_auto=False, aspect="auto", title="Numeric Feature Correlation",
                               color_continuous_scale="RdBu_r")
    return charts
