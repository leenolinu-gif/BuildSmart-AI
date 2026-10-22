"""BuildSmart AI Streamlit application."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.helper import (
    DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGETS,
    budget_category, configure_logging, money, timestamp,
)
from utils.preprocessing import load_or_create_dataset
from utils.prediction import load_model, predict_project, generate_recommendations
from utils.visualization import cost_breakdown_chart, model_comparison_chart, dashboard_charts
from utils.report_generator import create_pdf_report

configure_logging()
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="BuildSmart AI", page_icon="🏗️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 45%, #F7FAFF 100%); }
.hero { padding: 1.7rem 1.8rem; border-radius: 22px; color: white;
        background: linear-gradient(120deg, #123B70, #2563EB 65%, #20A4B8);
        box-shadow: 0 12px 32px rgba(37,99,235,.18); margin-bottom: 1.2rem; }
.hero h1 { color: white; font-size: 2.4rem; margin: 0; font-weight: 800; }
.hero p { color: #EAF2FF; font-size: 1.02rem; margin: .55rem 0 0 0; }
.metric-card { background: white; border: 1px solid #E4ECF7; border-radius: 16px;
               padding: 1rem 1.1rem; box-shadow: 0 5px 20px rgba(20,50,90,.05); min-height: 100px; }
.metric-label { color: #64748B; font-size: .85rem; font-weight: 600; }
.metric-value { color: #123B70; font-size: 1.5rem; font-weight: 800; margin-top: .35rem; }
.info-card { background: white; border: 1px solid #E4ECF7; border-radius: 14px;
             padding: 1rem; margin: .35rem 0; }
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 700; }
section[data-testid="stSidebar"] { background: #F0F6FF; }
small { color: #64748B; }
</style>
""", unsafe_allow_html=True)

def metric_card(label: str, value: str) -> None:
    """Render a consistent metric card."""
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Load or create the demonstration dataset."""
    return load_or_create_dataset()

def get_metrics() -> pd.DataFrame:
    """Load metrics from the trained model bundle or metrics file."""
    bundle = load_model()
    if bundle and bundle.get("metrics"):
        return pd.DataFrame(bundle["metrics"])
    if METRICS_PATH.exists():
        try:
            return pd.DataFrame(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            LOGGER.exception("Could not read metrics file.")
    return pd.DataFrame()

def show_header(title: str, subtitle: str) -> None:
    """Render the application page header."""
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def show_home(df: pd.DataFrame, model_bundle: dict | None) -> None:
    show_header("BuildSmart AI", "Intelligent Construction Cost Estimation & Economic Analysis")
    st.markdown("### Plan with cost clarity. Build with confidence.")
    st.write("An AI-assisted decision-support prototype for early-stage construction budgeting, "
             "economic indicators, and transparent cost recommendations.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Projects in dataset", f"{len(df):,}")
    with c2: metric_card("Average project cost", money(df["Total Cost"].mean()))
    with c3: metric_card("Average cost / sq. ft.", money((df["Total Cost"] / df["Building Area"]).mean()))
    with c4: metric_card("Model status", "Ready" if model_bundle else "Train required")
    st.markdown("### Explore the platform")
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="info-card"><h4>🎯 Cost Prediction</h4><p>Estimate cost components and compare your budget against the predicted total.</p></div>', unsafe_allow_html=True)
    with cards[1]:
        st.markdown('<div class="info-card"><h4>📊 Analytics</h4><p>Explore cost patterns by building type, location, material quality, and budget category.</p></div>', unsafe_allow_html=True)
    with cards[2]:
        st.markdown('<div class="info-card"><h4>🧠 Model Insights</h4><p>Compare candidate regressors and inspect held-out evaluation metrics.</p></div>', unsafe_allow_html=True)
    st.info("Demonstration data are synthetic. Use validated project quantities, current regional rates, and professional review before making financial commitments.")
    if not model_bundle:
        st.warning("No trained model was found. Run `python train_model.py` from the project directory, then refresh this page.")

def show_prediction(model_bundle: dict | None) -> None:
    show_header("Cost Prediction", "Configure the project to generate an indicative estimate.")
    if not model_bundle:
        st.error("The model is not trained yet. Open a terminal in the project folder and run `python train_model.py`.")
        return
    with st.form("prediction_form"):
        st.subheader("Project specifications")
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Building Area (sq. ft.)", min_value=300, max_value=500000, value=1800, step=100,
                                   help="Total built-up floor area in square feet.")
            floors = st.number_input("Number of Floors", min_value=1, max_value=60, value=2)
            building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
            quality = st.selectbox("Material Quality", ["Basic", "Standard", "Premium", "Luxury"], index=1)
        with col2:
            location = st.selectbox("Project Location", ["Urban", "Semi Urban", "Rural"])
            foundation = st.selectbox("Foundation Type", ["Concrete", "Pile", "Raft"])
            roof = st.selectbox("Roof Type", ["Flat", "Sloped", "Metal"])
            duration = st.number_input("Construction Duration (months)", min_value=1, max_value=120, value=12)
        with col3:
            inflation = st.slider("Inflation Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=.25)
            material_index = st.number_input("Material Price Index", min_value=50.0, max_value=250.0, value=115.0, step=1.0,
                                              help="Index relative to a chosen base period; dataset reference values are synthetic.")
            labour_rate = st.number_input("Labour Rate (₹/day)", min_value=200.0, max_value=5000.0, value=1000.0, step=50.0)
            equipment_input = st.number_input("Equipment Cost Input (₹)", min_value=0.0, max_value=100000000.0, value=750000.0, step=50000.0)
        budget = st.number_input("Expected Budget (₹)", min_value=0.0, max_value=1000000000.0, value=5000000.0, step=100000.0,
                                 help="Enter 0 if no budget has been set.")
        submitted = st.form_submit_button("✨ Predict Construction Cost", type="primary", use_container_width=True)
    if submitted:
        inputs = {
            "Building Area": float(area), "Floors": int(floors), "Building Type": building_type,
            "Material Quality": quality, "Location": location, "Foundation Type": foundation,
            "Roof Type": roof, "Construction Duration": int(duration), "Inflation Rate": float(inflation),
            "Material Price Index": float(material_index), "Labour Rate": float(labour_rate),
            "Equipment Cost Input": float(equipment_input),
        }
        # The model's schema treats Equipment Cost as a numeric input. Preserve user input above.
        with st.spinner("Analysing project characteristics and estimating cost components..."):
            results = predict_project(model_bundle, inputs)
            recommendations = generate_recommendations({**inputs, "Expected Budget": budget}, results)
        results["Budget Utilization %"] = results["Total Cost"] / budget * 100 if budget else None
        st.session_state["last_prediction"] = {"inputs": inputs, "results": results, "recommendations": recommendations}
        history = st.session_state.get("prediction_history", [])
        history.append({"Timestamp": timestamp(), "Building Type": building_type, "Location": location,
                        "Area (sq. ft.)": area, "Estimated Total Cost": results["Total Cost"],
                        "Cost / sq. ft.": results["Cost Per Square Foot"], "Risk": results["Risk Level"]})
        st.session_state["prediction_history"] = history[-20:]
        st.success("Estimate generated successfully.")
    report = st.session_state.get("last_prediction")
    if report:
        inputs, results, recommendations = report["inputs"], report["results"], report["recommendations"]
        st.markdown("---")
        st.subheader("Estimated project cost")
        a, b, c, d = st.columns(4)
        with a: metric_card("Total construction cost", money(results["Total Cost"]))
        with b: metric_card("Recommended budget (+10%)", money(results["Recommended Project Budget"]))
        with c: metric_card("Cost per sq. ft.", money(results["Cost Per Square Foot"]))
        with d: metric_card("Budget category", results["Budget Category"])
        x, y = st.columns([1.1, 1])
        with x:
            st.plotly_chart(cost_breakdown_chart(results), use_container_width=True)
        with y:
            st.markdown("#### Cost components")
            for label, key in [("Materials", "Material Cost"), ("Labour", "Labour Cost"),
                               ("Equipment", "Equipment Cost"), ("Miscellaneous", "Miscellaneous Cost")]:
                st.write(f"**{label}:** {money(results[key])}")
            st.write(f"**Budget utilization:** {results['Budget Utilization %']:.1f}%" if results["Budget Utilization %"] is not None else "**Budget utilization:** Not provided")
            st.write(f"**Cost efficiency score:** {results['Cost Efficiency Score']}/100")
            st.write(f"**Project risk:** {results['Risk Level']}")
            st.progress(min(results["Cost Efficiency Score"] / 100, 1.0), text="Cost efficiency indicator")
        st.markdown("#### Economic indicators")
        e1, e2, e3, e4 = st.columns(4)
        with e1: metric_card("Materials share", f"{results['Material Percentage']:.1f}%")
        with e2: metric_card("Labour share", f"{results['Labour Percentage']:.1f}%")
        with e3: metric_card("Equipment share", f"{results['Equipment Percentage']:.1f}%")
        with e4: metric_card("Estimated inflation exposure", money(results["Inflation Impact Estimate"]))
        st.markdown("#### AI-assisted recommendations")
        for recommendation in recommendations:
            st.markdown(f'<div class="info-card">💡 {recommendation}</div>', unsafe_allow_html=True)
        pdf_bytes = create_pdf_report(inputs, results, recommendations)
        st.download_button("📄 Download PDF Estimate Report", data=pdf_bytes,
                           file_name="buildsmart_cost_estimate.pdf", mime="application/pdf",
                           use_container_width=True)
        if st.button("Reset current estimate"):
            st.session_state.pop("last_prediction", None)
            st.rerun()

def show_dashboard(df: pd.DataFrame) -> None:
    show_header("Analytics Dashboard", "Explore the synthetic project portfolio and cost drivers.")
    df = df.copy()
    df["Cost Per Square Foot"] = df["Total Cost"] / df["Building Area"].clip(lower=1)
    high_idx, low_idx = df["Total Cost"].idxmax(), df["Total Cost"].idxmin()
    common_type = df["Building Type"].mode().iloc[0]
    location_avg = df.groupby("Location")["Total Cost"].mean()
    cols = st.columns(4)
    with cols[0]: metric_card("Average construction cost", money(df["Total Cost"].mean()))
    with cols[1]: metric_card("Average material cost", money(df["Material Cost"].mean()))
    with cols[2]: metric_card("Average labour cost", money(df["Labour Cost"].mean()))
    with cols[3]: metric_card("Average cost / sq. ft.", money(df["Cost Per Square Foot"].mean()))
    cols2 = st.columns(4)
    with cols2[0]: metric_card("Highest-cost project", money(df.loc[high_idx, "Total Cost"]))
    with cols2[1]: metric_card("Lowest-cost project", money(df.loc[low_idx, "Total Cost"]))
    with cols2[2]: metric_card("Most expensive location", str(location_avg.idxmax()))
    with cols2[3]: metric_card("Most common building type", str(common_type))
    charts = dashboard_charts(df)
    for left, right, first, second in [
        (0, 1, "budget", "type"), (0, 1, "location", "quality"),
        (0, 1, "hist", "sqft"), (0, 1, "trend", "corr")
    ]:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(charts[first], use_container_width=True)
        with c2: st.plotly_chart(charts[second], use_container_width=True)
    st.markdown("#### Most economical location by average cost")
    st.success(f"{location_avg.idxmin()} has the lowest average total cost in this synthetic dataset ({money(location_avg.min())}). This reflects the generated assumptions, not a universal market ranking.")
    with st.expander("Preview dataset"):
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"),
                           "construction_dataset_preview.csv", "text/csv")

def show_model_performance(model_bundle: dict | None) -> None:
    show_header("Model Performance", "Compare candidate regression models using held-out test data.")
    metrics = get_metrics()
    if metrics.empty:
        st.warning("Model evaluation is not available. Run `python train_model.py` to train and evaluate the models.")
        return
    if model_bundle:
        st.success(f"Selected model: {model_bundle.get('best_model_name', 'Unknown')}")
        st.caption(f"Training dataset rows: {model_bundle.get('dataset_rows', 'Unknown')} · Trained at: {model_bundle.get('trained_at', 'Unknown')}")
    st.dataframe(metrics.style.format({"MAE": "₹{:,.0f}", "MSE": "₹{:,.0f}", "RMSE": "₹{:,.0f}",
                                       "R2 Score": "{:.4f}", "Training Time (s)": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)
    if "R2 Score" in metrics:
        fig = px.bar(metrics.sort_values("R2 Score", ascending=False), x="Model", y="R2 Score",
                     title="R² score comparison (higher is better)")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Metric guide")
    st.write("- **MAE:** average absolute prediction error, in the target currency units.")
    st.write("- **MSE:** average squared error; larger errors receive a greater penalty.")
    st.write("- **RMSE:** square root of MSE, in the same units as the predicted costs.")
    st.write("- **R²:** proportion of target variation explained by the model; closer to 1 is generally better.")
    st.caption("Metrics are computed on a held-out 20% test split. They describe performance on the synthetic dataset and do not establish real-world accuracy.")

def show_reports() -> None:
    show_header("Report Generation", "Create a professional PDF from your latest estimate.")
    report = st.session_state.get("last_prediction")
    if not report:
        st.info("Generate a cost prediction first. Your most recent estimate will appear here for PDF export.")
        return
    st.write("The latest estimate is ready to export.")
    st.json({"Total Cost": report["results"]["Total Cost"], "Recommended Project Budget": report["results"]["Recommended Project Budget"],
             "Risk Level": report["results"]["Risk Level"], "Model": report["results"]["Model Name"]})
    pdf_bytes = create_pdf_report(report["inputs"], report["results"], report["recommendations"])
    st.download_button("Download professional PDF report", pdf_bytes, "buildsmart_project_report.pdf",
                       "application/pdf", use_container_width=True)

def show_about(df: pd.DataFrame) -> None:
    show_header("About BuildSmart AI", "A portfolio-ready construction economics decision-support prototype.")
    st.markdown("""
### What it does
BuildSmart AI estimates construction cost components from project characteristics and provides
budget utilization, cost intensity, escalation exposure, risk signals, and rule-based recommendations.

### Intended users
Homeowners, contractors, civil engineers, architects, project managers, quantity surveyors, and
construction firms planning early-stage budgets.

### Method
1. A synthetic dataset is generated when the Excel file is absent.
2. Numeric and categorical fields are imputed, scaled or one-hot encoded in a scikit-learn pipeline.
3. Four regression approaches are compared on an 80/20 train/test split.
4. The best model is selected using RMSE, with MAE, MSE, and R² also reported.
5. The prediction is converted into cost components and economic indicators.

### Limitations and responsible use
- The bundled dataset is synthetic, not observed market data.
- Cost estimates depend on assumptions and may not capture design complexity, soil conditions,
  local taxes, permits, utilities, contractor overheads, contingency, or current quotations.
- The cost efficiency score is a heuristic indicator, not an industry-certified benchmark.
- Never use the estimate alone for tendering, lending, contractual commitments, or safety-critical decisions.
- Validate with local schedule-of-rates data, a bill of quantities, and a qualified cost professional.

### Future scope
- Real-time material-price API and regional schedule-of-rates integration
- Weather and site-condition integration
- GST and government tax calculations
- Cloud deployment, authentication, database storage, and audit trails
- Multi-currency support and historical project comparison
- Conversational AI assistant for construction planning
- Geospatial location adjustment and uncertainty intervals
""")
    st.markdown(f"**Dataset rows:** {len(df):,} · **Dataset path:** `data/construction_dataset.xlsx`")
    st.markdown("**Technology:** Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · Matplotlib · Joblib · openpyxl · ReportLab")

def main() -> None:
    """Run the Streamlit application."""
    try:
        df = get_dataset()
    except Exception as exc:
        LOGGER.exception("Dataset initialization failed.")
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
    model_bundle = load_model()
    with st.sidebar:
        st.markdown("## 🏗️ BuildSmart AI")
        st.caption("Construction economics platform")
        page = st.radio("Navigation", ["Home", "Cost Prediction", "Analytics Dashboard",
                                      "Model Performance", "Report Generation", "About Project"])
        st.markdown("---")
        st.markdown("**Project health**")
        st.write("🟢 Dataset available" if len(df) else "🔴 Dataset unavailable")
        st.write("🟢 Model trained" if model_bundle else "🟠 Model needs training")
        st.caption("Illustrative estimate only · Not professional cost advice")
        if st.button("Refresh application", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    if page == "Home":
        show_home(df, model_bundle)
    elif page == "Cost Prediction":
        show_prediction(model_bundle)
    elif page == "Analytics Dashboard":
        show_dashboard(df)
    elif page == "Model Performance":
        show_model_performance(model_bundle)
    elif page == "Report Generation":
        show_reports()
    else:
        show_about(df)
    history = st.session_state.get("prediction_history", [])
    if history:
        with st.sidebar.expander(f"Recent predictions ({len(history)})"):
            st.dataframe(pd.DataFrame(history).tail(10), hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
