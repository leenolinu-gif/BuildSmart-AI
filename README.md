# BuildSmart AI
### Intelligent Construction Cost Estimation & Economic Analysis Platform

BuildSmart AI is a Streamlit-based construction economics decision-support prototype. It estimates construction cost components from project attributes, compares regression models, visualizes cost drivers, produces transparent recommendations, and exports a PDF report.

> **Important:** The included dataset is synthetic. Predictions are for learning and demonstration, not a contractor quotation, bill of quantities, tender price, or professional cost certification. Validate any real project estimate with local rates, quantities, design details, and a qualified cost professional.

## Features

- Modern Streamlit interface with navigation for Home, Cost Prediction, Analytics Dashboard, Model Performance, Report Generation, and About.
- Automatically generates a 10,000-row Excel dataset if the dataset is missing or undersized.
- Trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- Evaluates models with MAE, MSE, RMSE, R², and training time; saves the best model using Joblib.
- Predicts material, labour, equipment, and miscellaneous costs.
- Calculates total cost, recommended budget, cost per square foot, budget utilization, budget category, component shares, inflation exposure, a heuristic efficiency score, and risk level.
- Rule-based recommendations that explain relevant cost and budget considerations.
- Interactive Plotly charts and downloadable PDF estimate reports.
- Prediction history for the current Streamlit session.
- Modular utilities, logging, exception handling, type hints, and documentation.

## Technology stack

- Python 3.13+
- pandas, NumPy
- scikit-learn
- Streamlit
- Plotly, Matplotlib
- Joblib, openpyxl
- ReportLab for PDF generation
- Git and GitHub

## Project structure

```text
BuildSmart_AI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── construction_dataset.xlsx
├── models/
│   ├── best_model.pkl
│   └── model_metrics.json
├── reports/
├── images/
├── assets/
└── utils/
    ├── __init__.py
    ├── helper.py
    ├── preprocessing.py
    ├── prediction.py
    ├── visualization.py
    └── report_generator.py
```

## Installation

### 1. Install Python

Install Python 3.13 or a compatible supported version and verify:

```bash
python --version
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the models

From the project root:

```bash
python train_model.py
```

The training script creates `data/construction_dataset.xlsx` when needed, trains four regression candidates using an 80/20 split, evaluates them, and writes `models/best_model.pkl` and `models/model_metrics.json`.

### 5. Launch the app

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## Dataset fields

The synthetic dataset includes project ID, building area, floors, building type, material quality, location, foundation type, roof type, duration, inflation rate, material price index, labour rate, equipment cost input, cost components, total cost, and budget category.

The dataset uses plausible illustrative ranges and simplified cost relationships. It does not encode a specific state schedule of rates or current market quotations. All generated values should be treated as example data.

## Model methodology

- Numeric features: median imputation and standard scaling.
- Categorical features: most-frequent imputation and one-hot encoding.
- Train/test split: 80% training and 20% test data with a fixed random seed.
- Candidate estimators: Linear Regression, Decision Tree Regressor, Random Forest Regressor, and multi-output Gradient Boosting.
- Model selection: lowest test-set RMSE, with MAE, MSE, R², and training time also reported.
- Targets: material cost, labour cost, equipment cost, and miscellaneous cost. Total cost is the sum of predicted components.

A lower RMSE/MAE is generally better; a higher R² is generally better. Metrics on synthetic data do not establish accuracy on real projects.

## Screenshots

Add screenshots of the Home, Cost Prediction, Analytics Dashboard, and Model Performance pages to the `images/` folder before publishing. The app itself runs without screenshots.

## GitHub setup

```bash
git init
git add .
git commit -m "Build initial BuildSmart AI platform"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/BuildSmart_AI.git
git push -u origin main
```

Review the generated dataset and model size before pushing if your repository has storage limits. Never commit credentials, API keys, or private project information.

## Future scope

- Real-time material price APIs and region-specific schedule-of-rates data
- Weather, logistics, and site-condition integration
- GST and applicable government tax modules
- Cloud deployment and CI checks
- User authentication, database storage, and estimate audit trails
- Multi-currency support
- Historical project comparisons and calibrated local models
- Uncertainty intervals and sensitivity analysis
- AI chat assistant for construction planning

## License

This project is released under the MIT License. Add a `LICENSE` file with the standard MIT license text and your chosen copyright holder before public distribution.

## Author

BuildSmart AI — Construction Economics & AI Portfolio Project.
