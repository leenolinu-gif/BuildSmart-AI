# BuildSmart AI — Step-by-Step Setup Guide

## Step 1: Extract the project
Extract `BuildSmart_AI.zip` and open the `BuildSmart_AI` folder in Visual Studio Code.

## Step 2: Open a terminal
In VS Code select **Terminal → New Terminal**. Make sure the terminal is inside the `BuildSmart_AI` project root.

## Step 3: Create a virtual environment
Windows:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 4: Install packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 5: Generate data and train the model
```bash
python train_model.py
```
This creates a synthetic 10,000-row Excel dataset, compares four regression models, and saves the winning model and evaluation metrics.

## Step 6: Run the web application
```bash
streamlit run app.py
```
Open the local address printed in the terminal (typically `http://localhost:8501`).

## Step 7: Explore the app
1. Home: overview and data/model status.
2. Cost Prediction: enter project specifications and predict costs.
3. Analytics Dashboard: explore portfolio charts.
4. Model Performance: compare model metrics.
5. Report Generation: download a PDF of the latest estimate.
6. About Project: review methodology, limitations, and future scope.

## Troubleshooting
- If `python` is not found on Windows, install Python and enable **Add Python to PATH**, then restart VS Code.
- If the model is missing, rerun `python train_model.py`.
- If a package import fails, confirm the virtual environment is activated and run `pip install -r requirements.txt` again.
- If the app was already open after generating data/model files, click **Refresh application** in the sidebar or restart Streamlit.
