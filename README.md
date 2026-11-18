<<<<<<< HEAD
# AutoInsight ML — Modular Edition

This project reorganizes the supplied 1,613-line Streamlit application into focused Python modules.
The existing sidebar navigation and feature implementations are retained.

## Project structure

```text
AutoInsight-ML-Modular/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── common.py
│   ├── data.py
│   ├── visualization.py
│   ├── missing_values.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── tuning.py
│   ├── artifacts.py
│   └── prediction.py
├── utils/
│   ├── __init__.py
│   └── display.py
└── assets/
```

## Run in VS Code (Windows PowerShell)

Open this folder in VS Code, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Using the virtual environment's Python executable directly avoids the PowerShell
script execution-policy issue with `Activate.ps1`.

## Notes

- Keep the folder structure intact; do not move individual module files.
- Run Streamlit from the project root (the folder containing `app.py`).
- The original application used a sidebar radio for navigation. That interaction is
  intentionally retained to minimize behavioral changes during this refactor.
- `modules/common.py` centralizes the original dependency imports.
- `utils/display.py` is available for future consolidation of repeated UI helpers;
  the original functions are preserved in their source modules for compatibility.

## Suggested validation

1. Upload a CSV and check Dashboard and all six EDA tabs.
2. Apply missing-value strategies.
3. Run classification and regression training separately.
4. Compare original features with SelectKBest and PCA.
5. Inspect evaluation plots and metrics.
6. Run hyperparameter tuning.
7. Download a model artifact.
8. Test manual and CSV batch prediction.
=======
# AutoInsight ML

An interactive Streamlit app for CSV data profiling, exploratory data analysis, and baseline model comparison for classification or regression. It handles missing numeric and categorical values inside model pipelines and compares models on a held-out split.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Start the app: `streamlit run app.py`.

Use datasets you are allowed to process. The app does not save the uploaded CSV. The model scores are a quick baseline; tune and validate models before using them for decisions.
>>>>>>> b2b61b8e00fdbee0f09f0d3236eafbceb22de07b
