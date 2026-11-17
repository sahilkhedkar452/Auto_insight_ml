# AutoInsight ML

An interactive Streamlit app for CSV data profiling, exploratory data analysis, and baseline model comparison for classification or regression. It handles missing numeric and categorical values inside model pipelines and compares models on a held-out split.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. Start the app: `streamlit run app.py`.

Use datasets you are allowed to process. The app does not save the uploaded CSV. The model scores are a quick baseline; tune and validate models before using them for decisions.
