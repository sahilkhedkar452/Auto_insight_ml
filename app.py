"""AutoInsight ML: quick dataset profiling, visual exploration, and model comparison."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model: Any, X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical))
    return Pipeline([("prepare", ColumnTransformer(transformers, remainder="drop")), ("model", model)])


def profile_data(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    with st.expander("Column types and missing values", expanded=True):
        summary = pd.DataFrame({"type": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique(dropna=True)})
        st.dataframe(summary, use_container_width=True)
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)


def explore_data(df: pd.DataFrame) -> None:
    st.subheader("Explore distributions")
    column = st.selectbox("Column to visualize", list(df.columns))
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        st.plotly_chart(px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}"), use_container_width=True)
    else:
        counts = series.astype("string").fillna("(missing)").value_counts().head(30).rename_axis(column).reset_index(name="count")
        st.plotly_chart(px.bar(counts, x=column, y="count", title=f"Top values in {column}"), use_container_width=True)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        st.plotly_chart(px.imshow(numeric.corr(), text_auto=".2f", aspect="auto", title="Numeric feature correlations"), use_container_width=True)


def compare_models(df: pd.DataFrame, target: str, task: str, test_fraction: float, seed: int) -> None:
    data = df.dropna(subset=[target]).copy()
    X, y = data.drop(columns=[target]), data[target]
    if X.empty:
        st.error("Choose a target that leaves at least one feature column.")
        return
    if task == "Classification" and y.nunique() < 2:
        st.error("Classification needs at least two target classes.")
        return
    if len(data) < 10:
        st.error("Upload at least 10 complete target rows for a useful train/test split.")
        return
    stratify = y if task == "Classification" and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=stratify)
    except ValueError as exc:
        st.error(f"Could not create a train/test split: {exc}")
        return
    if task == "Classification":
        models = {"Logistic Regression": LogisticRegression(max_iter=1200), "Random Forest": RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced"), "Extra Trees": ExtraTreesClassifier(n_estimators=250, random_state=seed, class_weight="balanced")}
    else:
        models = {"Ridge Regression": Ridge(), "Random Forest": RandomForestRegressor(n_estimators=250, random_state=seed), "Extra Trees": ExtraTreesRegressor(n_estimators=250, random_state=seed), "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=seed)}
    rows = []
    fitted = {}
    progress = st.progress(0)
    for index, (name, estimator) in enumerate(models.items(), start=1):
        try:
            pipeline = build_pipeline(estimator, X_train)
            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            duration = time.perf_counter() - started
            predictions = pipeline.predict(X_test)
            if task == "Classification":
                rows.append({"model": name, "accuracy": accuracy_score(y_test, predictions), "F1 weighted": f1_score(y_test, predictions, average="weighted", zero_division=0), "fit seconds": duration})
            else:
                rows.append({"model": name, "R²": r2_score(y_test, predictions), "MAE": mean_absolute_error(y_test, predictions), "fit seconds": duration})
            fitted[name] = pipeline
        except Exception as exc:
            st.warning(f"{name} could not be trained: {exc}")
        progress.progress(index / len(models))
    if not rows:
        st.error("No model completed successfully. Review the data and target type.")
        return
    scores = pd.DataFrame(rows)
    sort_column = "F1 weighted" if task == "Classification" else "R²"
    scores = scores.sort_values(sort_column, ascending=False, na_position="last")
    st.subheader("Model comparison")
    st.caption("Metrics use a held-out test split; results are a baseline, not a guarantee of future performance.")
    st.dataframe(scores, hide_index=True, use_container_width=True)
    winner = scores.iloc[0]
    st.success(f"Top model on this split: **{winner['model']}** ({sort_column}: {winner[sort_column]:.3f}).")
    st.download_button("Download comparison CSV", scores.to_csv(index=False).encode("utf-8"), "model-comparison.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="AutoInsight ML", page_icon="📊", layout="wide")
    st.title("📊 AutoInsight ML")
    st.caption("Profile a dataset, explore it visually, and compare baseline models.")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if not uploaded:
        st.info("Start with a CSV file. Avoid uploading sensitive personal data.")
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return
    if df.empty or len(df.columns) == 0:
        st.error("This CSV has no data columns.")
        return
    profile_tab, explore_tab, model_tab = st.tabs(["Data profile", "EDA", "Model comparison"])
    with profile_tab:
        profile_data(df)
    with explore_tab:
        explore_data(df)
    with model_tab:
        st.subheader("Choose a prediction task")
        task = st.radio("Task", ["Classification", "Regression"], horizontal=True)
        target = st.selectbox("Target column", list(df.columns))
        c1, c2 = st.columns(2)
        test_fraction = c1.slider("Test data", 0.1, 0.4, 0.2, 0.05)
        seed = c2.number_input("Random seed", min_value=0, max_value=99999, value=42)
        if st.button("Compare models", type="primary"):
            compare_models(df, target, task, test_fraction, int(seed))


if __name__ == "__main__":
    main()
