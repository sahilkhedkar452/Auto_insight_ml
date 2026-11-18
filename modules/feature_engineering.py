"""AutoInsight ML: feature engineering."""
from .common import *


def feature_experiment_section(df):
    st.header("Feature Engineering Experiments")

    st.write(
        "Compare original features with a selected or extracted "
        "feature representation."
    )

    target = st.selectbox(
        "Target column",
        df.columns.tolist(),
        key="feature_target",
    )

    task = st.radio(
        "Task",
        ["Classification", "Regression"],
        horizontal=True,
        key="feature_task",
    )

    method = st.selectbox(
        "Feature engineering method",
        ["SelectKBest", "PCA"],
    )

    feature_count = st.slider(
        "Number of features/components",
        1,
        max(1, len(df.drop(columns=[target]).columns)),
        min(3, max(1, len(df.drop(columns=[target]).columns))),
    )

    imputation = st.selectbox(
        "Imputation for model pipeline",
        ["Median", "Mean", "Most frequent", "Constant"],
        key="feature_imputation",
    )

    if st.button("Run feature engineering experiment",
                 type="primary"):
        # Original baseline
        st.subheader("Experiment A: Original features")
        train_models(
            df, target, task, 0.2, 42,
            imputation, "None", feature_count,
        )
        original = st.session_state.get("comparison")

        st.subheader("Experiment B: Engineered features")
        train_models(
            df, target, task, 0.2, 42,
            imputation, method, feature_count,
        )
        engineered = st.session_state.get("comparison")

        if original is not None and engineered is not None:
            st.info(
                "The second experiment is now the active trained model. "
                "Both comparisons use the same random split settings."
            )
