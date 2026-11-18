"""AutoInsight ML - modular Streamlit application.

Run with: python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd

from modules.data import clean_dataset, data_profile
from modules.visualization import explore_data
from modules.missing_values import missing_value_section
from modules.modeling import train_models
from modules.evaluation import evaluation_section
from modules.feature_engineering import feature_experiment_section
from modules.tuning import tuning_section
from modules.artifacts import save_model_section
from modules.prediction import prediction_section

st.set_page_config(
    page_title="AutoInsight ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg,#4776E6,#8E54E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #8792A2;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120,130,150,.08);
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.15);
    }
    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 AutoInsight ML</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explore data • Engineer features • Train models • Generate predictions</div>',
    unsafe_allow_html=True
)


DEFAULTS = {
    "df": None,
    "source_name": None,
    "trained_model": None,
    "model_info": None,
    "comparison": None,
    "prediction_results": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value




def main():
    with st.sidebar:
        st.title("⚙️ AutoInsight ML")
        st.caption("Data Science Workspace")
        st.divider()

        uploaded = st.file_uploader(
            "Upload Dataset",
            type=["csv"],
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Exploratory Data Analysis",
                "Missing Values",
                "Feature Engineering",
                "Model Training",
                "Model Evaluation",
                "Hyperparameter Tuning",
                "Predictions",
                "Save Model",
            ],
        )

        st.divider()
        st.caption("AutoInsight ML • Machine Learning Platform")

    if uploaded is not None:
        try:
            new_name = uploaded.name

            if st.session_state["source_name"] != new_name:
                source = pd.read_csv(uploaded)
                df, duplicates = clean_dataset(source)

                st.session_state["df"] = df
                st.session_state["source_name"] = new_name
                st.session_state["duplicates"] = duplicates

                # Clear models belonging to the previous dataset
                st.session_state["trained_model"] = None
                st.session_state["model_info"] = None
                st.session_state["comparison"] = None
                st.session_state["evaluation"] = None

        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
            return

    df = st.session_state["df"]

    if df is None:
        st.info("Upload a CSV dataset from the sidebar to begin.")
        st.markdown("""
        ### Welcome to AutoInsight ML

        Explore your dataset, analyze data quality, experiment with
        feature engineering, train machine learning models and
        generate predictions.

        **Your workflow:**
        - Dataset exploration
        - Intelligent preprocessing
        - Feature engineering
        - Model comparison
        - Model evaluation
        - Hyperparameter optimization
        - Predictions
        """)
        return

    if df.empty or df.shape[1] == 0:
        st.error("The uploaded dataset is empty.")
        return

    if page == "Dashboard":
        data_profile(df, st.session_state["duplicates"])

    elif page == "Exploratory Data Analysis":
        explore_data(df)

    elif page == "Missing Values":
        missing_value_section(df)

    elif page == "Feature Engineering":
        feature_experiment_section(
            st.session_state.get("imputed_df", df)
        )

    elif page == "Model Training":
        st.header("Machine Learning Studio")

        task = st.radio(
            "Choose task",
            ["Classification", "Regression"],
            horizontal=True,
        )

        target = st.selectbox(
            "Select target column",
            df.columns,
        )

        st.subheader("Training Configuration")

        c1, c2 = st.columns(2)
        test_size = c1.slider(
            "Test size",
            0.1, 0.4, 0.2, 0.05,
        )
        seed = c2.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
        )

        imputation = st.selectbox(
            "Numerical/Categorical imputation",
            ["Median", "Mean", "Most frequent", "Constant"],
        )

        feature_method = st.selectbox(
            "Feature engineering",
            ["None", "SelectKBest", "PCA"],
        )

        max_features = len(df.drop(columns=[target]).columns)

        feature_count = st.slider(
            "Number of selected features/components",
            1,
            max(1, max_features),
            min(3, max(1, max_features)),
            disabled=(feature_method == "None"),
        )

        if st.button("Train and Compare Models", type="primary"):
            train_models(
                df, target, task, test_size, int(seed),
                imputation, feature_method, feature_count,
            )

    elif page == "Model Evaluation":
        info = st.session_state.get("model_info")
        if info:
            evaluation_section(info["task"])
        else:
            st.info("Train a model first.")

    elif page == "Hyperparameter Tuning":
        tuning_section()

    elif page == "Predictions":
        prediction_section()

    elif page == "Save Model":
        save_model_section()



if __name__ == "__main__":
    main()
