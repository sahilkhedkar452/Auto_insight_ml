"""AutoInsight ML: prediction."""
from .common import *


def prediction_section():
    st.header("Prediction Centre")

    pipeline = st.session_state["trained_model"]
    info = st.session_state["model_info"]

    if pipeline is None or info is None:
        st.info("Train a model before generating predictions.")
        return

    st.success(
        f"Active model: {info['model']} | Task: {info['task']}"
    )

    features = info["features"]

    mode = st.radio(
        "Prediction method",
        ["Manual Multiple Entries", "Upload CSV Dataset"],
        horizontal=True,
    )

    if mode == "Manual Multiple Entries":
        st.subheader("Enter multiple records")

        count = st.number_input(
            "Number of records",
            min_value=1,
            max_value=100,
            value=1,
        )

        input_rows = []

        with st.form("manual_prediction_form"):
            for i in range(int(count)):
                st.markdown(f"**Record {i + 1}**")
                cols = st.columns(3)
                row = {}

                for j, feature in enumerate(features):
                    col = cols[j % 3]

                    row[feature] = col.text_input(
                        feature,
                        key=f"manual_{i}_{feature}",
                    )

                input_rows.append(row)

            submitted = st.form_submit_button(
                "Generate Predictions",
                type="primary",
            )

        if submitted:
            input_df = pd.DataFrame(input_rows)

            for col in input_df.columns:
                original_dtype = (
                    st.session_state.get("df")[col].dtype
                    if col in st.session_state["df"].columns
                    else object
                )
                if pd.api.types.is_numeric_dtype(original_dtype):
                    input_df[col] = pd.to_numeric(
                        input_df[col], errors="coerce"
                    )

            try:
                predictions = pipeline.predict(input_df)

                result = input_df.copy()

                if info["task"] == "Classification":
                    encoder = info.get("label_encoder")
                    if encoder is not None:
                        predictions = encoder.inverse_transform(
                            predictions.astype(int)
                        )

                result["Prediction"] = predictions

                st.subheader("Prediction Results")
                render_df(result)
                st.session_state["prediction_results"] = result
                download_csv(result, "manual-predictions.csv")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:
        uploaded = st.file_uploader(
            "Upload CSV for prediction",
            type=["csv"],
            key="prediction_csv",
        )

        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)

                missing = [
                    col for col in features
                    if col not in input_df.columns
                ]

                if missing:
                    st.error(f"Missing required features: {missing}")
                    return

                input_df = input_df[features].copy()

                predictions = pipeline.predict(input_df)

                if info["task"] == "Classification":
                    encoder = info.get("label_encoder")
                    if encoder is not None:
                        predictions = encoder.inverse_transform(
                            predictions.astype(int)
                        )

                result = input_df.copy()
                result["Prediction"] = predictions

                if hasattr(pipeline, "predict_proba"):
                    try:
                        probabilities = pipeline.predict_proba(input_df)
                        if probabilities.shape[1] == 2:
                            result["Prediction Confidence"] = (
                                probabilities.max(axis=1)
                            )
                    except Exception:
                        pass

                st.subheader("Batch Prediction Results")
                st.metric("Records predicted", len(result))
                render_df(result.head(100))

                st.session_state["prediction_results"] = result
                download_csv(result, "batch-predictions.csv")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
