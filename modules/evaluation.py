"""AutoInsight ML: evaluation."""
from .common import *


def evaluation_section(task):
    st.header("Detailed Model Evaluation")

    evaluation = st.session_state.get("evaluation")
    info = st.session_state.get("model_info")

    if evaluation is None or info is None:
        st.info("Train models first.")
        return

    pipeline = st.session_state["trained_model"]
    y_test = evaluation["y_test"]
    predictions = evaluation["predictions"]
    X_test = evaluation["X_test"]

    if task == "Classification":
        st.subheader("Classification Report")

        report = classification_report(
            y_test, predictions,
            output_dict=True, zero_division=0,
        )
        render_df(pd.DataFrame(report).T.round(4))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            labels={"x": "Predicted", "y": "Actual"},
        )
        st.plotly_chart(fig, use_container_width=True)

        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(X_test)

                if probabilities.shape[1] == 2:
                    from sklearn.metrics import RocCurveDisplay
                    fig, ax = st.columns(1)[0].empty(), None
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    RocCurveDisplay.from_predictions(
                        y_test, probabilities[:, 1], ax=ax
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    PrecisionRecallDisplay.from_predictions(
                        y_test, probabilities[:, 1], ax=ax
                    )
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as exc:
                st.caption(f"Probability plots unavailable: {exc}")

    else:
        residuals = np.asarray(y_test) - np.asarray(predictions)

        comparison = pd.DataFrame({
            "Actual": np.asarray(y_test),
            "Predicted": np.asarray(predictions),
            "Residual": residuals,
        })

        fig = px.scatter(
            comparison,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted",
        )
        minimum = min(comparison["Actual"].min(),
                      comparison["Predicted"].min())
        maximum = max(comparison["Actual"].max(),
                      comparison["Predicted"].max())
        fig.add_shape(
            type="line",
            x0=minimum, y0=minimum,
            x1=maximum, y1=maximum,
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(
            comparison, x="Residual",
            marginal="box",
            title="Residual Distribution",
            color_discrete_sequence=["#AB63FA"],
        )
        st.plotly_chart(fig, use_container_width=True)

        render_df(comparison.head(100))
