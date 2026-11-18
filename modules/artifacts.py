"""AutoInsight ML: artifacts."""
from .common import *


def save_model_section():
    st.header("Save Trained Model")

    pipeline = st.session_state["trained_model"]
    info = st.session_state["model_info"]

    if pipeline is None or info is None:
        st.info("Train a model before saving.")
        return

    artifact = {
        "pipeline": pipeline,
        "info": info,
    }

    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    buffer.seek(0)

    st.download_button(
        "Download Trained Model",
        data=buffer,
        file_name="autoinsight-model.joblib",
        mime="application/octet-stream",
        type="primary",
    )

    st.warning(
        "Only load model files that you created or trust. "
        "Joblib/pickle files can execute code when loaded."
    )
