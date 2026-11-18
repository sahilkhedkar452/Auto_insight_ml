"""AutoInsight ML: tuning."""
from .common import *


def tuning_section():
    st.header("Hyperparameter Tuning")

    if st.session_state["trained_model"] is None:
        st.info("Train a model before tuning.")
        return

    info = st.session_state["model_info"]

    model_name = info["model"]
    task = info["task"]

    st.write(f"Current model: **{model_name}**")

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    if model_name in ["Random Forest", "Extra Trees"]:
        parameter_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    elif model_name == "Logistic Regression":
        parameter_grid = {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs"],
        }
    elif model_name == "Ridge Regression":
        parameter_grid = {
            "model__alpha": [0.01, 0.1, 1, 10, 100],
        }
    elif model_name == "KNN":
        parameter_grid = {
            "model__n_neighbors": [3, 5, 7, 9],
        }
    elif model_name == "Decision Tree":
        parameter_grid = {
            "model__max_depth": [None, 3, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    else:
        st.info(
            "A custom search grid for this model is not configured yet."
        )
        return

    search_type = st.radio(
        "Search method",
        ["Grid Search", "Randomized Search"],
        horizontal=True,
    )

    if st.button("Start tuning", type="primary"):
        pipeline = st.session_state["trained_model"]

        try:
            if search_type == "Grid Search":
                search = GridSearchCV(
                    pipeline,
                    parameter_grid,
                    cv=3,
                    scoring=(
                        "f1_weighted" if task == "Classification"
                        else "r2"
                    ),
                    n_jobs=-1,
                )
            else:
                search = RandomizedSearchCV(
                    pipeline,
                    parameter_grid,
                    n_iter=8,
                    cv=3,
                    scoring=(
                        "f1_weighted" if task == "Classification"
                        else "r2"
                    ),
                    n_jobs=-1,
                    random_state=42,
                )

            with st.spinner("Searching for better parameters..."):
                search.fit(X_train, y_train)

            st.success("Hyperparameter tuning completed.")
            st.json(search.best_params_)
            st.metric("Best cross-validation score",
                      round(search.best_score_, 4))

            st.session_state["trained_model"] = search.best_estimator_
            st.session_state["tuned_model"] = search.best_estimator_

        except Exception as exc:
            st.error(f"Tuning failed: {exc}")
