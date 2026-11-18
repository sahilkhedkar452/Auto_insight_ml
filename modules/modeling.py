"""AutoInsight ML: modeling."""
from .common import *


def build_pipeline(
    model: Any,
    X: pd.DataFrame,
    imputation: str,
    feature_method: str,
    feature_count: int,
    task: str,
    scale_numeric: bool = True,
):
    numeric, categorical = detect_columns(X)
    transformers = []

    numeric_strategy = (
        "mean" if imputation == "Mean"
        else "most_frequent" if imputation == "Most frequent"
        else "constant" if imputation == "Constant"
        else "median"
    )

    if numeric:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy=numeric_strategy))
        ]

        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric",
            Pipeline(numeric_steps),
            numeric,
        ))

    if categorical:
        cat_strategy = (
            "constant" if imputation == "Constant" else "most_frequent"
        )
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy=cat_strategy,
                    fill_value="Unknown",
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=100,
                )),
            ]),
            categorical,
        ))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
    )

    steps = [("preprocessor", preprocessor)]

    if feature_method == "SelectKBest":
        score_func = f_classif if task == "Classification" else f_regression
        steps.append((
            "feature_selection",
            SelectKBest(
                score_func=score_func,
                k=feature_count,
            ),
        ))

    elif feature_method == "PCA":
        steps.append((
            "feature_extraction",
            PCA(n_components=feature_count),
        ))

    steps.append(("model", model))

    return Pipeline(steps)


def get_models(task, seed):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1500, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=seed,
                class_weight="balanced", n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, random_state=seed,
                class_weight="balanced", n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=seed, class_weight="balanced"
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(probability=True, class_weight="balanced"),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=seed
            ),
            "AdaBoost": AdaBoostClassifier(random_state=seed),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=2000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=seed
        ),
        "AdaBoost": AdaBoostRegressor(random_state=seed),
    }


def train_models(
    df, target, task, test_size, seed,
    imputation, feature_method, feature_count,
):
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target]

    if X.empty:
        st.error("At least one feature column is required.")
        return

    if len(data) < 10:
        st.error("At least 10 target-complete records are recommended.")
        return

    if task == "Classification":
        if y.nunique() < 2:
            st.error("Classification requires at least two classes.")
            return

        if y.nunique() > 50:
            st.error(
                "This target has more than 50 classes. "
                "Check whether it is actually a regression target."
            )
            return

    # Encode classification target labels for consistency
    label_encoder = None
    if task == "Classification":
        label_encoder = LabelEncoder()
        y = pd.Series(
            label_encoder.fit_transform(y.astype(str)),
            index=y.index,
            name=target,
        )

    stratify = None
    if task == "Classification":
        counts = y.value_counts()
        if counts.min() >= 2 and int(len(data) * test_size) >= y.nunique():
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    if feature_method == "SelectKBest":
        max_features = min(
            feature_count,
            max(1, X.shape[1]),
        )
        feature_count = max_features

    elif feature_method == "PCA":
        # PCA operates after one-hot encoding, so this is a safe
        # upper bound based on row and original feature counts.
        feature_count = max(
            1, min(feature_count, X.shape[1], len(X_train) - 1)
        )

    models = get_models(task, seed)
    results = []
    fitted = {}

    progress = st.progress(0)
    status = st.empty()

    for index, (name, model) in enumerate(models.items(), start=1):
        status.write(f"Training {name}...")

        try:
            pipeline = build_pipeline(
                model, X_train, imputation,
                feature_method, feature_count, task,
                scale_numeric=True,
            )

            started = time.perf_counter()
            pipeline.fit(X_train, y_train)
            duration = time.perf_counter() - started

            predictions = pipeline.predict(X_test)

            row = {
                "Model": name,
                "Training Time (s)": round(duration, 4),
            }

            if task == "Classification":
                row.update({
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision (weighted)": precision_score(
                        y_test, predictions,
                        average="weighted", zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        y_test, predictions,
                        average="weighted", zero_division=0
                    ),
                    "F1 (weighted)": f1_score(
                        y_test, predictions,
                        average="weighted", zero_division=0
                    ),
                })

                if hasattr(pipeline, "predict_proba"):
                    try:
                        probabilities = pipeline.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2:
                            row["ROC AUC"] = roc_auc_score(
                                y_test, probabilities[:, 1]
                            )
                        else:
                            row["ROC AUC"] = roc_auc_score(
                                y_test, probabilities,
                                multi_class="ovr",
                                average="weighted",
                            )
                    except Exception:
                        row["ROC AUC"] = np.nan

            else:
                mse = mean_squared_error(y_test, predictions)
                row.update({
                    "R²": r2_score(y_test, predictions),
                    "MAE": mean_absolute_error(y_test, predictions),
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "MAPE": (
                        mean_absolute_percentage_error(
                            y_test, predictions
                        ) if (np.asarray(y_test) != 0).all() else np.nan
                    ),
                })

            results.append(row)

            fitted[name] = {
                "pipeline": pipeline,
                "predictions": predictions,
                "y_test": y_test,
                "X_test": X_test,
            }

        except Exception as exc:
            st.warning(f"{name} failed: {exc}")

        progress.progress(index / len(models))

    status.empty()
    progress.empty()

    if not results:
        st.error("No model could be trained.")
        return

    results_df = pd.DataFrame(results)

    score_col = "F1 (weighted)" if task == "Classification" else "R²"
    results_df = results_df.sort_values(
        score_col, ascending=False, na_position="last"
    ).reset_index(drop=True)

    winner = results_df.iloc[0]["Model"]
    best = fitted[winner]

    st.session_state["comparison"] = results_df
    st.session_state["trained_model"] = best["pipeline"]
    st.session_state["model_info"] = {
        "target": target,
        "task": task,
        "model": winner,
        "features": list(X.columns),
        "label_encoder": label_encoder,
        "imputation": imputation,
        "feature_method": feature_method,
        "feature_count": feature_count,
    }

    st.session_state["evaluation"] = best
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    st.success(f"Best model on the held-out split: {winner}")

    st.subheader("Model Performance Comparison")
    render_df(results_df)
    download_csv(results_df, "model-comparison.csv")

    fig = px.bar(
        results_df,
        x="Model",
        y=score_col,
        color=score_col,
        color_continuous_scale="Viridis",
        title=f"Model comparison by {score_col}",
        text_auto=".3f",
    )
    st.plotly_chart(fig, use_container_width=True)
