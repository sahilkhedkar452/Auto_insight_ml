"""AutoInsight ML: visualization."""
from .common import *
from modules.data import detect_columns

def detect_columns(df):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric, categorical


def explore_data(df):
    st.header("Exploratory Data Analysis")

    numeric, categorical = detect_columns(df)

    if not numeric and not categorical:
        st.warning("No analyzable columns found.")
        return

    eda_tabs = st.tabs([
        "Distributions",
        "Relationships",
        "Outliers",
        "Correlations",
        "Missing Values",
        "Categorical Analysis",
    ])

    # --------------------------------------------------------
    # DISTRIBUTIONS
    # --------------------------------------------------------

    with eda_tabs[0]:
        st.subheader("Numerical Distribution Analysis")

        if numeric:
            col = st.selectbox("Choose numerical column", numeric,
                               key="dist_col")

            chart_type = st.selectbox(
                "Chart type",
                ["Histogram", "Box Plot", "Violin Plot", "Density Histogram"],
                key="dist_type"
            )

            if chart_type == "Histogram":
                fig = px.histogram(
                    df, x=col, marginal="box",
                    color_discrete_sequence=["#636EFA"],
                    title=f"Distribution of {col}",
                )
            elif chart_type == "Box Plot":
                fig = px.box(
                    df, y=col,
                    color_discrete_sequence=["#00CC96"],
                )
            elif chart_type == "Violin Plot":
                fig = px.violin(
                    df, y=col, box=True, points="outliers",
                    color_discrete_sequence=["#AB63FA"],
                )
            else:
                fig = px.histogram(
                    df, x=col, histnorm="probability density",
                    marginal="rug",
                    color_discrete_sequence=["#EF553B"],
                )

            st.plotly_chart(fig, use_container_width=True)

            series = safe_numeric(df[col]).dropna()

            if len(series):
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("Mean", f"{series.mean():.3f}")
                x2.metric("Median", f"{series.median():.3f}")
                x3.metric("Skewness", f"{series.skew():.3f}")
                x4.metric("Kurtosis", f"{series.kurtosis():.3f}")
        else:
            st.info("No numerical columns available.")

    # --------------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------------

    with eda_tabs[1]:
        st.subheader("Feature Relationships")

        if len(numeric) >= 2:
            x = st.selectbox("X-axis", numeric, key="scatter_x")
            y = st.selectbox(
                "Y-axis",
                [c for c in numeric if c != x],
                key="scatter_y",
            )

            color_options = ["None"] + categorical
            color = st.selectbox("Color grouping", color_options)

            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=None if color == "None" else color,
                hover_data=df.columns.tolist()[:8],
                trendline="ols" if st.checkbox(
                    "Show linear trendline", key="trend"
                ) else None,
                title=f"{x} vs {y}",
            )
            st.plotly_chart(fig, use_container_width=True)

            corr = df[[x, y]].corr().iloc[0, 1]
            st.metric("Pearson correlation", f"{corr:.4f}")
        else:
            st.info("At least two numerical columns are required.")

        if len(numeric) >= 2:
            st.subheader("Pairwise Feature Analysis")
            selected = st.multiselect(
                "Select features for pair plot",
                numeric,
                default=numeric[:min(4, len(numeric))],
                key="pair_cols",
            )
            if 2 <= len(selected) <= 6:
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected,
                    title="Pairwise Relationships",
                )
                fig.update_traces(diagonal_visible=False)
                st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    with eda_tabs[2]:
        st.subheader("Outlier Detection")

        if numeric:
            col = st.selectbox("Select feature", numeric, key="outlier_col")
            method = st.radio(
                "Detection method",
                ["IQR", "Z-score"],
                horizontal=True,
            )

            series = safe_numeric(df[col]).dropna()

            if method == "IQR":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (series < lower) | (series > upper)
            else:
                mean = series.mean()
                std = series.std()
                threshold = st.slider(
                    "Absolute Z-score threshold",
                    1.5, 5.0, 3.0, 0.5
                )
                if std == 0 or pd.isna(std):
                    mask = pd.Series(False, index=series.index)
                else:
                    mask = ((series - mean) / std).abs() > threshold

            outliers = series[mask]

            a, b, c = st.columns(3)
            a.metric("Outliers", len(outliers))
            b.metric("Normal values", len(series) - len(outliers))
            c.metric("Outlier percentage",
                     f"{len(outliers) / max(len(series), 1) * 100:.2f}%")

            fig = px.box(
                df, y=col, points="outliers",
                color_discrete_sequence=["#AB63FA"],
                title=f"Outlier distribution: {col}",
            )
            st.plotly_chart(fig, use_container_width=True)

            if len(outliers):
                st.write("Detected outlier records")
                render_df(df.loc[outliers.index].copy())

            st.caption(
                "Detection is diagnostic. Outliers are not automatically deleted."
            )
        else:
            st.info("No numerical columns available.")

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    with eda_tabs[3]:
        st.subheader("Correlation Analysis")

        if len(numeric) >= 2:
            method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman", "kendall"],
            )
            corr = df[numeric].corr(method=method)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title=f"{method.title()} Correlation Heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)

            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            if not pairs.empty:
                pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
                pairs["Absolute Correlation"] = pairs["Correlation"].abs()
                pairs = pairs.sort_values(
                    "Absolute Correlation", ascending=False
                )
                st.subheader("Strongest Feature Relationships")
                render_df(pairs.head(15))
        else:
            st.info("At least two numerical columns are needed.")

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    with eda_tabs[4]:
        st.subheader("Missing Value Analysis")

        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing):
            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing Percentage":
                    (missing.values / len(df) * 100).round(2),
            })

            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Percentage",
                color="Missing Percentage",
                color_continuous_scale="Turbo",
                title="Missing Values by Column",
                text="Missing Percentage",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Missingness matrix
            matrix = df.isna().astype(int)
            fig = px.imshow(
                matrix.T.iloc[:, :min(500, len(df))],
                color_continuous_scale=["#E8F0FE", "#EF553B"],
                aspect="auto",
                title="Missingness Matrix (first 500 rows)",
                labels={"x": "Row", "y": "Feature", "color": "Missing"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found.")

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    with eda_tabs[5]:
        st.subheader("Categorical Feature Analysis")

        if categorical:
            col = st.selectbox("Select categorical column",
                               categorical, key="cat_col")

            counts = (
                df[col].astype("string")
                .fillna("(Missing)")
                .value_counts()
                .head(30)
                .rename_axis("Category")
                .reset_index(name="Count")
            )

            chart = st.radio(
                "Representation",
                ["Bar Chart", "Pie Chart", "Treemap"],
                horizontal=True,
            )

            if chart == "Bar Chart":
                fig = px.bar(
                    counts, x="Category", y="Count",
                    color="Count", color_continuous_scale="Turbo",
                )
            elif chart == "Pie Chart":
                fig = px.pie(
                    counts, names="Category", values="Count",
                    hole=0.35,
                )
            else:
                fig = px.treemap(
                    counts, path=["Category"], values="Count",
                    color="Count", color_continuous_scale="Viridis",
                )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")
