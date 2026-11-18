"""AutoInsight ML: missing values."""
from .common import *
from modules.data import detect_columns


def missing_value_section(df):
    st.header("Intelligent Missing Value Handling")

    numeric, categorical = detect_columns(df)
    missing_cols = [c for c in df.columns if df[c].isna().any()]

    if not missing_cols:
        st.success("Your dataset has no missing values.")
        return df

    st.info(
        "Recommendations are based on column type, skewness, "
        "missing percentage and available observations."
    )

    result = df.copy()

    for col in missing_cols:
        series = df[col]
        recommendation = get_recommendation(series)
        pct = series.isna().mean() * 100

        st.markdown(f"### {col}")
        st.caption(
            f"{int(series.isna().sum())} missing values ({pct:.2f}%)"
        )
        st.write(f"**Recommended method:** {recommendation}")

        options = (
            ["Recommended", "Mean", "Median", "Most frequent",
             "Constant", "Drop rows"]
            if col in numeric
            else ["Recommended", "Most frequent", "Constant", "Drop rows"]
        )

        selected = st.selectbox(
            f"Handling method for {col}",
            options,
            key=f"impute_{col}",
        )

        if selected == "Recommended":
            selected = recommendation

        if selected == "Drop rows":
            continue

        if selected == "Mean":
            if pd.api.types.is_numeric_dtype(series):
                fill = series.mean()
            else:
                fill = series.mode().iloc[0]

        elif selected == "Median":
            fill = series.median()

        elif selected == "Most frequent":
            modes = series.mode()
            fill = modes.iloc[0] if len(modes) else "Unknown"

        else:
            fill = st.text_input(
                f"Constant value for {col}",
                value="Unknown",
                key=f"constant_{col}",
            )
            if col in numeric:
                try:
                    fill = float(fill)
                except ValueError:
                    st.error("Enter a numeric constant for this column.")
                    continue

        result[col] = result[col].fillna(fill)

    if st.button("Apply Missing Value Handling", type="primary"):
        # Drop rows for columns where the user selected that strategy
        for col in missing_cols:
            if st.session_state.get(f"impute_{col}") == "Drop rows":
                result = result.dropna(subset=[col])

        st.session_state["imputed_df"] = result.reset_index(drop=True)
        st.success("Missing-value handling applied.")
        st.metric(
            "Remaining missing cells",
            int(result.isna().sum().sum()),
        )
        render_df(result.head(50))
        download_csv(result, "imputed-dataset.csv")

    return st.session_state.get("imputed_df", df)
