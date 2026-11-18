"""AutoInsight ML: data."""
from .common import *

def detect_columns(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def data_profile(df, duplicates):
    st.header("Dataset Overview")

    numeric, categorical = detect_columns(df)

    a, b, c, d, e = st.columns(5)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Missing cells", f"{df.isna().sum().sum():,}")
    d.metric("Duplicates removed", duplicates)
    e.metric("Numeric features", len(numeric))

    st.subheader("Dataset Preview")
    render_df(df.head(100))

    st.subheader("Column Quality Report")

    report = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2),
        "Unique Values": df.nunique(),
        "Recommended Imputation": [
            get_recommendation(df[c]) for c in df.columns
        ],
    })

    render_df(report.reset_index(names="Column"))

    st.subheader("Dataset Statistics")

    if numeric:
        render_df(df[numeric].describe().T.round(3))

    download_csv(report.reset_index(names="Column"),
                 "data-quality-report.csv",
                 "Download Quality Report")


def get_recommendation(series):
    """Recommend an imputation strategy from column characteristics."""
    missing_pct = series.isna().mean() * 100

    if missing_pct == 0:
        return "No imputation required"

    if pd.api.types.is_numeric_dtype(series):
        skew = series.skew()
        if abs(skew) > 1:
            return "Median"
        return "Mean"

    return "Most frequent"


def clean_dataset(df: pd.DataFrame):
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()

    # Remove entirely empty columns
    cleaned = cleaned.dropna(axis=1, how="all")

    # Trim text columns
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        cleaned[col] = cleaned[col].replace("", np.nan)

    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned, duplicates


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def detect_columns(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical
