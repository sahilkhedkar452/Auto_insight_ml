"""Shared third-party imports used by AutoInsight ML modules."""
from __future__ import annotations
import io
import time
import warnings
from typing import Any
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso, LinearRegression, LogisticRegression, Ridge
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings('ignore')



# Shared UI helpers retained from the original application.
def render_df(df):
    st.dataframe(df, use_container_width=True, height=350)


def download_csv(df, filename, label="Download CSV"):
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )
