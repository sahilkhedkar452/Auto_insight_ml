"""Reusable display helpers."""
import streamlit as st


def render_df(df, height=350):
    st.dataframe(df, use_container_width=True, height=height)


def download_csv(df, filename, label="Download CSV"):
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )
