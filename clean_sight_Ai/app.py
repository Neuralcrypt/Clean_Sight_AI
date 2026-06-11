import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from preprocessing import apply_preprocessing_pipeline, profile_dataframe
from utils import compute_health_score, ensure_dirs, get_duplicate_count, get_missing_value_stats, read_uploaded_file
from preprocessing import detect_outliers_iqr
from visualization import (
    plot_boxplots,
    plot_correlation_heatmap,
    plot_histograms,
    plot_missing_values_bar,
    plot_target_distribution,
)


st.set_page_config(
    page_title="CleanSight AI",
    page_icon="🧹",
    layout="wide",
)

# ---------- Black & White professional theme ----------
st.markdown(
    """
    <style>
    :root {
        --bg: #ffffff;
        --text: #996814;
        --muted: #4b4b4b;
        --border: rgba(0, 0, 0, 0.12);
    }

    html, body, [class*="stApp"] {
        background-color: var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--border);
    }

.st-emotion-cache-1rb4m9n {
        background-color: #ffffff;
    }

    /* Ensure all Streamlit text is black */
    div, span, label, p, h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    .stPlotlyChart {
        color: var(--text);
    }

    /* Force plotly labels to black */
.js-plotly-plot .plotly text,
    .js-plotly-plot .plotly .infolayer text,
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #0b0b0b !important;
        color: #0b0b0b !important;
    }

    /* Fix Streamlit form controls (selectboxes, inputs) */
    div[data-baseweb] *,
    .stSelectbox div[data-baseweb],
    .stSelectbox div[data-baseweb] *,
    .stTextInput input,
    input[type='text'],
    textarea, button {
        color: var(--text) !important;
    }

    /* Force selectbox dropdown background + option text */
    div[role='listbox'] {
        background: #ffffff;
        border: 1px solid var(--border);
    }

    div[role='option'],
    div[role='option'] span {
        background: #ffffff !important;
        color: var(--text) !important;
    }

    /* Selected value text inside the collapsed select */
    .stSelectbox div[data-baseweb] span,
    .stSelectbox div[data-baseweb] label {
        color: var(--text) !important;
    }

    /* Hover/focus states */
    div[role='option']:hover,
    div[role='option'][aria-selected='true'] {
        background: #f3f3f3 !important;
        color: var(--text) !important;
    }


    /* Style primary button (Run Cleaning) */
    div.stButton > button[kind='primary'],
    button[kind='primary'],
    div.stButton > button {
        background: #ffffff !important;
        color: var(--text) !important;
        border: 1px solid var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ensure_dirs()


def dataset_metrics_for_report(df: pd.DataFrame, outlier_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Collect the metrics needed for the dataset health score."""
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    outliers = int(sum(outlier_counts.values())) if outlier_counts else 0
    return {"rows": rows, "cols": cols, "missing_cells": missing_cells, "duplicate_rows": duplicate_rows, "outliers": outliers}


def main():
    st.title("🧹 CleanSight AI")
    st.caption("Upload a dataset, clean it automatically, visualize results, and download an ML-ready CSV.")

    # Sidebar controls
    with st.sidebar:
        st.header("Cleaning Controls")


        st.subheader("Missing Value Handling")
        missing_strategy = st.selectbox(
            "Strategy",
            options=["none", "mean", "median", "mode", "knn"],
            index=0,
            help="Automatically fills missing values.",
        )
        knn_neighbors = st.number_input("KNN Neighbors", min_value=2, max_value=50, value=5, step=1)

        st.subheader("Duplicate Removal")
        remove_duplicates_flag = st.checkbox("Remove duplicate rows", value=True)

        st.subheader("Data Type Correction")
        type_correction_flag = st.checkbox("Auto-correct data types", value=True)

        st.subheader("Outlier Detection (IQR)")
        outlier_action = st.selectbox("Action", options=["none", "remove", "cap"], index=0)
        iqr_factor = st.slider("IQR Factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

        st.subheader("Categorical Encoding")
        encoding_method = st.selectbox("Encoding", options=["onehot", "label"], index=0)

        st.subheader("Feature Scaling")
        scaling_method = st.selectbox("Scaling", options=["standard", "minmax"], index=0)

        run_button = st.button("Run Cleaning", type="primary")

    # Upload section
    upload = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

    if not upload:
        st.info("Upload a dataset to begin.")
        return

    try:
        df_raw = read_uploaded_file(upload)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    if df_raw.empty:
        st.error("Uploaded file contains no data.")
        return

    # Display upload preview
    st.success("Dataset uploaded successfully!")

    st.subheader("Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("First 10 rows")
        st.dataframe(df_raw.head(10), use_container_width=True)

    with c2:
        st.metric("Rows", df_raw.shape[0])
        st.metric("Columns", df_raw.shape[1])

    # Target column selector is defined after dataset load.


    # Profiling dashboard
    st.header("📊 Dataset Profiling Dashboard")

    profile = profile_dataframe(df_raw)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", profile["rows"])
        st.metric("Columns", profile["cols"])
    with c2:
        st.metric("Missing Values", profile["missing_values"])
        st.metric("Duplicates", profile["duplicate_rows"])
    with c3:
        st.metric("Memory (MB)", f"{profile['memory_usage_mb']:.2f}")

    missing_series = get_missing_value_stats(df_raw)
    st.write("**Data Types Summary**")
    st.json(profile["dtypes_summary"])

    # Determine target dropdown options (after data is known)
    # Note: Streamlit reruns; we keep it simple by using a stable key.
    st.sidebar.subheader("Target Column")
    target_col = st.sidebar.selectbox(
        "(Optional) Target column",
        options=["— (not set)"] + df_raw.columns.astype(str).tolist(),
        index=0,
        key="target_col_selector",
    )
    if target_col == "— (not set)":
        target_col = None


    # Tabs for visualization and report
    tab_overview, tab_missing, tab_vis, tab_report, tab_download = st.tabs(
        ["Overview", "Missing Values", "Visualizations", "Data Quality Report", "Download"]
    )

    with tab_overview:
        col_num, col_cat = st.columns(2)
        with col_num:
            st.write("**Numerical Columns**")
            st.write(profile["numeric_columns"][:50])
        with col_cat:
            st.write("**Categorical Columns**")
            st.write(profile["categorical_columns"][:50])

        st.write("**Missing Values (Top Columns)**")
        top_missing = missing_series[missing_series > 0].head(15)
        st.dataframe(top_missing.to_frame("missing_count"), use_container_width=True)


    with tab_missing:
        st.plotly_chart(plot_missing_values_bar(df_raw), use_container_width=True)

    # Execute cleaning when user clicks
    df_clean = None
    health_before = None
    health_after = None
    outlier_counts_before = None

    # Compute outliers before for report (preview only)
    # Lightweight: run within cleaning later if needed; for health we re-use by running after.

    if run_button:
        with st.spinner("Running cleaning pipeline..."):
            outlier_action_param = None
            if outlier_action != "none":
                outlier_action_param = outlier_action

            try:
                df_clean, details = apply_preprocessing_pipeline(
                    df_raw,
                    missing_strategy=missing_strategy,
                    duplicate_removal=remove_duplicates_flag,
                    type_correction=type_correction_flag,
                    outlier_mode="iqr" if outlier_action_param else None,
                    outlier_action=outlier_action_param,
                    iqr_factor=iqr_factor,
                    encoding_method=encoding_method,
                    scaling_method=scaling_method,
                    knn_neighbors=int(knn_neighbors),
                )
            except Exception as e:
                st.error(f"Cleaning failed: {e}")
                return

        # Outlier counts after cleaning (approx) for health report
        from preprocessing import detect_outliers_iqr


        outlier_before = detect_outliers_iqr(df_raw)
        outlier_after = detect_outliers_iqr(df_clean)

        # Health score (0-100)
        before_score, after_score = compute_health_score(
            before=dataset_metrics_for_report(df_raw, outlier_counts=outlier_before.outlier_counts),
            after=dataset_metrics_for_report(df_clean, outlier_counts=outlier_after.outlier_counts),
        )


        st.subheader("✅ Cleaning Complete")
        st.write("**Preview of Cleaned Data**")
        st.dataframe(df_clean.head(10), use_container_width=True)

        # Health report
        with tab_report:
            st.metric("Before Score", before_score)
            st.metric("After Score", after_score)
            st.write("**Before (missing/duplicates/outliers)**")
            st.write({
                "missing_values": int(df_raw.isna().sum().sum()),
                "duplicates": int(get_duplicate_count(df_raw)),
                "outliers": int(sum(outlier_before.outlier_counts.values())),
            })
            st.write("**After (missing/duplicates/outliers)**")
            st.write({
                "missing_values": int(df_clean.isna().sum().sum()),
                "duplicates": int(get_duplicate_count(df_clean)),
                "outliers": int(sum(outlier_after.outlier_counts.values())),
            })

        # Visualizations
        with tab_vis:
            st.plotly_chart(plot_missing_values_bar(df_clean), use_container_width=True, key="vis_missing")
            st.plotly_chart(plot_correlation_heatmap(df_clean), use_container_width=True, key="vis_corr")
            st.plotly_chart(plot_boxplots(df_clean), use_container_width=True, key="vis_box")
            st.plotly_chart(plot_histograms(df_clean), use_container_width=True, key="vis_hist")
            st.plotly_chart(plot_target_distribution(df_clean, target_col), use_container_width=True, key="vis_target")


        # Download
        with tab_download:
            st.write("Ready to download cleaned dataset.")
            output_path = os.path.join("outputs", f"auto_cleaned_{os.path.splitext(upload.name)[0]}.csv")
            try:
                df_clean.to_csv(output_path, index=False)
                with open(output_path, "rb") as f:
                    btn = st.download_button(
                        label="Download Cleaned CSV",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="text/csv",
                    )
                    st.success(f"Saved to: {output_path}")
            except Exception as e:
                st.error(f"Failed to save/download cleaned data: {e}")


if __name__ == "__main__":
    main()

