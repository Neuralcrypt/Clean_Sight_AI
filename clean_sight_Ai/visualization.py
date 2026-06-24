from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_missing_values_bar(df: pd.DataFrame, top_n: int = 20):
    """Create a Plotly bar chart for missing values."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if missing.empty:
        fig = go.Figure()
        fig.update_layout(title="Missing Values: None")
        return fig

    fig = px.bar(x=missing.index.astype(str), y=missing.values, labels={"x": "Column", "y": "Missing Count"})
    fig.update_layout(title="Missing Values by Column", xaxis_tickangle=-45)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson"):
    """Create a correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Correlation Heatmap: Not enough numeric columns")
        return fig

    corr = numeric.corr(method=method)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Correlation Heatmap")
    return fig


def plot_boxplots(df: pd.DataFrame, max_cols: int = 8):
    """Create boxplots for numeric columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = numeric_cols[:max_cols]

    if not numeric_cols:
        fig = go.Figure()
        fig.update_layout(title="Boxplots: No numeric columns")
        return fig

    melt_df = df[numeric_cols].melt(var_name="feature", value_name="value")
    fig = px.box(melt_df, x="feature", y="value", points=False)
    fig.update_layout(title="Boxplots for Numerical Features", xaxis_tickangle=-45)
    return fig


def plot_histograms(df: pd.DataFrame, max_cols: int = 6):
    """Create histograms for numerical feature distributions."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:max_cols]

    if not numeric_cols:
        fig = go.Figure()
        fig.update_layout(title="Histograms: No numeric columns")
        return fig

    fig = px.histogram(df, x=numeric_cols[0], nbins=30, marginal="box", title=f"Histogram: {numeric_cols[0]}")

   
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: Optional[str]):
    """Create a distribution plot for a target column if provided."""
    if not target_col or target_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Target Distribution: Not provided")
        return fig

    series = df[target_col]

    if pd.api.types.is_numeric_dtype(series.dtype):
        fig = px.histogram(df, x=target_col, nbins=30, marginal="box", title=f"Target Distribution: {target_col}")
    else:
        value_counts = series.astype(str).value_counts().head(30)
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Target Categories: {target_col}")
        fig.update_layout(xaxis_tickangle=-45)

    return fig

