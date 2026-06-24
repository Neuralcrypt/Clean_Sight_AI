from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from utils import correct_data_types


@dataclass
class OutlierResult:
    """Holds outlier bounds and counts."""

    bounds: Dict[str, Tuple[float, float]]
    outlier_counts: Dict[str, int]


def profile_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Compute dataset profiling metrics."""
    rows, cols = df.shape

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    dtypes_summary = df.dtypes.astype(str).to_dict()

    return {
        "rows": rows,
        "cols": cols,
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "memory_usage_mb": memory_mb,
        "dtypes_summary": dtypes_summary,
    }


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    knn_neighbors: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fill missing values using the selected strategy.

    For categorical columns:
    - mean/median are applied only to numeric columns
    - mode is applied to categorical columns

    Args:
        df: Input dataframe.
        strategy: one of {mean, median, mode, knn}
        knn_neighbors: number of neighbors for KNN imputation.

    Returns:
        cleaned_df, details
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in out.columns if c not in numeric_cols]

    details: Dict[str, object] = {"strategy": strategy}

    if strategy == "mean":
        for c in numeric_cols:
            out[c] = out[c].fillna(out[c].mean())
        for c in categorical_cols:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].mode(dropna=True).iloc[0] if not out[c].mode(dropna=True).empty else out[c])

    elif strategy == "median":
        for c in numeric_cols:
            out[c] = out[c].fillna(out[c].median())
        for c in categorical_cols:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].mode(dropna=True).iloc[0] if not out[c].mode(dropna=True).empty else out[c])

    elif strategy == "mode":
        for c in out.columns:
            if out[c].isna().any():
                modes = out[c].mode(dropna=True)
                if not modes.empty:
                    out[c] = out[c].fillna(modes.iloc[0])

    elif strategy == "knn":
        # KNN works on numeric only; non-numeric columns will be filled with mode.
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            numeric_data = imputer.fit_transform(out[numeric_cols])
            out[numeric_cols] = pd.DataFrame(numeric_data, columns=numeric_cols, index=out.index)

        for c in categorical_cols:
            if out[c].isna().any():
                modes = out[c].mode(dropna=True)
                if not modes.empty:
                    out[c] = out[c].fillna(modes.iloc[0])

    else:
        raise ValueError("Invalid missing value strategy.")

    details["missing_after"] = int(out.isna().sum().sum())
    return out, details


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove duplicate rows from the dataframe."""
    duplicates_before = int(df.duplicated().sum())
    out = df.drop_duplicates().reset_index(drop=True)
    duplicates_after = int(out.duplicated().sum())
    return out, {"duplicates_before": duplicates_before, "duplicates_removed": duplicates_before - duplicates_after}


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> OutlierResult:
    """Detect outliers for numerical columns using the IQR method."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bounds: Dict[str, Tuple[float, float]] = {}
    outlier_counts: Dict[str, int] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            bounds[col] = (np.nan, np.nan)
            outlier_counts[col] = 0
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr

        bounds[col] = (low, high)
        outlier_counts[col] = int(((df[col] < low) | (df[col] > high)).sum())

    return OutlierResult(bounds=bounds, outlier_counts=outlier_counts)


def remove_or_cap_outliers(
    df: pd.DataFrame,
    mode: str,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Remove or cap outliers using IQR bounds."""
    result = detect_outliers_iqr(df, factor=factor)
    out = df.copy()

    if mode == "remove":
        mask = pd.Series(True, index=out.index)
        for col, (low, high) in result.bounds.items():
            if np.isnan(low) or np.isnan(high):
                continue
            mask &= ~((out[col] < low) | (out[col] > high))
        out = out.loc[mask].reset_index(drop=True)

    elif mode == "cap":
        for col, (low, high) in result.bounds.items():
            if np.isnan(low) or np.isnan(high):
                continue
            out[col] = out[col].clip(lower=low, upper=high)

    else:
        raise ValueError("mode must be 'remove' or 'cap'.")

    details: Dict[str, object] = {
        "mode": mode,
        "outlier_counts": result.outlier_counts,
        "rows_before": df.shape[0],
        "rows_after": out.shape[0],
    }
    return out, details


def encode_categoricals(
    df: pd.DataFrame,
    method: str,
    max_cardinality_for_encoding: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Encode categorical columns using label or one-hot encoding."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in out.columns if c not in numeric_cols]

   
    selected_cat_cols = [c for c in cat_cols if out[c].nunique(dropna=True) <= max_cardinality_for_encoding]
    dropped_cat_cols = [c for c in cat_cols if c not in selected_cat_cols]

    details: Dict[str, object] = {"method": method, "encoded_columns": selected_cat_cols, "dropped_high_cardinality": dropped_cat_cols}

    if method == "label":
        le_models: Dict[str, LabelEncoder] = {}
        for c in selected_cat_cols:
            le = LabelEncoder()
            
            out[c] = out[c].astype(str).fillna("__missing__")
            out[c] = le.fit_transform(out[c])
            le_models[c] = le
       
        for c in dropped_cat_cols:
            le = LabelEncoder()
            out[c] = out[c].astype(str).fillna("__missing__")
            out[c] = le.fit_transform(out[c])
            le_models[c] = le

    elif method == "onehot":
       
        if selected_cat_cols:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            
            cat_data = out[selected_cat_cols].astype(str).fillna("__missing__")
            encoded = enc.fit_transform(cat_data)
            feature_names = enc.get_feature_names_out(selected_cat_cols)
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=out.index)

            out = out.drop(columns=selected_cat_cols)
            out = pd.concat([out, encoded_df], axis=1)

      
        for c in dropped_cat_cols:
            le = LabelEncoder()
            out[c] = out[c].astype(str).fillna("__missing__")
            out[c] = le.fit_transform(out[c])

    else:
        raise ValueError("Invalid encoding method.")

    return out, details


def scale_features(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Scale numerical columns only."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return out, {"method": method, "scaled_columns": []}

    details: Dict[str, object] = {"method": method, "scaled_columns": numeric_cols}

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method.")

    scaled = scaler.fit_transform(out[numeric_cols])
    out[numeric_cols] = pd.DataFrame(scaled, columns=numeric_cols, index=out.index)

    return out, details


def apply_preprocessing_pipeline(
    df: pd.DataFrame,
    missing_strategy: str,
    duplicate_removal: bool,
    type_correction: bool,
    outlier_mode: Optional[str],
    outlier_action: Optional[str],
    iqr_factor: float,
    encoding_method: str,
    scaling_method: str,
    knn_neighbors: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run the end-to-end preprocessing pipeline based on user options."""

    details: Dict[str, object] = {}
    out = df.copy()

    if type_correction:
        out, changes = correct_data_types(out)
        details["type_correction_changes"] = changes

    if duplicate_removal:
        out, dup_details = remove_duplicates(out)
        details["duplicate_handling"] = dup_details

    if missing_strategy != "none":
        out, miss_details = handle_missing_values(out, strategy=missing_strategy, knn_neighbors=knn_neighbors)
        details["missing_value_handling"] = miss_details

    if outlier_mode in {"iqr"} and outlier_action in {"remove", "cap"}:
        out, out_details = remove_or_cap_outliers(out, mode=outlier_action, factor=iqr_factor)
        details["outlier_handling"] = out_details

   
    out, enc_details = encode_categoricals(out, method=encoding_method)
    details["categorical_encoding"] = enc_details

    out, sc_details = scale_features(out, method=scaling_method)
    details["feature_scaling"] = sc_details

    return out, details

