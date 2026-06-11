import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dirs() -> None:
    """Ensure required folders (outputs) exist."""
    os.makedirs(os.path.join("outputs"), exist_ok=True)
    os.makedirs(os.path.join("datasets"), exist_ok=True)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or XLSX file into a pandas DataFrame."""
    filename = getattr(uploaded_file, "name", "") or "uploaded"
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        return pd.read_csv(uploaded_file)
    if ext in {".xlsx", ".xls"}:
        # openpyxl is required for xlsx
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")


def get_missing_value_stats(df: pd.DataFrame) -> pd.Series:
    """Return missing counts per column."""
    return df.isna().sum().sort_values(ascending=False)


def get_duplicate_count(df: pd.DataFrame) -> int:
    """Return number of duplicate rows in the DataFrame."""
    return int(df.duplicated().sum())


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numerical and categorical based on dtype heuristics."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns.tolist() if c not in numeric_cols]
    return numeric_cols, cat_cols


def estimate_memory_usage_mb(df: pd.DataFrame) -> float:
    """Estimate memory usage of the dataframe in MB."""
    mem_bytes = df.memory_usage(deep=True).sum()
    return float(mem_bytes / (1024 * 1024))


_NUMERIC_STRING_RE = re.compile(r"^[\s]*[-+]?((\d+\.?\d*)|(\d*\.?\d+))([eE][-+]?\d+)?[\s]*$")


def try_convert_numeric_series(s: pd.Series) -> pd.Series:
    """Try converting a series with numeric strings to numeric dtype.

    Leaves non-convertible values as-is.
    """
    if s.dtype.kind in {"i", "u", "f"}:
        return s

    if not (s.dtype == "object" or pd.api.types.is_string_dtype(s.dtype)):
        return s

    # Normalize common missing markers
    s2 = s.copy()
    s2 = s2.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "NULL": np.nan, "null": np.nan})

    # Only attempt conversion if values look numeric often enough
    sample = s2.dropna().astype(str)
    if sample.empty:
        return s

    ratio = sample.map(lambda x: bool(_NUMERIC_STRING_RE.match(x))).mean()
    if ratio < 0.7:
        return s

    converted = pd.to_numeric(s2, errors="coerce")
    # If conversion produced a lot of non-nulls, keep it
    if converted.notna().mean() >= 0.7:
        return converted
    return s


def try_parse_datetime_series(s: pd.Series) -> pd.Series:
    """Try converting a series into datetime dtype if it looks datetime-like."""
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return s

    if not (s.dtype == "object" or pd.api.types.is_string_dtype(s.dtype)):
        return s

    s2 = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    sample = s2.dropna().astype(str)
    if sample.empty:
        return s

    # Heuristic: presence of typical datetime patterns
    looks_like = sample.head(50).str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean()
    looks_like = looks_like or sample.head(50).str.contains(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}").mean()

    if looks_like < 0.2:
        return s

    parsed = pd.to_datetime(s2, errors="coerce", infer_datetime_format=True, utc=False)
    if parsed.notna().mean() >= 0.5:
        return parsed
    return s


def correct_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Attempt to correct common data type issues.

    Returns:
        cleaned_df, changes_dict where value describes what conversion happened.
    """
    out = df.copy()
    changes: Dict[str, str] = {}

    for col in out.columns:
        original_dtype = out[col].dtype
        new_series = try_convert_numeric_series(out[col])
        if new_series.dtype != original_dtype:
            out[col] = new_series
            changes[col] = f"numeric_string_to_{out[col].dtype}"
            continue

        new_series2 = try_parse_datetime_series(out[col])
        if new_series2.dtype != original_dtype:
            out[col] = new_series2
            changes[col] = f"datetime_parse_to_{out[col].dtype}"

    return out, changes


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names to be ML-friendly."""
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in out.columns]
    out.columns = [re.sub(r"[^0-9a-zA-Z_\-]", "", str(c)) for c in out.columns]
    return out


def safe_top_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top n rows for display."""
    return df.head(n).copy()


def compute_health_score(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[int, int]:
    """Compute dataset health score (0-100) from before/after metrics.

    A simple beginner-friendly scoring heuristic:
    - Missingness: penalize proportion of missing values
    - Duplicates: penalize duplicated row ratio
    - Outliers: penalize outlier counts ratio
    """

    def score(metrics: Dict[str, Any]) -> int:
        rows = max(int(metrics.get("rows", 1)), 1)
        cols = max(int(metrics.get("cols", 1)), 1)
        total_cells = rows * cols

        missing_cells = float(metrics.get("missing_cells", 0.0))
        missing_ratio = min(missing_cells / total_cells, 1.0)

        dup_rows = float(metrics.get("duplicate_rows", 0.0))
        dup_ratio = min(dup_rows / rows, 1.0)

        outliers = float(metrics.get("outliers", 0.0))
        # outliers is an absolute count across numeric columns
        outlier_ratio = min(outliers / max(total_cells, 1), 1.0)

        # Convert to rewards/penalties
        penalty = 0.45 * missing_ratio + 0.25 * dup_ratio + 0.30 * outlier_ratio
        s = int(round(100 * (1.0 - penalty)))
        return int(max(0, min(100, s)))

    return score(before), score(after)

