# api/inference/loader.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from mediwatch_api.model_loader import load_model  # <-- use unified loader

# Keep the same public API used by your views
def get_model():
    """
    Returns the loaded model (baked-in joblib if available, else MLflow).
    Cached by model_loader.load_model via @lru_cache.
    """
    return load_model()

def get_expected_columns() -> List[str]:
    """
    Returns the expected RAW input columns before preprocessing.
    Works if the loaded model is a scikit-learn Pipeline or has feature_names_in_.
    If the model is an MLflow pyfunc and does not expose these, returns [].
    """
    pipe = get_model()

    # Try sklearn-style attributes
    pre = getattr(pipe, "named_steps", {}).get("pre", None) if hasattr(pipe, "named_steps") else None

    cols = None
    if pre is not None:
        cols = getattr(pre, "feature_names_in_", None)

    if cols is None:
        cols = getattr(pipe, "feature_names_in_", None)

    if cols is not None:
        try:
            return list(cols)
        except Exception:
            return [str(c) for c in cols]

    # If we cannot infer (e.g., MLflow pyfunc), return empty (we'll not reorder)
    return []

def make_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame from incoming JSON, aligned to expected columns when known.
    """
    df = pd.DataFrame.from_records(records)

    expected = get_expected_columns()
    if expected:
        for c in expected:
            if c not in df.columns:
                df[c] = np.nan  # keep sklearn compatibility
        df = df[expected]

    # Normalize NA values
    df = df.replace({pd.NA: np.nan})
    df = df.where(pd.notnull(df), np.nan)

    # Tidy object columns
    if not df.empty:
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols) > 0:
            df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())
            noneish = {"", "none", "null", "nan", "<na>"}
            for col in obj_cols:
                mask = df[col].str.lower().isin(noneish)
                if mask.any():
                    df.loc[mask, col] = np.nan

    return df
