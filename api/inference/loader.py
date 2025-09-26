# api/inference/loader.py
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

_model = None
_lock = threading.Lock()


def _resolve_model_path(path_str: str) -> Path:
    """
    Resolve MODEL_PATH robustly:
      - absolute paths are returned as-is
      - relative paths are tried relative to:
          1) settings.BASE_DIR (api/)
          2) repo root (api/..)
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    api_base = Path(settings.BASE_DIR)
    candidates = [
        api_base / p,           # e.g., api/artifacts/model_pipeline.joblib
        api_base.parent / p,    # e.g., <repo-root>/artifacts/model_pipeline.joblib
    ]
    for c in candidates:
        if c.exists():
            return c

    # If not found, default to repo-root candidate (useful for error messages)
    return candidates[-1]


def _load_model() -> Any:
    """
    Load the joblib pipeline once, lazily, in a thread-safe way.
    MODEL_PATH can be absolute or relative (see _resolve_model_path).
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                path_env = os.getenv("MODEL_PATH", "artifacts/model_pipeline.joblib")
                resolved = _resolve_model_path(path_env)
                print(f"[inference] MODEL_PATH={path_env} -> resolved={resolved}")
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"MODEL_PATH not found at: {resolved}\n"
                        f"Set an absolute path or place the artifact under the repo root.\n"
                        f"Tip: if you run Django from 'api/', use MODEL_PATH=../artifacts/....joblib"
                    )
                _model = joblib.load(resolved)
                print("[inference] Model loaded.")
    return _model


def get_model():
    """Public accessor used by views."""
    return _load_model()


def get_expected_columns() -> List[str]:
    """
    Returns the expected RAW input columns before preprocessing.
    We read them from the saved ColumnTransformer (or pipeline) metadata.
    Avoid boolean evaluation on arrays (no `a or b`).
    """
    pipe = get_model()
    pre = pipe.named_steps.get("pre")

    cols = None
    if pre is not None:
        cols = getattr(pre, "feature_names_in_", None)

    if cols is None:
        cols = getattr(pipe, "feature_names_in_", None)

    if cols is not None:
        # cols may be a numpy array or Index; normalize to list
        try:
            return list(cols)
        except Exception:
            return [str(c) for c in cols]

    return []


def make_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame from incoming JSON, aligned to expected columns.

    - Adds any missing expected columns (filled with np.nan)
    - Reorders columns to match training
    - Normalizes missing sentinels to np.nan (avoids NAType float() errors)
    - Tidies string/object columns (strip + map common 'none-ish' tokens to np.nan)
    """
    df = pd.DataFrame.from_records(records)

    # Align to expected raw schema
    expected = get_expected_columns()
    if expected:
        for c in expected:
            if c not in df.columns:
                df[c] = np.nan  # use np.nan (NOT pd.NA) for sklearn compatibility
        df = df[expected]

    # Normalize pandas NA scalars and any nulls to np.nan
    df = df.replace({pd.NA: np.nan})
    df = df.where(pd.notnull(df), np.nan)

    # Tidy string/object columns
    if not df.empty:
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols) > 0:
            # ensure string dtype then strip
            df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())

            # map common none-ish tokens to NaN (case-insensitive)
            noneish = {"", "none", "null", "nan", "<na>"}
            for col in obj_cols:
                # produce a boolean Series; never use the Series directly in `if`
                mask = df[col].str.lower().isin(noneish)
                if mask.any():
                    df.loc[mask, col] = np.nan

    return df
