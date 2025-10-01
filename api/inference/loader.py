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

# ---------- Globals ----------
_model = None
_model_lock = threading.Lock()
_expected_columns: List[str] | None = None


# ---------- Utils ----------
def _resolve_model_path(path_str: str) -> Path:
    """
    Resolve MODEL_PATH relative to the api/ directory (settings.BASE_DIR)
    and also the repo root (parent of api/), falling back to absolute.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    api_base = Path(settings.BASE_DIR)  # .../api
    candidates = [api_base / p, api_base.parent / p]
    for c in candidates:
        if c.exists():
            return c
    # Best-effort fallback (even if it may not exist yet)
    return api_base.parent / p


def _is_dataframe_like(obj: Any) -> bool:
    return isinstance(obj, (pd.DataFrame,))


# ---------- Loaders ----------
def _load_model_local() -> Any:
    """
    Load a sklearn/joblib pipeline from disk (MODEL_PATH).
    """
    model_path_env = os.getenv("MODEL_PATH", "artifacts/model_pipeline.joblib")
    resolved = _resolve_model_path(model_path_env)
    print(f"[inference] MODEL_SOURCE=local  MODEL_PATH={model_path_env} -> {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(
            f"MODEL_PATH not found at: {resolved}\n"
            "Set an absolute path or place the artifact under the repo root.\n"
            "Tip: if you run Django from 'api/', use MODEL_PATH=../artifacts/....joblib"
        )
    model = joblib.load(resolved)
    return model


def _load_model_mlflow() -> Any:
    """
    Load a model from the MLflow Model Registry.

    Env:
      - MLFLOW_TRACKING_URI       (e.g., http://host.docker.internal:5000)
      - MLFLOW_MODEL_URI          (optional: full URI like models:/Name/Production)
      - MLFLOW_MODEL_NAME         (default: MediWatchReadmit)
      - MLFLOW_MODEL_STAGE        (default: Production)
    """
    try:
        import mlflow
        import mlflow.sklearn  # for sklearn flavor
        import mlflow.pyfunc   # fallback
    except Exception as e:
        raise RuntimeError(
            "mlflow is not installed in the API environment. "
            "Add 'mlflow' to requirements.txt and rebuild the image."
        ) from e

    tracking = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking:
        mlflow.set_tracking_uri(tracking)
        print(f"[inference] Using MLflow tracking URI: {tracking}")
    else:
        print("[inference] WARNING: MLFLOW_TRACKING_URI is not set. "
              "If the model URI requires a server, loading will fail.")

    uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
    if not uri:
        name = os.getenv("MLFLOW_MODEL_NAME", "MediWatchReadmit").strip()
        stage = os.getenv("MLFLOW_MODEL_STAGE", "Production").strip()
        uri = f"models:/{name}/{stage}"

    print(f"[inference] Loading model from MLflow URI: {uri}")

    # Prefer sklearn flavor; fallback to pyfunc if needed.
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception:
        model = mlflow.pyfunc.load_model(uri)

    return model


def _load_model() -> Any:
    """
    Singleton loader controlled by MODEL_SOURCE env var:
      - MODEL_SOURCE=mlflow -> load from MLflow registry
      - else -> load from local joblib
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                mode = os.getenv("MODEL_SOURCE", "local").lower()
                if mode == "mlflow":
                    _model = _load_model_mlflow()
                else:
                    _model = _load_model_local()
                print("[inference] Model loaded.")
    return _model


# ---------- Public API ----------
def get_model() -> Any:
    return _load_model()


def get_expected_columns() -> List[str]:
    """
    Inspect the pipeline to infer expected input column names.
    Cached after first call.
    """
    global _expected_columns
    if _expected_columns is not None:
        return _expected_columns

    model = get_model()

    # Heuristics for sklearn Pipeline with a ColumnTransformer step named "pre"
    cols: List[str] | None = None
    try:
        # sklearn Pipeline
        if hasattr(model, "named_steps") and "pre" in model.named_steps:
            pre = model.named_steps["pre"]
            # Try to read columns from transformers_
            if hasattr(pre, "transformers_"):
                cols = []
                for name, transformer, column_selection in pre.transformers_:
                    if isinstance(column_selection, list):
                        cols.extend(column_selection)
                    elif str(column_selection) == "all":  # OneHotEncoder on all
                        # Can't reliably enumerate; ignore
                        pass
                cols = list(dict.fromkeys([c for c in cols if isinstance(c, str)]))  # unique + keep order
    except Exception as e:
        print(f"[inference] Could not infer expected columns from pipeline: {e}")

    # Fallback: environment hint
    if not cols:
        hint = os.getenv("INPUT_COLUMNS")  # comma-separated string
        if hint:
            cols = [c.strip() for c in hint.split(",") if c.strip()]

    # Final fallback to a safe default for this project
    if not cols:
        cols = [
            "race", "gender", "age",
            "time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses",
            "diabetesMed", "A1Cresult", "max_glu_serum",
        ]

    _expected_columns = cols
    return _expected_columns


def make_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert incoming JSON records into a DataFrame aligned to expected columns.
    Missing keys are filled with NA; extra keys are ignored (unless the pipeline handles them).
    """
    if not isinstance(records, list):
        raise ValueError("`records` must be a list of JSON objects.")

    cols = get_expected_columns()
    df = pd.DataFrame.from_records(records)

    # Keep only known columns (pipeline will handle encoding/Impute)
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = pd.NA

    # Reorder columns to align with training
    df = df[cols].copy()

    # Normalize simple NA-like strings
    df = df.replace({"?": pd.NA, "None": pd.NA, "NA": pd.NA})

    return df
