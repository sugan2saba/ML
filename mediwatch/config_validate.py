# mediwatch/config_validate.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import os
import pandas as pd

@dataclass
class ValidationResult:
    ok: bool
    msgs: List[str]

    def raise_if_failed(self):
        if not self.ok:
            bullet = "\n - "
            raise RuntimeError("Config/Data validation failed:" + bullet + bullet.join(self.msgs))

def _file_exists(p: Path, msgs: List[str], label: str):
    if not p.exists():
        msgs.append(f"{label} missing: {p}")
        return False
    return True

def validate_config_and_data(cfg: dict, strict_columns: bool = True) -> ValidationResult:
    msgs: List[str] = []
    ok = True

    # ---- Resolve paths from config
    processed_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    x_train = processed_dir / cfg["data"].get("x_train", "X_train.parquet")
    y_train = processed_dir / cfg["data"].get("y_train", "y_train.parquet")
    x_valid = processed_dir / cfg["data"].get("x_valid", "X_valid.parquet")
    y_valid = processed_dir / cfg["data"].get("y_valid", "y_valid.parquet")
    x_test  = processed_dir / cfg["data"].get("x_test",  "X_test.parquet")
    y_test  = processed_dir / cfg["data"].get("y_test",  "y_test.parquet")
    target_col = cfg["data"].get("target_col", "readmit_30d")

    reports_dir = Path(cfg.get("output", {}).get("reports_dir", "reports"))
    artifacts_dir = Path(cfg.get("output", {}).get("artifacts_dir", "artifacts"))

    # ---- Basic path checks
    if not processed_dir.exists():
        msgs.append(f"Processed directory not found: {processed_dir}")
        ok = False

    for pth, label in [
        (x_train, "X_train"), (y_train, "y_train"),
        (x_valid, "X_valid"), (y_valid, "y_valid"),
        (x_test,  "X_test"),  (y_test,  "y_test")
    ]:
        ok = _file_exists(pth, msgs, label) and ok

    # Ensure output dirs exist (create if missing)
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        msgs.append(f"Could not create output dirs: {e}")
        ok = False

    # ---- Load small samples to verify readability & schema
    try:
        Xt = pd.read_parquet(x_train, engine="pyarrow").head(5)
        Yt = pd.read_parquet(y_train, engine="pyarrow").head(5)
        Xv = pd.read_parquet(x_valid, engine="pyarrow").head(5)
        Yv = pd.read_parquet(y_valid, engine="pyarrow").head(5)
        Xs = pd.read_parquet(x_test,  engine="pyarrow").head(5)
        Ys = pd.read_parquet(y_test,  engine="pyarrow").head(5)
    except Exception as e:
        msgs.append(f"Failed reading parquet files (pyarrow). Error: {e}")
        return ValidationResult(False, msgs)

    # ---- Target column presence / type sanity
    for df_y, label in [(Yt, "y_train"), (Yv, "y_valid"), (Ys, "y_test")]:
        if target_col not in df_y.columns:
            msgs.append(f"Target column '{target_col}' not found in {label}")
            ok = False
        else:
            # type sanity: should be numeric/binary
            dtype = df_y[target_col].dtype
            if not (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_float_dtype(dtype)):
                msgs.append(f"Target column '{target_col}' in {label} has suspicious dtype: {dtype}")

    # ---- Column alignment across splits
    cols_train = list(Xt.columns)
    cols_valid = list(Xv.columns)
    cols_test  = list(Xs.columns)

    if strict_columns:
        if cols_train != cols_valid:
            msgs.append("Feature columns differ between X_train and X_valid.")
            # Optional: show differences
            diff1 = set(cols_train) - set(cols_valid)
            diff2 = set(cols_valid) - set(cols_train)
            if diff1: msgs.append(f"  In train not in valid: {sorted(list(diff1))[:10]}...")
            if diff2: msgs.append(f"  In valid not in train: {sorted(list(diff2))[:10]}...")
            ok = False
        if cols_train != cols_test:
            msgs.append("Feature columns differ between X_train and X_test.")
            diff1 = set(cols_train) - set(cols_test)
            diff2 = set(cols_test) - set(cols_train)
            if diff1: msgs.append(f"  In train not in test: {sorted(list(diff1))[:10]}...")
            if diff2: msgs.append(f"  In test not in train: {sorted(list(diff2))[:10]}...")
            ok = False
    else:
        # Soft check: just warn
        if set(cols_train) != set(cols_valid):
            msgs.append("WARN: set of features differs between train and valid (soft check).")
        if set(cols_train) != set(cols_test):
            msgs.append("WARN: set of features differs between train and test (soft check).")

    # ---- MLflow tracking URI presence (warn only)
    tracking_env = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_env and not cfg.get("mlflow", {}).get("tracking_uri"):
        msgs.append("WARN: MLflow tracking URI not set (using MLflow defaults). You can export MLFLOW_TRACKING_URI or set mlflow.tracking_uri in config.")

    return ValidationResult(ok, msgs)
