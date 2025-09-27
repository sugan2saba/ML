# api/mediwatch_api/model_loader.py
import os, pathlib
from functools import lru_cache

def _joblib_exists(p: str) -> bool:
    try:
        return pathlib.Path(p).exists()
    except Exception:
        return False

@lru_cache(maxsize=1)
def load_model():
    """
    Loads a model either from baked-in joblib (preferred) or from MLflow.
    """
    p = os.getenv("MODEL_PATH", "/models/current.joblib")
    if _joblib_exists(p):
        import joblib
        return joblib.load(p)

    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    name = os.getenv("MODEL_NAME", "HGB")
    stage = os.getenv("MODEL_STAGE")       # e.g., Production
    version = os.getenv("MODEL_VERSION")   # e.g., 3

    if stage:
        uri = f"models:/{name}/{stage}"
    elif version:
        uri = f"models:/{name}/{version}"
    else:
        uri = f"models:/{name}/Production"

    # Try sklearn flavor first; fall back to pyfunc
    try:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri)
    except Exception:
        import mlflow.pyfunc
        return mlflow.pyfunc.load_model(uri)

def model_info() -> dict:
    """
    Returns metadata for the active model: source (baked/mlflow),
    selected name, stage/version, and resolved file path when baked.
    """
    info = {
        "source": None,          # "baked" or "mlflow"
        "path": None,            # resolved path if baked
        "name": os.getenv("MODEL_NAME", None),
        "stage": os.getenv("MODEL_STAGE", None),
        "version": os.getenv("MODEL_VERSION", None),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", None),
        "mode_hint": os.getenv("MODEL_MODE", None),  # set by entrypoint when falling back to mlflow
    }

    p = os.getenv("MODEL_PATH", "/models/current.joblib")
    if _joblib_exists(p):
        info["source"] = "baked"
        info["path"] = str(pathlib.Path(p).resolve())
        # If MODEL_NAME isn't set, try to infer from filename
        if not info["name"]:
            info["name"] = pathlib.Path(p).name
        return info

    # No local file → MLflow mode
    info["source"] = "mlflow"
    return info
