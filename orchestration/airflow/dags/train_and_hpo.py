from __future__ import annotations
import os, json, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np

from airflow.decorators import dag, task
from airflow.models.xcom_arg import XComArg

# ---- Config (env-overridable) ----
REPO_ROOT = Path(os.getenv("REPO_ROOT", "/opt/airflow"))  # inside the container
DATA_RAW = Path(os.getenv("DATA_RAW", "/opt/airflow/data/raw/diabetic_data.csv"))
SPLIT_DIR = Path(os.getenv("SPLIT_DIR", "/opt/airflow/data/splits"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/opt/airflow/artifacts"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "MediWatch Readmit")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "MediWatchReadmit")

# ---- Helpers (tiny preprocessing) ----
def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop obvious IDs if present
    drop_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Normalize 'readmitted' to binary target: '<30' -> 1 else 0
    if "readmitted" in df.columns:
        y = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)
        df = df.drop(columns=["readmitted"])
        df.insert(0, "target", y.values)
    # NA normalization
    df = df.replace({"?": np.nan, "None": np.nan, "NA": np.nan})
    return df

def _train_valid_test_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7 * n)
    n_valid = int(0.15 * n)
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train+n_valid]
    test_idx = idx[n_train+n_valid:]
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), df.iloc[test_idx].copy()

# ---- DAG ----
@dag(schedule=None, start_date=datetime(2024,1,1), catchup=False, tags=["mediwatch","mlflow","ray"])
def train_and_hpo():

    @task()
    def prepare_data() -> Dict[str, Any]:
        """
        Load raw diabetes CSV, clean, split, and persist CSV splits for reproducibility.
        """
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(DATA_RAW)
        df = _basic_clean(raw)
        assert "target" in df.columns, "Target not found after cleaning."
        train, valid, test = _train_valid_test_split(df, seed=42)

        train.to_csv(SPLIT_DIR / "train.csv", index=False)
        valid.to_csv(SPLIT_DIR / "valid.csv", index=False)
        test.to_csv(SPLIT_DIR / "test.csv", index=False)

        return {
            "n_train": len(train),
            "n_valid": len(valid),
            "n_test": len(test),
            "split_dir": str(SPLIT_DIR),
        }

    @task()
    def ray_tune_hpo(split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Ray Tune to optimize a simple HistGradientBoostingClassifier.
        Logs each trial to MLflow; returns best hyperparams.
        """
        import ray
        from ray import tune
        import mlflow
        from mlflow.models import infer_signature
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        split_dir = Path(split_info["split_dir"])
        train = pd.read_csv(split_dir / "train.csv")
        valid = pd.read_csv(split_dir / "valid.csv")

        y_train = train.pop("target").values
        y_valid = valid.pop("target").values

        cat = train.select_dtypes(include="object").columns.tolist()
        num = train.select_dtypes(exclude="object").columns.tolist()

        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])

        def trainable(config):
            # one run = one MLflow run
            with mlflow.start_run(nested=True):
                clf = HistGradientBoostingClassifier(
                    learning_rate=config["lr"],
                    max_depth=config["max_depth"],
                    max_leaf_nodes=config["max_leaf_nodes"],
                    l2_regularization=config["l2"],
                    random_state=42
                )
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                pipe.fit(train, y_train)

                proba = pipe.predict_proba(valid)[:, 1]
                auc = roc_auc_score(y_valid, proba)

                # Log params/metrics to MLflow
                mlflow.log_params(config)
                mlflow.log_metric("valid_auc", float(auc))

                # Small input example for signature
                ex = valid.head(5)
                sig = infer_signature(ex, pipe.predict_proba(ex)[:,1])
                mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=ex, signature=sig)

                # Report back to Ray
                tune.report(valid_auc=auc)

        search_space = {
            "lr": tune.loguniform(1e-3, 0.2),
            "max_depth": tune.randint(3, 13),
            "max_leaf_nodes": tune.randint(15, 65),
            "l2": tune.loguniform(1e-6, 1e-1),
        }

        # Start Ray in-process (no cluster needed)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(mode="max", metric="valid_auc", num_samples=20),
        )
        results = tuner.fit()
        best = results.get_best_result()
        best_config = best.config
        best_auc = best.metrics["valid_auc"]

        # Save best params to artifacts for traceability
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_DIR / "best_params.json", "w") as f:
            json.dump({"config": best_config, "valid_auc": best_auc}, f, indent=2)

        return {"best_params": best_config, "best_valid_auc": float(best_auc)}

    @task()
    def train_best_and_register(hpo_out: Dict[str, Any], split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train best model on train+valid, evaluate on test, log to MLflow, register in Model Registry.
        """
        import mlflow
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, f1_score

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        split_dir = Path(split_info["split_dir"])
        train = pd.read_csv(split_dir / "train.csv")
        valid = pd.read_csv(split_dir / "valid.csv")
        test = pd.read_csv(split_dir / "test.csv")

        # Combine train+valid for final fit
        y_train = train.pop("target").values
        y_valid = valid.pop("target").values
        y_test = test.pop("target").values
        X_train_full = pd.concat([train, valid], axis=0).reset_index(drop=True)
        y_train_full = np.concatenate([y_train, y_valid], axis=0)

        cat = X_train_full.select_dtypes(include="object").columns.tolist()
        num = X_train_full.select_dtypes(exclude="object").columns.tolist()

        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])

        cfg = hpo_out["best_params"]
        clf = HistGradientBoostingClassifier(
            learning_rate=cfg["lr"],
            max_depth=cfg["max_depth"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            l2_regularization=cfg["l2"],
            random_state=42
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        with mlflow.start_run() as run:
            pipe.fit(X_train_full, y_train_full)

            proba_test = pipe.predict_proba(test)[:, 1]
            pred_test = (proba_test >= 0.5).astype(int)
            test_auc = float(roc_auc_score(y_test, proba_test))
            test_f1 = float(f1_score(y_test, pred_test))

            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_params(cfg)

            ex = test.head(5)
            sig = infer_signature(ex, pipe.predict_proba(ex)[:,1])
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=ex, signature=sig)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

            # Optional: add tags
            client = MlflowClient()
            client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "source", "airflow_train_and_hpo")
            client.set_model_version_tag(REGISTERED_MODEL_NAME, result.version, "test_auc", str(test_auc))

            return {"registered_model": REGISTERED_MODEL_NAME, "version": result.version, "test_auc": test_auc}

    info = prepare_data()
    hpo = ray_tune_hpo(info)
    reg = train_best_and_register(hpo, info)

train_and_hpo()
