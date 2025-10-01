# scripts/train_baseline_mlflow.py
from __future__ import annotations

import os
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from mediwatch.preprocess import build_preprocessor
from mediwatch.config import load_config, get_path
from mediwatch.config_validate import validate_config_and_data
from mediwatch.align import align_like_train


# ----------------------------
# Data & plotting helpers
# ----------------------------
def load_splits(cfg: dict) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    data_dir = Path(get_path(cfg, "data", "processed_dir", default="data/processed"))
    tgt = get_path(cfg, "data", "target_col", default="readmit_30d")
    Xtr = pd.read_parquet(data_dir / get_path(cfg, "data", "x_train", default="X_train.parquet"))
    ytr = pd.read_parquet(data_dir / get_path(cfg, "data", "y_train", default="y_train.parquet"))[tgt].astype(int).values
    Xva = pd.read_parquet(data_dir / get_path(cfg, "data", "x_valid", default="X_valid.parquet"))
    yva = pd.read_parquet(data_dir / get_path(cfg, "data", "y_valid", default="y_valid.parquet"))[tgt].astype(int).values
    Xte = pd.read_parquet(data_dir / get_path(cfg, "data", "x_test",  default="X_test.parquet"))
    yte = pd.read_parquet(data_dir / get_path(cfg, "data", "y_test",  default="y_test.parquet"))[tgt].astype(int).values
    return Xtr, ytr, Xva, yva, Xte, yte


def log_curves(y_true, y_prob, tag: str, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title(f"ROC ({tag})"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    ax[1].plot(rec, prec, label=f"PR-AUC={prauc:.3f}")
    ax[1].set_title(f"PR ({tag})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()

    fig.tight_layout()
    out = out_dir / f"curves_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path=f"plots/{tag}")


# ----------------------------
# Model factory
# ----------------------------
def make_model(kind: str, args, cfg: dict):
    if kind == "lr":
        max_iter = args.max_iter if args.max_iter is not None else get_path(cfg, "baseline", "lr", "max_iter", default=200)
        C = args.C if args.C is not None else get_path(cfg, "baseline", "lr", "C", default=1.0)
        # NOTE: n_jobs is not a valid arg for lbfgs; omit it to avoid errors
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            C=C,
            class_weight="balanced",
            random_state=42,
        )
        params = {"model": "LogisticRegression", "solver": "lbfgs", "max_iter": max_iter, "C": C}
    elif kind == "rf":
        n_estimators = args.n_estimators if args.n_estimators is not None else get_path(cfg, "baseline", "rf", "n_estimators", default=400)
        max_depth_cfg = args.max_depth if args.max_depth is not None else get_path(cfg, "baseline", "rf", "max_depth", default=0)
        max_depth = None if (max_depth_cfg in [None, 0]) else max_depth_cfg
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        params = {"model": "RandomForest", "n_estimators": n_estimators, "max_depth": max_depth}
    else:
        raise ValueError("Unknown --model. Use 'lr' or 'rf'.")
    return model, params


# ----------------------------
# Registry helper
# ----------------------------
def maybe_register_and_stage(
    run_id: str,
    model_artifact_subpath: str,
    model_name: str,
    stage: Optional[str],
    archive_existing: bool = True
):
    """
    Register the logged model (artifact) under 'model_name' and optionally
    transition the new version to a stage (Staging/Production).
    """
    model_uri = f"runs:/{run_id}/{model_artifact_subpath}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"[registry] Registered {model_name} as version {result.version}")

    if stage:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        print(f"[registry] Transitioned {model_name} v{result.version} -> {stage}")

    return result.version


# ----------------------------
# Main
# ----------------------------
def main(args):
    # Load environment (.env) and decide tracking mode
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    # MLflow tracking URI priority: CLI > env > fallback
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri.strip())
        print(f"[mlflow] Using CLI tracking URI: {mlflow.get_tracking_uri()} (Option B)")
    else:
        env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if env_uri:
            mlflow.set_tracking_uri(env_uri)
            print(f"[mlflow] Using env tracking URI: {env_uri} (Option B)")
        else:
            mlflow.set_tracking_uri("./mlruns")
            print("[mlflow] No MLFLOW_TRACKING_URI set; defaulting to ./mlruns (Option A)")
            print("         To enable Model Registry, start a server backed by a DB (e.g. SQLite).")

    # Experiment selection
    if args.experiment:
        experiment = args.experiment
    else:
        cfg_tmp = load_config(args.config)
        experiment = get_path(cfg_tmp, "mlflow", "experiment", default="MediWatch-Readmit")
    mlflow.set_experiment(experiment)

    # Registered model name selection (CLI > env > default)
    model_name = (
        args.model_name
        or os.getenv("MLFLOW_MODEL_NAME")
        or "MediWatchReadmit"
    )

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", experiment)
    print("Registered Model (if --register):", model_name)

    # Load config & validate inputs
    cfg = load_config(args.config)
    result = validate_config_and_data(cfg, strict_columns=False)
    for m in result.msgs:
        print("[validate]", m)
    result.raise_if_failed()

    # I/O dirs
    reports_dir = Path(get_path(cfg, "output", "reports_dir", default="reports")); reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(get_path(cfg, "output", "artifacts_dir", default="artifacts")); artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Data
    Xtr, ytr, Xva, yva, Xte, yte = load_splits(cfg)

    # Align schemas for safety
    Xva, miss_v, extra_v, ch_v = align_like_train(Xtr, Xva)
    Xte, miss_t, extra_t, ch_t = align_like_train(Xtr, Xte)
    if ch_v:
        print(f"[auto-align] valid: +{len(miss_v)} missing, -{len(extra_v)} extra")
    if ch_t:
        print(f"[auto-align] test:  +{len(miss_t)} missing, -{len(extra_t)} extra")

    # Preprocessor & model
    pre, _, _ = build_preprocessor(Xtr)
    clf, param_log = make_model(args.model, args, cfg)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print("Run ID:", run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())

        # Params
        mlflow.log_params(param_log)

        # Fit
        pipe.fit(Xtr, ytr)

        # Metrics
        p_val = pipe.predict_proba(Xva)[:, 1]
        p_tst = pipe.predict_proba(Xte)[:, 1]
        metrics_val: Dict[str, float] = {
            "val_auroc": float(roc_auc_score(yva, p_val)),
            "val_prauc": float(average_precision_score(yva, p_val)),
            "val_brier": float(brier_score_loss(yva, p_val)),
            "val_pos_rate": float(float(np.mean(yva))),
        }
        metrics_tst: Dict[str, float] = {
            "test_auroc": float(roc_auc_score(yte, p_tst)),
            "test_prauc": float(average_precision_score(yte, p_tst)),
            "test_brier": float(brier_score_loss(yte, p_tst)),
            "test_pos_rate": float(float(np.mean(yte))),
        }
        mlflow.log_metrics({**metrics_val, **metrics_tst})

        print("VAL:", json.dumps({k: round(v, 4) for k, v in metrics_val.items()}, indent=2))
        print("TEST:", json.dumps({k: round(v, 4) for k, v in metrics_tst.items()}, indent=2))

        # Plots
        run_dir = reports_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_curves(yva, p_val, "valid", run_dir)
        log_curves(yte, p_tst, "test", run_dir)

        # Persist pipeline (run-specific)
        model_path = artifacts_dir / f"{param_log['model']}_pipeline_{run_id}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")

        # Also persist a stable filename for Docker Option-1
        stable_path = artifacts_dir / "model_pipeline.joblib"
        try:
            joblib.dump(pipe, stable_path)
            print(f"[artifact] Saved convenience copy -> {stable_path}")
        except Exception as e:
            print(f"[artifact] WARNING: could not write stable artifact: {e!r}")

        # Log MLflow Model with signature + example (under artifact_path="model")
        input_example = Xtr.head(3)
        try:
            signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])
        except Exception:
            signature = infer_signature(input_example, pipe.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )
        print("Logged MLflow model under:", mlflow.get_artifact_uri())

        # Optional tags
        mlflow.set_tags({
            "dataset": get_path(cfg, "data", "processed_dir", default="data/processed"),
            "target_col": get_path(cfg, "data", "target_col", default="readmit_30d"),
            "selection_metric": "val_auroc",
            "baseline_family": param_log.get("model")
        })

        # --- Registration + optional stage ---
        if args.register:
            try:
                version = maybe_register_and_stage(
                    run_id=run_id,
                    model_artifact_subpath="model",
                    model_name=model_name,
                    stage=args.stage,
                    archive_existing=True
                )
                print(json.dumps({
                    "registered_model": model_name,
                    "version": int(version),
                    "stage": args.stage or "None"
                }, indent=2))
            except Exception as e:
                print("[registry] Registration failed:", repr(e))
                print("           If you're using a local file store (./mlruns),")
                print("           start an MLflow server with a DB backend, then re-run with --register.")
        else:
            print("[registry] Skipped (no --register).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None, help="Override experiment name (else env or config).")
    parser.add_argument("--model-name", default=None, help="Registered Model name to use if --register is set.")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (e.g., http://127.0.0.1:5000)")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--model", choices=["lr", "rf"], required=True)
    # LR
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--C", type=float, default=None)
    # RF
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)

    # Registry options (mirror train_mlflow.py)
    parser.add_argument("--register", action="store_true", help="Register the logged model in MLflow Model Registry.")
    parser.add_argument("--stage", choices=["Staging", "Production"], default=None,
                        help="If set with --register, transition new version to this stage.")

    args = parser.parse_args()
    main(args)
