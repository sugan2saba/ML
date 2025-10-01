# scripts/train_mlflow.py
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

from dotenv import load_dotenv  # NEW: load .env

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

# --- your project imports ---
from mediwatch.preprocess import build_preprocessor
from mediwatch.config import load_config, get_path
from mediwatch.config_validate import validate_config_and_data
from mediwatch.align import align_like_train


# ----------------------------
# Helpers
# ----------------------------
def load_splits(cfg: dict):
    data_dir = Path(get_path(cfg, "data", "processed_dir", default="data/processed"))
    tgt = get_path(cfg, "data", "target_col", default="readmit_30d")
    Xtr = pd.read_parquet(data_dir / get_path(cfg, "data", "x_train", default="X_train.parquet"))
    ytr = pd.read_parquet(data_dir / get_path(cfg, "data", "y_train", default="y_train.parquet"))[tgt].astype(int)
    Xva = pd.read_parquet(data_dir / get_path(cfg, "data", "x_valid", default="X_valid.parquet"))
    yva = pd.read_parquet(data_dir / get_path(cfg, "data", "y_valid", default="y_valid.parquet"))[tgt].astype(int)
    Xte = pd.read_parquet(data_dir / get_path(cfg, "data", "x_test",  default="X_test.parquet"))
    yte = pd.read_parquet(data_dir / get_path(cfg, "data", "y_test",  default="y_test.parquet"))[tgt].astype(int)
    return Xtr, ytr, Xva, yva, Xte, yte


def log_curves(y_true, y_prob, tag: str, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--")
    ax[0].set_title(f"ROC ({tag})")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    ax[1].plot(rec, prec, label=f"PR-AUC={prauc:.3f}")
    ax[1].set_title(f"PR ({tag})")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()

    fig.tight_layout()
    out = out_dir / f"curves_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path=f"plots/{tag}")


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
# Main training entrypoint
# ----------------------------
def main(args):
    # --- Load .env so CLI runs pick up MLFLOW_TRACKING_URI and friends ---
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")

    # --- MLflow wiring ---
    # Priority: CLI --tracking-uri > env MLFLOW_TRACKING_URI > fallback ./mlruns
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri.strip())
        print(f"[mlflow] Using CLI tracking URI: {mlflow.get_tracking_uri()} (Option B)")
    else:
        env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if env_uri:
            mlflow.set_tracking_uri(env_uri)
            print(f"[mlflow] Using env tracking URI: {env_uri} (Option B)")
        else:
            # Safe fallback for local runs (no server needed)
            os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"
            mlflow.set_tracking_uri("./mlruns")
            print("[mlflow] No tracking URI set; defaulting to ./mlruns (Option A)")
            print("         NOTE: Model Registry requires a DB-backed server (e.g., SQLite).")
            print("         Start one, e.g.:")
            print("           mlflow server --host 127.0.0.1 --port 5000 \\")
            print("             --backend-store-uri sqlite:///mlflow.db \\")
            print("             --artifacts-destination ./mlartifacts")

    # Experiment / model names (env overrides -> CLI -> config -> defaults)
    # Load config once here so it's available below too
    cfg = load_config(args.config)

    experiment = (
        args.experiment
        or os.getenv("MLFLOW_EXPERIMENT")
        or get_path(cfg, "mlflow", "experiment", default="MediWatch-Readmit")
    )
    mlflow.set_experiment(experiment)

    model_name = (
        args.model_name
        or os.getenv("MLFLOW_MODEL_NAME")
        or "MediWatchReadmit"
    )

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", experiment)
    print("Registered Model:", model_name)

    # --- Config & validation ---
    result = validate_config_and_data(cfg, strict_columns=False)
    for m in result.msgs:
        print("[validate]", m)
    result.raise_if_failed()

    # Output dirs
    reports_dir = Path(get_path(cfg, "output", "reports_dir", default="reports")); reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(get_path(cfg, "output", "artifacts_dir", default="artifacts")); artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    Xtr, ytr, Xva, yva, Xte, yte = load_splits(cfg)

    # auto-align (valid/test to train schema)
    Xva, miss_v, extra_v, ch_v = align_like_train(Xtr, Xva)
    Xte, miss_t, extra_t, ch_t = align_like_train(Xtr, Xte)
    if ch_v:
        print(f"[auto-align] valid: +{len(miss_v)} missing, -{len(extra_v)} extra")
    if ch_t:
        print(f"[auto-align] test:  +{len(miss_t)} missing, -{len(extra_t)} extra")

    # --- Hyperparams (CLI overrides cfg) ---
    lr = args.lr if args.lr is not None else get_path(cfg, "hgb", "learning_rate", default=0.05)
    n_estimators = args.n_estimators if args.n_estimators is not None else get_path(cfg, "hgb", "n_estimators", default=500)
    l2 = get_path(cfg, "hgb", "l2_regularization", default=0.0)

    # --- Pipeline ---
    pre, _, _ = build_preprocessor(Xtr)
    clf = HGB(
        learning_rate=lr,
        max_iter=n_estimators,
        l2_regularization=l2,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # --- Train & Log ---
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print("Run ID:", run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())

        # Params
        mlflow.log_params({
            "model_family": "HistGradientBoosting",
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "l2_regularization": l2,
            "random_state": 42
        })

        # Fit
        pipe.fit(Xtr, ytr)

        # Metrics
        p_val = pipe.predict_proba(Xva)[:, 1]
        p_tst = pipe.predict_proba(Xte)[:, 1]
        metrics_val = {
            "val_auroc": float(roc_auc_score(yva, p_val)),
            "val_prauc": float(average_precision_score(yva, p_val)),
            "val_brier": float(brier_score_loss(yva, p_val)),
            "val_pos_rate": float(yva.mean())
        }
        metrics_tst = {
            "test_auroc": float(roc_auc_score(yte, p_tst)),
            "test_prauc": float(average_precision_score(yte, p_tst)),
            "test_brier": float(brier_score_loss(yte, p_tst)),
            "test_pos_rate": float(yte.mean())
        }
        mlflow.log_metrics({**metrics_val, **metrics_tst})

        print("VAL:", json.dumps({k: round(v, 4) for k, v in metrics_val.items()}, indent=2))
        print("TEST:", json.dumps({k: round(v, 4) for k, v in metrics_tst.items()}, indent=2))

        # Plots
        run_dir = reports_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_curves(yva, p_val, "valid", run_dir)
        log_curves(yte, p_tst, "test", run_dir)

        # Persist sklearn pipeline as a plain artifact (nice to have)
        model_path = artifacts_dir / f"HGB_pipeline_{run_id}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="artifacts")

        # Also keep a stable filename for Docker Option-1 builds
        stable_path = artifacts_dir / "model_pipeline.joblib"
        try:
            joblib.dump(pipe, stable_path)
            print(f"[artifact] Also wrote {stable_path} (convenience copy)")
        except Exception as e:
            print(f"[artifact] WARNING: could not write stable artifact: {e!r}")

        # MLflow Model: add signature + input_example for safer serving/loader
        input_example = Xtr.head(3)
        try:
            signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])
        except Exception:
            # Some models may not provide predict_proba; fallback to predict
            signature = infer_signature(input_example, pipe.predict(input_example))

        # Always log the model under artifact_path="model" for consistent registration URI
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        print("Logged MLflow model under:", mlflow.get_artifact_uri())

        # Optional tags (useful for auto-promotion rules later)
        mlflow.set_tags({
            "dataset": get_path(cfg, "data", "processed_dir", default="data/processed"),
            "target_col": get_path(cfg, "data", "target_col", default="readmit_30d"),
            "selection_metric": "val_auroc",
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
    parser.add_argument("--model-name", default=None, help="Override Registered Model name (else env or default).")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (e.g., http://127.0.0.1:5000)")
    parser.add_argument("--run-name", default="hgb_train")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)

    # Registry options
    parser.add_argument("--register", action="store_true", help="Register the logged model in MLflow Model Registry.")
    parser.add_argument("--stage", choices=["Staging", "Production"], default=None,
                        help="If set with --register, transition new version to this stage.")

    args = parser.parse_args()
    main(args)
