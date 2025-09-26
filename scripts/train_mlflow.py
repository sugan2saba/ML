# scripts/train_mlflow.py
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os 
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

from mediwatch.preprocess import build_preprocessor
from mediwatch.config import load_config, get_path
from mediwatch.config_validate import validate_config_and_data
from mediwatch.align import align_like_train

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
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title(f"ROC ({tag})"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    ax[1].plot(rec, prec, label=f"PR-AUC={praux:.3f}" if (praux := prauc) else f"PR-AUC={prauc:.3f}")
    ax[1].set_title(f"PR ({tag})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()

    fig.tight_layout()
    out = out_dir / f"curves_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    mlflow.log_artifact(str(out), artifact_path=f"plots/{tag}")

def main(args):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"
        print("[mlflow] No tracking URI set, defaulting to ./mlruns")
    print("Tracking URI:", mlflow.get_tracking_uri())
    cfg = load_config(args.config)
    # validate (soft column check; files must exist)
    result = validate_config_and_data(cfg, strict_columns=False)
    for m in result.msgs: print("[validate]", m)
    result.raise_if_failed()

    experiment = args.experiment or get_path(cfg, "mlflow", "experiment", default="MediWatch-Readmit")
    track_uri = get_path(cfg, "mlflow", "tracking_uri")
    if track_uri: mlflow.set_tracking_uri(track_uri)
    mlflow.set_experiment(experiment)

    reports_dir = Path(get_path(cfg, "output", "reports_dir", default="reports")); reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(get_path(cfg, "output", "artifacts_dir", default="artifacts")); artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Tracking URI:", mlflow.get_tracking_uri())

    Xtr, ytr, Xva, yva, Xte, yte = load_splits(cfg)

    # auto-align
    Xva, miss_v, extra_v, ch_v = align_like_train(Xtr, Xva)
    Xte, miss_t, extra_t, ch_t = align_like_train(Xtr, Xte)
    if ch_v: print(f"[auto-align] valid: +{len(miss_v)} missing, -{len(extra_v)} extra")
    if ch_t: print(f"[auto-align] test:  +{len(miss_t)} missing, -{len(extra_t)} extra")

    # hyperparams: CLI overrides config
    lr = args.lr if args.lr is not None else get_path(cfg, "hgb", "learning_rate", default=0.05)
    n_estimators = args.n_estimators if args.n_estimators is not None else get_path(cfg, "hgb", "n_estimators", default=500)
    l2 = get_path(cfg, "hgb", "l2_regularization", default=0.0)

    pre, _, _ = build_preprocessor(Xtr)
    clf = HGB(
        learning_rate=lr,
        max_iter=n_estimators,
        l2_regularization=l2,
        random_state=42
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print("Run ID:", run_id)
        print("Artifact URI:", mlflow.get_artifact_uri())

        mlflow.log_params({
            "model": "HistGradientBoosting",
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "l2_regularization": l2
        })

        # fit
        pipe.fit(Xtr, ytr)

        # metrics
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
        for k, v in {**metrics_val, **metrics_tst}.items():
            mlflow.log_metric(k, v)

        print("VAL:", json.dumps({k: round(v, 4) for k, v in metrics_val.items()}, indent=2))
        print("TEST:", json.dumps({k: round(v, 4) for k, v in metrics_tst.items()}, indent=2))

        # plots
        run_dir = reports_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_curves(yva, p_val, "valid", run_dir)
        log_curves(yte, p_tst, "test", run_dir)

        # persist pipeline
        model_path = artifacts_dir / f"HGB_pipeline_{run_id}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        # log MLflow model with signature
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="MediWatchReadmit_HGB",
            input_example=Xtr.head(3)
        )

        print("Logged artifacts under:", mlflow.get_artifact_uri())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--run-name", default="hgb_train")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()
    main(args)
