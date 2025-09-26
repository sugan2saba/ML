# scripts/hpo_optuna.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

from mediwatch.preprocess import build_preprocessor

DATA_DIR = Path("data/processed")

def load_splits():
    Xtr = pd.read_parquet(DATA_DIR / "X_train.parquet")
    ytr = pd.read_parquet(DATA_DIR / "y_train.parquet")["readmit_30d"].astype(int)
    Xva = pd.read_parquet(DATA_DIR / "X_valid.parquet")
    yva = pd.read_parquet(DATA_DIR / "y_valid.parquet")["readmit_30d"].astype(int)
    return Xtr, ytr, Xva, yva

def objective(trial, experiment: str):
    Xtr, ytr, Xva, yva = load_splits()
    pre, _, _ = build_preprocessor(Xtr)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0)
    }

    clf = HGB(random_state=42, **params)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    with mlflow.start_run(run_name=f"hpo_trial_{trial.number}", nested=True):
        mlflow.log_params({"model":"HGB", **params})
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:,1]
        auroc = float(roc_auc_score(yva, p))
        prauc = float(average_precision_score(yva, p))
        brier = float(brier_score_loss(yva, p))
        mlflow.log_metrics({"val_auroc": auroc, "val_prauc": prauc, "val_brier": brier})
        # Optuna optimizes AUROC by default
        trial.set_user_attr("val_prauc", prauc)
        trial.set_user_attr("val_brier", brier)
        return auroc

def main(args):
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="HPO_Optuna"):
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: objective(t, args.experiment), n_trials=args.n_trials, show_progress_bar=True)

        best = study.best_trial
        mlflow.log_params({f"best_{k}": v for k,v in best.params.items()})
        mlflow.log_metrics({
            "best_val_auroc": best.value,
            "best_val_prauc": best.user_attrs.get("val_prauc", float("nan")),
            "best_val_brier": best.user_attrs.get("val_brier", float("nan"))
        })

        print("\nBest trial:")
        print("  AUROC:", best.value)
        print("  Params:", json.dumps(best.params, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="MediWatch-Readmit")
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args()
    main(args)
