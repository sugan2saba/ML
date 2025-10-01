<!-- This is for new users of the repo (students, teammates). It should explain the full workflow from data → training → API. -->
# MediWatch User Guide

This guide explains how to run the project end-to-end.

## 1. Data
- Download dataset: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/brandao/diabetes)
mkdir -p data/raw
- Place under `data/raw/diabetic_data.csv`

## 2. Preprocessing
```bash
python -m scripts.make_splits
# Creates data/processed/{train,valid,test}.csv

3. Training

Note: to start mlflow local experiment (local mode, no server)
If you skip running a server, MLflow defaults to ./mlruns.
Launch the UI with:
mlflow ui --backend-store-uri ./mlruns
This is fine for solo experiments, but Model Registry will not be available.


recommended: To start mlflow (recommended: MLflow server + registry)
Start MLflow tracking server (in one terminal):

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000

This creates:
mlflow.db (experiment metadata, model registry)
mlartifacts/ (artifacts and model files)
Check .env → ensure:
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Now train models:
Baseline (LogReg / RandomForest):
python -m scripts.train_baseline_mlflow --model rf
Advanced (HGB + Optuna HPO):
python -m scripts.train_mlflow --model hgb
All runs, metrics, and artifacts are logged to the MLflow server at http://127.0.0.1:5000.
Promoting a model
After training, open MLflow UI → Models → MediWatchReadmit.
Choose the best run → register → promote to Staging or Production.


4. Model Registry
Promote best model:
python -m scripts.promote_model --name MediWatchReadmit --stage Production
5. API
Option 1: Run Docker image that includes the model artifact.
Option 2: Run Docker image that pulls Production model from MLflow.
See README.md for exact Docker run commands.
6. Monitoring
Run drift detection:
python -m scripts.monitor_evidently --reference-is-recent
HTML report in reports/monitoring/evidently_report.html.

---

# 3. `docs/OPS_PLAYBOOK.md` (for operations / DevOps)

This is for *operators / maintainers* — how to promote, roll back, and interpret monitoring.

```markdown
# MediWatch Ops Playbook

## Promoting a Model
1. Train and log model with MLflow.
2. In MLflow UI:
   - Go to **Models → MediWatchReadmit**
   - Select best run → "Register Model"
   - Promote to **Production**
3. Verify with:
```bash
mlflow models serve -m "models:/MediWatchReadmit/Production"
Rolling Back
In MLflow UI, change stage back to Staging or Archived.
Promote previous version back to Production.
Restart API container (Option 2 will always pull the latest Production).
Monitoring & Drift
Predictions are logged to logs/predictions.csv
GitHub Actions runs scripts/monitor_evidently.py nightly
Download report from Actions → Artifacts → evidently-report
Check:
dataset_drift: true/false
number_of_drifted_columns
Gate: CI will fail if drift detected (--fail-on-drift).
