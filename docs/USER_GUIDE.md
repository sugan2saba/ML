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
# Creates data/processed/{train,valid,test}parquet files 

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

To register a model to MLFlow use paramerter --register
Promoting a model
After training, open MLflow UI → Models → MediWatchReadmit.
Choose the best run → register → promote to Staging or Production. examples

python -m scripts.train_baseline_mlflow --model rf --register 
python -m scripts.train_baseline_mlflow --model lr --max-iter 2000 --C 0.5 --register 
python -m scripts.train_mlflow  --register --model-name HBO
python -m scripts.train_mlflow --experiment MediWatch-Readmit --run-name hgb_v1 --register-name MediWatchReadmit



4. Model Registry
Promote best model:
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m scripts.promote_model --model-name MediWatchReadmit --stage Production


5. API

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/sugan2saba/ml:latest \
  --file docker/Dockerfile \
  --push \
  .



# from repo root on your Mac

docker pull ghcr.io/sugan2saba/ml:latest
docker run --rm -p 8000:8000 -e DJANGO_SECRET_KEY=dev -e DEBUG=1 ghcr.io/sugan2saba/ml:latest
#done
#### to run the repo from windows: 

 #start the docker application
 docker version
 docker compose version
 
 git clone 
 
 cd mltest
 
 docker pull ghcr.io/mlflow/mlflow:latest
 docker pull ghcr.io/sugan2saba/ml:latest
 
 docker compose up -d
 start http://localhost:5000
 start http://127.0.0.1:8000/health
 #verify mlflow by running http://localhost:5000 & http://127.0.0.1:8000/health in browser 
 
 #in power shell run the following commands to run some training model
 
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 pip install --upgrade pip
 pip install mlflow scikit-learn
 $env:MLFLOW_TRACKING_URI="http://localhost:5000"
 python .\train_register.py
 deactivate
 
 #the training model Promoted HGB latest version should be staged to Production
 
 docker compose restart api
 docker compose logs -f api
 
 
 start http://127.0.0.1:8000/health
 start http://127.0.0.1:8000/static/rest_framework/css/default.css
 
 #to clean everything
 
 docker compose down --remove-orphans
 docker volume ls
 docker volume rm mltest_mlflow_db mltest_mlflow_artifacts
 docker network prune -f

#### end windows run

See README.md for exact Docker run commands.


6. Monitoring
drift report, this has to be a GitHub workflow setup , runs on certain time and generate the report, Interpreting Drift Reports (Evidently) A nightly GitHub Actions job runs scripts/monitor_evidently.py over logs/predictions.csv and uploads: evidently_report.html (visual dashboard) evidently_report.json (machine-parsed) Where to view GitHub → your repo → Actions → Evidently Drift Monitoring → select a run Download Artifacts → evidently-report → open the HTML locally do I have to run anything manually to prepare logs/predictions.csv 

for testing purpose I have gerenarted logs/predictions.csv using script scripts/create_prediction.py


# Overwrite the file with 200 rows
python scripts/create_prediction.py --n-rows 200 --mode overwrite

# Append 60 rows stamped as "now" to a custom path
python scripts/create_prediction.py --n-rows 60 --out logs/predictions.csv --mode append

# (Optional) Generate "reference" rows 7 days ago (if your monitor uses time splits)
python scripts/create_prediction.py --n-rows 150 --mode overwrite --days-ago 7

Run drift detection:
python -m scripts.monitor_evidently --log-csv logs/predictions.csv --out-dir reports/monitoring --reference-is-recent

python -m scripts.monitor_evidently --reference-is-recent
HTML report in reports/monitoring/evidently_report.html.

---

9.Airflow
Ensure Airflow is running:

git close 
cd orchestration/airflow
docker compose up -d
Open http://localhost:8080 (airflow/airflow).
In the Airflow UI, run the DAG train_and_hpo.
orchestration/airflow/dags/train_and_hpo.py
Tasks:
prepare_data → ray_tune_hpo → train_best_and_register
Output:
Trials & final model logged to MLflow
Model registered as a new version (e.g., v5) of MediWatchReadmit
Promote the new version to Staging or Production (Section 1).
Airflow UI (host): http://127.0.0.1:8080

8. `docs/OPS_PLAYBOOK.md` (for operations / DevOps)

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
