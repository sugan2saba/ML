OPS_PLAYBOOK.md — MediWatch Operations Runbook
This is the day-to-day playbook for operating the MediWatch system: promoting or rolling back models, interpreting drift reports, forcing a retrain, rebuilding the API image, and fixing static assets in the API.
Assumptions:
MLflow Tracking Server runs locally at http://127.0.0.1:5000 (or reachable host).
Your API image lives at ghcr.io/sugan2saba/ml (change if different).
Two deployment modes:
Option 1: image with baked model (:withmodel tag)
Option 2: image pulls Production model from MLflow Registry at runtime (:latest tag)
0) Quick Reference
Check health: GET http://127.0.0.1:8000/health
Check schema: GET http://127.0.0.1:8000/schema
MLflow UI: http://127.0.0.1:5000
Airflow UI (if using orchestrator): http://localhost:8080 (user/pass: airflow / airflow)
Actions UI (drift report): GitHub → your repo → Actions → Evidently Drift Monitoring → latest run → Artifacts
1) Model Promotion (to Staging/Production)
You can promote via MLflow UI or CLI script.
A) Promote via MLflow UI
Open MLflow UI → Models → MediWatchReadmit.
Click the version you want (e.g., Version 4).
Click Transition → choose Staging or Production → Confirm.
The API in Option 2 will read the model from models:/MediWatchReadmit/Production.
B) Promote via CLI helper
We provide scripts/promote_model.py:
# Mac/Linux
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m scripts.promote_model --name MediWatchReadmit --version 4 --stage Production

# Windows PowerShell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m scripts.promote_model --name MediWatchReadmit --version 4 --stage Production
Tip: Keep previous version in Staging as a quick rollback target.
2) Rollback (to a prior version)
Open MLflow UI → Models → MediWatchReadmit.
Pick the previous stable version (e.g., Version 3).
Transition it to Production (and optionally demote the current one).
Apply rollback to running API
Option 2 (Registry): if the API loads the model on startup, just restart the container(s) to pick up the new Production version.
Option 1 (Baked): you must rebuild the image with the desired model baked in (see Section 5).
3) Interpreting Drift Reports (Evidently)
A nightly GitHub Actions job runs scripts/monitor_evidently.py over logs/predictions.csv and uploads:
evidently_report.html (visual dashboard)
evidently_report.json (machine-parsed)
Where to view
GitHub → your repo → Actions → Evidently Drift Monitoring → select a run
Download Artifacts → evidently-report → open the HTML locally
What to look at
Dataset drift: Is dataset_drift true?
Number of drifted columns: How many features drifted?
Target drift (if enabled): Large change in output distribution (pred/prob).
CI Gate (pass/fail)
The workflow fails if Evidently flags dataset_drift=true (or based on a stricter rule you set).
Tighten the drift gate (optional)
If you want to fail only if ≥ N drifted columns, edit .github/workflows/monitoring.yml and replace the “Fail the job” step with:
- name: Fail the job if drift ≥ 3 columns
  run: |
    python - <<'PY'
    import json, sys, pathlib
    p = pathlib.Path("reports/monitoring/evidently_report.json")
    if not p.exists():
        print("No report found; skipping drift gate.")
        sys.exit(0)
    d = json.loads(p.read_text())
    dataset = None
    for m in d.get("metrics", []):
        if m.get("metric") == "DataDriftPreset":
            dataset = m.get("result", {})
            break
    n = int(dataset.get("number_of_drifted_columns", 0)) if dataset else 0
    print(f"DRIFTED_COLUMNS={n}")
    sys.exit(1 if n >= 3 else 0)
    PY
When drift is detected
Acknowledge the red CI run (drift alert).
Inspect the HTML report to see which features drifted.
Force retrain (Section 4) to adapt the model.
Promote the new model to Staging/Production (Section 1).
Restart API (Option 2) or rebuild & redeploy (Option 1).
4) Force Retrain
You can retrain via Airflow DAG or manual scripts.
A) Retrain via Airflow (recommended for reproducibility)
Ensure Airflow is running:
cd orchestration/airflow
docker compose up -d
Open http://localhost:8080 (airflow/airflow).
In the Airflow UI, run the DAG train_and_hpo.
Tasks:
prepare_data → ray_tune_hpo → train_best_and_register
Output:
Trials & final model logged to MLflow
Model registered as a new version (e.g., v5) of MediWatchReadmit
Promote the new version to Staging or Production (Section 1).
B) Manual retrain (quickest path)
From your repo root:
# point training to your MLflow server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# run your training (examples)
python -m scripts.train_mlflow --model hgb
# or
python -m scripts.train_baseline_mlflow --model rf

# promote best version (after checking MLflow UI)
python -m scripts.promote_model --name MediWatchReadmit --version <N> --stage Production
If you use Option 1 (baked), continue to Section 5 to rebuild the API image with the new model.
5) Update Docker Image (build & push)
Two patterns:
A) Option 2 (no model baked; loads from Registry)
Rebuild only when code/deps change (model updates don’t require rebuild).
# Build (runtime Dockerfile, no baked model)
docker build -t ghcr.io/sugan2saba/ml:latest -f docker/Dockerfile .
docker push ghcr.io/sugan2saba/ml:latest
Run (pulls Production at startup):
docker run --rm -p 8000:8000 \
  -e DJANGO_SECRET_KEY=dev -e DEBUG=1 \
  -e MODEL_SOURCE=mlflow \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=MediWatchReadmit \
  -e MLFLOW_MODEL_STAGE=Production \
  ghcr.io/sugan2saba/ml:latest
Note: Use host.docker.internal inside Docker on Mac/Windows to reach host MLflow.
B) Option 1 (baked model inside image)
Ensure the best model artifact exists locally:
artifacts/model_pipeline.joblib
(Copy/rename your chosen model file to this name.)
Build a multi-arch image and push:
# login to GHCR once
echo '<YOUR_GITHUB_PAT>' | docker login ghcr.io -u sugan2saba --password-stdin

docker buildx create --use >/dev/null 2>&1 || true
docker buildx inspect --bootstrap

docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/sugan2saba/ml:withmodel \
  --build-arg MODEL_SRC=artifacts/model_pipeline.joblib \
  -f docker/Dockerfile.withmodel \
  . --push
Run (no mounts needed):
docker run --rm -p 8000:8000 -e DJANGO_SECRET_KEY=dev -e DEBUG=1 ghcr.io/sugan2saba/ml:withmodel
Tip: Tag releases: :v0.4.0, :2025-09-29, etc., alongside :latest/:withmodel.
6) Static Files Missing in API (DRF CSS/JS 404)
If you see logs like:
Not Found: /static/rest_framework/css/...
Collect and serve static assets with WhiteNoise.
A) Requirements
Add to requirements.txt:
whitenoise
B) settings.py changes (Django)
INSTALLED_APPS = [
  "django.contrib.staticfiles",
  "rest_framework",
  # ...
]

MIDDLEWARE = [
  "django.middleware.security.SecurityMiddleware",
  "whitenoise.middleware.WhiteNoiseMiddleware",  # right after SecurityMiddleware
  # ...
]

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
C) Dockerfile: run collectstatic at build time
In docker/Dockerfile (and Dockerfile.withmodel), after copying code:
WORKDIR /app/api
RUN python manage.py collectstatic --noinput
Rebuild & run the API; the browsable DRF UI should load CSS/JS correctly.
7) Troubleshooting
API won’t start (Option 2)
Symptom: “Cannot load model from MLflow.”
Checks:
Container ENV: MODEL_SOURCE=mlflow, MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE.
MLflow server reachable from container: use http://host.docker.internal:5000 on Mac/Windows.
At least one Production model version exists.
API won’t start (Option 1)
Symptom: FileNotFoundError: MODEL_PATH not found
Checks:
Was the model baked into the image? Ensure COPY ${MODEL_SRC} /models/model_pipeline.joblib ran during build.
If running runtime image: ensure you mount the host file to the container path in MODEL_PATH.
Predictions file missing
Symptom: logs/predictions.csv not found.
Fix:
Ensure PRED_LOG_DIR is set (default logs/).
For Docker, mount a host logs folder to /logs.
Make at least one /predict call to generate the file.
Drift report empty / CI skips
Symptom: “Not enough rows to run Evidently.”
Fix: Generate more predictions (≥ --min-rows, default 50) or shorten windows (increase coverage).
GHCR push denied
Fix: docker login ghcr.io with a PAT with write:packages.
8) Operational Play Patterns
Small change to model config: run Airflow DAG with limited trials → register → promote to Staging → test API → promote to Production → restart API (Option 2) / rebuild (Option 1).
Drift alert: inspect report → retrain → promote → restart/redeploy.
Rollback: set previous version to Production in MLflow → restart API (Option 2) / rebuild (Option 1).
9) Commands Cheat Sheet
# MLflow server (local)
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

# Promote model (CLI)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m scripts.promote_model --name MediWatchReadmit --version 5 --stage Production

# Airflow up
cd orchestration/airflow
docker compose up -d

# Rebuild API (runtime)
docker build -t ghcr.io/sugan2saba/ml:latest -f docker/Dockerfile .
docker push ghcr.io/sugan2saba/ml:latest

# Rebuild API (with model baked)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/sugan2saba/ml:withmodel \
  --build-arg MODEL_SRC=artifacts/model_pipeline.joblib \
  -f docker/Dockerfile.withmodel . --push

# Run API (Option 2 / Registry)
docker run --rm -p 8000:8000 \
  -e DJANGO_SECRET_KEY=dev -e DEBUG=1 \
  -e MODEL_SOURCE=mlflow \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=MediWatchReadmit \
  -e MLFLOW_MODEL_STAGE=Production \
  ghcr.io/sugan2saba/ml:latest
That’s it. This playbook covers the core ops motions you’ll use day-to-day: promote/rollback, drift triage, retrain, rebuild, and asset fixes. If you want, we can add small screenshots of the MLflow/Actions UIs later for your final submission.