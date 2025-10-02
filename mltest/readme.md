#Run this once in windows to run train_register.py
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