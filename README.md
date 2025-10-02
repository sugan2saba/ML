# MediWatch – Patient Readmission Prediction

# MediWatch is a machine learning system to predict patient readmission risk, with full MLOps pipeline support.

## Quickstart

### Prerequisites
- Python 3.10+
- pip / venv
- Docker 

### Setup
```bash
git clone https://github.com/sugan2saba/ML.git
cd mediwatch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw

## 1. Data
- Download dataset: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/brandao/diabetes)

- Place under `data/raw/diabetic_data.csv`

## 2. Preprocessing also notebooks/01_eda.ipynb has some anasysis 
```bash
python -m scripts.make_splits
# Creates data/processed/{train,valid,test}parquet files 

# Train for local run 
python -m scripts.train_mlflow --model rf
python -m scripts.promote_model --name MediWatchReadmit --stage Production

#Train and Register to MLFLOW
python -m scripts.train_baseline_mlflow --model rf --register 
python -m scripts.train_baseline_mlflow --model lr --max-iter 2000 --C 0.5 --register 
python -m scripts.train_mlflow  --register --model-name HBO
python -m scripts.train_mlflow --experiment MediWatch-Readmit --run-name hgb_v1 --register-name MediWatchReadmit

#Run API (Docker Option 1 – baked model)
docker run --rm -p 8000:8000 \
  -e DJANGO_SECRET_KEY=dev -e DEBUG=1 \
  ghcr.io/sugan2saba/ml:latest
# Run API (Docker Option 2 – MLflow model)
docker run --rm -p 8000:8000 \
  -e DJANGO_SECRET_KEY=dev -e DEBUG=1 \
  -e MODEL_SOURCE=mlflow \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=MediWatchReadmit \
  -e MLFLOW_MODEL_STAGE=Production \
  ghcr.io/sugan2saba/ml:latest
### Test
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/schema
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"records":[{"race":"Caucasian","gender":"Female","age":"[60-70)","time_in_hospital":3,"num_lab_procedures":41,"num_medications":12,"number_diagnoses":8,"diabetesMed":"Yes","A1Cresult":"None","max_glu_serum":"None"}]}'
