# MLOps Assignment 01 — Heart Disease Classification

This repository implements the MLOps assignment using the UCI Heart Disease dataset.

## Assignment coverage

1. Data acquisition and EDA
2. Feature engineering and two classification models
3. MLflow experiment tracking
4. Reproducible model packaging
5. Pytest + GitHub Actions CI/CD
6. FastAPI + Docker serving
7. Kubernetes/Minikube deployment
8. API logging + Prometheus metrics
9. Documentation/report

## Quick start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

```bash
python src/download_data.py
```

The script downloads the UCI Heart Disease Cleveland data and stores it in `data/raw/heart_disease.csv`.

### 3. Train models

```bash
python src/train.py
```

The training script:
- cleans the data
- builds preprocessing pipelines
- evaluates Logistic Regression and Random Forest
- performs cross-validation
- tunes Random Forest
- logs parameters, metrics, plots and models to MLflow
- saves the selected production pipeline to `models/heart_disease_model.joblib`

### 4. Start MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://127.0.0.1:5000.

### 5. Start API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Health:
```bash
curl http://127.0.0.1:8000/health
```

Prediction:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### 6. Run tests and lint

```bash
pytest -q
ruff check .
```

### 7. Docker

```bash
docker build -t heart-disease-api .
docker run --rm -p 8000:8000 heart-disease-api
```

### 8. Kubernetes / Minikube

```bash
minikube start
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get pods
kubectl get service
```

For Minikube:
```bash
minikube service heart-disease-api --url
```

## Important

Run `python src/train.py` before starting the API so that `models/heart_disease_model.joblib` exists.

## Suggested screenshots

Capture:
- EDA class distribution
- correlation heatmap
- MLflow experiment runs
- test/lint GitHub Actions success
- Docker build and running container
- `/health` and `/predict`
- Kubernetes pods/services
- Prometheus metrics
