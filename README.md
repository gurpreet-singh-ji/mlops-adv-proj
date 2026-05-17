# 🚀 End-to-End MLOps Pipeline with CI/CD, Docker, EKS & Monitoring

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.15.0-orange?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.53.0-945DD6?logo=dvc)](https://dvc.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-black?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![AWS EKS](https://img.shields.io/badge/AWS-EKS-FF9900?logo=amazonaws)](https://aws.amazon.com/eks/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-blue)](https://dagshub.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana)](https://grafana.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade MLOps project demonstrating the full lifecycle of a machine learning system — from raw data ingestion to model deployment on Kubernetes, with automated CI/CD, containerization, and real-time monitoring via Prometheus and Grafana.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline (DVC)](#-ml-pipeline-dvc)
- [Getting Started](#-getting-started)
- [Running the Pipeline](#-running-the-pipeline)
- [Docker](#-docker)
- [CI/CD with GitHub Actions](#-cicd-with-github-actions)
- [Kubernetes Deployment (EKS)](#-kubernetes-deployment-eks)
- [Monitoring with Prometheus & Grafana](#-monitoring-with-prometheus--grafana)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

This project implements a complete, **production-ready MLOps pipeline** for a text classification task. It goes far beyond training a model — it handles:

- **Reproducible pipelines** with DVC and versioned data on AWS S3
- **Experiment tracking** with MLflow hosted on DagsHub
- **Automated model registration** when evaluation thresholds are met
- **Containerized serving** with a Flask API and multi-stage Docker builds
- **Automated CI/CD** via GitHub Actions — test, build, push to ECR, deploy to EKS
- **Real-time monitoring** with Prometheus scraping metrics from the Flask app and Grafana dashboards for visualization

This project is designed to reflect how ML systems are built, shipped, and operated in real engineering teams.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data & Versioning                        │
│   Raw Data → DVC Pipeline → AWS S3 (remote storage)            │
│   Params tracked in params.yaml | Pipeline defined in dvc.yaml  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    ML Pipeline (6 stages)                        │
│  Data Ingestion → Preprocessing → Feature Engineering →         │
│  Model Building → Model Evaluation → Model Registration         │
│                   (MLflow + DagsHub tracking)                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     Flask API (app/)                            │
│         Serves model predictions over HTTP on port 5000         │
│         Exposes /metrics endpoint for Prometheus scraping       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              CI/CD — GitHub Actions                             │
│  Push → Run Tests → Build Docker Image → Push to AWS ECR →     │
│  Deploy to AWS EKS (Kubernetes)                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              Monitoring Stack                                   │
│  Prometheus (EC2) ← scrapes Flask /metrics                     │
│  Grafana (EC2) ← visualizes Prometheus data                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **ML / NLP** | scikit-learn, NLTK, NumPy, pandas |
| **Pipeline Versioning** | DVC 3.53.0 |
| **Experiment Tracking** | MLflow 2.15.0, DagsHub |
| **Remote Storage** | AWS S3, DVC S3 plugin |
| **API Serving** | Flask 3.0.3, Gunicorn |
| **Containerization** | Docker (multi-stage build) |
| **CI/CD** | GitHub Actions |
| **Container Registry** | AWS ECR |
| **Orchestration** | AWS EKS (Kubernetes), eksctl, kubectl |
| **Monitoring** | Prometheus, Grafana, prometheus_client |
| **Config Management** | params.yaml, python-dotenv |

---

## 📁 Project Structure

```
mlops-adv-proj/
│
├── .dvc/                        # DVC config and cache metadata
├── .github/
│   └── workflows/
│       └── ci.yaml              # GitHub Actions CI/CD pipeline
│
├── app/                         # Flask serving application
│   └── app.py                   # API with /predict and /metrics endpoints
│
├── docs/                        # Project documentation (Sphinx)
├── k8s/                         # Kubernetes manifests
│   ├── deployment.yaml          # EKS deployment spec
│   └── service.yaml             # LoadBalancer service definition
│
├── models/                      # Serialized model artifacts
│   ├── model.pkl                # Trained classifier
│   └── vectorizer.pkl           # Fitted TF-IDF vectorizer
│
├── notebooks/                   # Exploratory and experiment notebooks
├── references/                  # Data dictionaries and manuals
├── reports/
│   ├── metrics.json             # Evaluation metrics (tracked by DVC)
│   └── experiment_info.json     # MLflow run info for registration
│
├── scripts/                     # Utility and test scripts for CI
├── src/
│   ├── data/
│   │   ├── data_ingestion.py    # Pulls and splits raw data
│   │   └── data_preprocessing.py# Text cleaning and normalization
│   ├── features/
│   │   └── feature_engineering.py # TF-IDF vectorization
│   ├── model/
│   │   ├── model_building.py    # Trains and logs model to MLflow
│   │   ├── model_evaluation.py  # Evaluates and writes metrics.json
│   │   └── register_model.py    # Registers best model in MLflow registry
│   └── logger/                  # Centralized logging setup
│
├── tests/                       # Unit and integration tests
├── dockerfile                   # Multi-stage Docker build
├── dvc.yaml                     # DVC pipeline definition (6 stages)
├── dvc.lock                     # Locked pipeline state
├── params.yaml                  # Configurable pipeline parameters
├── requirements.txt             # Full dependency list
├── Makefile                     # Shortcut commands
├── setup.py                     # Makes src pip-installable
└── tox.ini                      # Tox test configuration
```

---

## 🔄 ML Pipeline (DVC)

The pipeline is defined in `dvc.yaml` and orchestrated via DVC. Each stage has tracked dependencies, outputs, and parameters — enabling full reproducibility.

```
data_ingestion
    └── data_preprocessing
            └── feature_eng
                    └── model_building
                            └── model_evaluation
                                    └── model_registration
```

| Stage | Script | Key Output |
|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | `data/raw/` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/interim/` |
| `feature_eng` | `src/features/feature_engineering.py` | `models/vectorizer.pkl` |
| `model_building` | `src/model/model_building.py` | `models/model.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | `reports/metrics.json` |
| `model_registration` | `src/model/register_model.py` | MLflow Model Registry |

**Pipeline parameters** (tunable in `params.yaml`):

```yaml
data_ingestion:
  test_size: 0.25

feature_engineering:
  max_features: 50
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.10
- Conda (recommended) or virtualenv
- Docker Desktop
- AWS CLI configured (`aws configure`)
- DVC (`pip install dvc[s3]`)

### 1. Clone and set up the environment

```bash
git clone https://github.com/gurpreet-singh-ji/mlops-adv-proj.git
cd mlops-adv-proj

conda create -n atlas python=3.10
conda activate atlas

pip install -r requirements.txt
pip install -e .   # makes src/ importable
```

### 2. Configure environment variables

Create a `.env` file in the root directory:

```env
CAPSTONE_TEST=<your_dagshub_token>
MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/mlops-adv-proj.mlflow
```

### 3. Pull versioned data from S3

```bash
dvc pull
```

---

## 🔁 Running the Pipeline

Run the full DVC pipeline end-to-end:

```bash
dvc repro
```

Check pipeline status (which stages are outdated):

```bash
dvc status
```

Run a specific stage only:

```bash
dvc repro model_evaluation
```

Push data artifacts to S3:

```bash
dvc push
```

---

## 🐳 Docker

The project uses a **multi-stage Docker build** to keep the final image lean:

- **Stage 1 (Builder):** Installs all Python dependencies and downloads NLTK data
- **Stage 2 (Final):** Copies only what's needed — no build tools, minimal footprint

### Build and run locally

```bash
# Build the image
docker build -t capstone-app:latest .

# Run with DagsHub token
docker run -p 8888:5000 \
  -e CAPSTONE_TEST=<your_dagshub_token> \
  capstone-app:latest

# Test the API
curl http://localhost:8888/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Your input text here"}'
```

The API will be available at `http://localhost:8888`.

---

## ⚙️ CI/CD with GitHub Actions

The `.github/workflows/ci.yaml` pipeline runs on every push to `master`:

```
Push to master
    │
    ├─ Run unit tests (pytest)
    │
    ├─ Build Docker image
    │
    ├─ Push image to AWS ECR
    │
    └─ Deploy to AWS EKS (kubectl apply)
```

### Required GitHub Secrets

Set these in your repository → Settings → Secrets and Variables → Actions:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |
| `ECR_REPOSITORY` | ECR repo name (e.g. `capstone-proj`) |
| `CAPSTONE_TEST` | DagsHub authentication token |

---

## ☸️ Kubernetes Deployment (EKS)

### Create EKS cluster

```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 --nodes-min 1 --nodes-max 1 \
  --managed
```

### Update kubeconfig

```bash
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
```

### Verify deployment

```bash
kubectl get pods
kubectl get svc          # get external LoadBalancer IP
kubectl get namespaces
```

### Access the app

```bash
# Get the external IP from the LoadBalancer service
kubectl get svc flask-app-service

# Test the endpoint
curl http://<external-ip>:5000
```

### Teardown (to avoid AWS charges)

```bash
kubectl delete deployment flask-app
kubectl delete service flask-app-service
kubectl delete secret capstone-secret
eksctl delete cluster --name flask-app-cluster --region us-east-1
```

---

## 📊 Monitoring with Prometheus & Grafana

The Flask app exposes a `/metrics` endpoint (via `prometheus_client`) that Prometheus scrapes every 15 seconds.

### Prometheus Setup (EC2 — t3.medium, port 9090)

```bash
# Download and extract
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz
sudo mv prometheus-2.46.0.linux-amd64 /etc/prometheus

# Configure scrape target in /etc/prometheus/prometheus.yml
# target: <flask-app-external-ip>:5000

# Run Prometheus
/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml
```

Access Prometheus UI: `http://<ec2-ip>:9090`

### Grafana Setup (EC2 — t3.medium, port 3000)

```bash
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
sudo apt install ./grafana_10.1.5_amd64.deb -y
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access Grafana UI: `http://<ec2-ip>:3000` (default: admin/admin)

Add Prometheus as a data source: `http://<prometheus-ec2-ip>:9090`

---

## ⚙️ Configuration

All tunable parameters live in `params.yaml`. Modify and re-run `dvc repro` to trigger only the affected downstream stages:

```yaml
data_ingestion:
  test_size: 0.25        # train/test split ratio

feature_engineering:
  max_features: 50       # top N TF-IDF features to use
```

DVC tracks parameter changes and automatically determines which stages need to be re-run.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, descriptive commits
4. Add or update tests in `tests/` as appropriate
5. Run tests: `pytest tests/`
6. Push and open a Pull Request

Please ensure your code follows the existing structure and that `dvc repro` runs cleanly before submitting.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with 🔧 by <a href="https://github.com/gurpreet-singh-ji">Gurpreet Singh</a>
  <br/>
  <sub>MLOps · DVC · MLflow · Docker · AWS EKS · Prometheus · Grafana</sub>
</div>
