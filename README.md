# AI Fraud Detection & Risk Intelligence Platform

> Production-oriented fraud detection platform combining **classical Machine Learning, LSTM-based behavioral sequence modeling, and Transformer-based temporal modeling**, exposed through a **Flask REST API**.

**Version:** `1.0.0`
**Status:** 🚧 In Development
**Domain:** Financial Services / Payments / Fraud Detection
**Primary API:** Flask
**Language:** Python

---

## 📌 Overview

The **AI Fraud Detection & Risk Intelligence Platform** is an end-to-end machine learning system designed to identify potentially fraudulent financial transactions.

The platform combines three complementary modeling approaches:

1. **XGBoost** — learns patterns from the current transaction and engineered aggregate features.
2. **LSTM** — learns sequential behavioral patterns from a customer's transaction history.
3. **Transformer** — uses self-attention to model relationships across transaction sequences.

The predictions from these models are passed to a configurable **Risk Engine**, which produces a final risk score and business decision.

```text
                         Transaction
                              │
                              ▼
                       Flask REST API
                              │
                              ▼
                    Request Validation
                              │
                              ▼
                     Feature Pipeline
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          XGBoost            LSTM         Transformer
       Transaction ML    Sequence Model   Attention Model
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                       Ensemble / Risk
                           Engine
                              │
                              ▼
                         Risk Score
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
               APPROVE      REVIEW      BLOCK
```

The goal is not simply to achieve high model accuracy.

The project focuses on the complete ML lifecycle:

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Baseline ML
 ↓
XGBoost
 ↓
Sequence Construction
 ↓
LSTM
 ↓
Transformer
 ↓
Model Evaluation
 ↓
Ensemble
 ↓
Flask API
 ↓
Testing
 ↓
Docker
 ↓
MLflow
 ↓
Monitoring
 ↓
CI/CD
 ↓
Deployment
```

---

# 🎯 Problem Statement

Financial transaction systems need to identify suspicious transactions while minimizing both:

* **False negatives** — fraudulent transactions classified as legitimate.
* **False positives** — legitimate transactions incorrectly flagged as fraud.

Fraud detection is therefore treated as an **imbalanced classification problem with a business decision layer**, rather than a simple accuracy optimization problem.

The system will produce:

```text
Fraud Probability
       ↓
Risk Level
       ↓
Business Decision
```

For example:

```json
{
  "transaction_id": "TXN10001",
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "decision": "REVIEW"
}
```

---

# 🏗️ Architecture

## High-Level Architecture

```text
                         ┌─────────────────────┐
                         │   Client / System   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Flask API       │
                         │   REST Endpoints    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Request Validation  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Feature Pipeline   │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │    XGBoost     │ │      LSTM      │ │  Transformer   │
        │ Current Txn    │ │   Behavioral   │ │   Temporal     │
        │ Features       │ │   Sequence     │ │   Attention    │
        └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │     Risk Engine     │
                         │ Ensemble + Policy   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Final Decision    │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
               APPROVE            REVIEW             BLOCK
```

---

# 🧠 Model Architecture

## 1. XGBoost

XGBoost is used as the primary structured-data model.

It receives features such as:

```text
Transaction amount
Merchant category
Payment method
Transaction hour
Day of week
Customer transaction count
Customer average transaction amount
Transaction velocity
New device indicator
Location-related features
Historical transaction statistics
```

Conceptually:

```text
Current Transaction
       │
       ▼
Feature Engineering
       │
       ▼
    XGBoost
       │
       ▼
Fraud Probability
```

Example:

```text
XGBoost Prediction = 0.72
```

---

## 2. LSTM

The LSTM model analyzes a customer's transaction history as a sequence.

Example:

```text
Customer C1029

T1 → ₹500
T2 → ₹700
T3 → ₹450
T4 → ₹800
T5 → ₹25,000
```

The sequence is transformed into model-ready tensors:

```text
Transaction History
        │
        ▼
Sequence Builder
        │
        ▼
LSTM
        │
        ▼
Hidden Representation
        │
        ▼
Fraud Probability
```

Example:

```text
LSTM Prediction = 0.84
```

The purpose of the LSTM is to capture **temporal behavioral patterns** that may not be represented by a single transaction.

---

## 3. Transformer

The Transformer model uses self-attention to model relationships across transaction sequences.

```text
Transaction Sequence
        │
        ▼
Embedding
        │
        ▼
Positional Information
        │
        ▼
Self-Attention
        │
        ▼
Feed Forward Network
        │
        ▼
Transformer Representation
        │
        ▼
Fraud Probability
```

Example:

```text
Transformer Prediction = 0.91
```

The Transformer provides an alternative sequence modeling approach to LSTM and allows us to study how attention-based architectures perform on transactional behavior.

---

# 🔀 Ensemble & Risk Engine

The three model predictions are combined through a configurable ensemble.

Example configuration:

```yaml
ensemble:
  xgboost_weight: 0.40
  lstm_weight: 0.30
  transformer_weight: 0.30
```

Example:

```text
XGBoost       → 0.72
LSTM          → 0.84
Transformer   → 0.91
                    │
                    ▼
              Risk Engine
                    │
                    ▼
             Final Score
                    │
                    ▼
                  0.87
```

The initial ensemble is configurable. The weights are **not assumed to be optimal** and will be evaluated using validation data.

Future experimentation may include:

* Weighted ensemble
* Probability calibration
* Stacking
* Meta-classifier
* Cost-sensitive decision thresholds

---

# 🚦 Risk Classification

The initial application-level thresholds are:

|    Risk Score | Risk Level |
| ------------: | ---------- |
| `0.00 – 0.29` | LOW        |
| `0.30 – 0.69` | MEDIUM     |
| `0.70 – 1.00` | HIGH       |

These thresholds are configurable and should ultimately be selected based on validation results and business costs.

---

# ⚖️ Decision Engine

Risk classification and business decision are intentionally separated.

Example:

```text
LOW
 ↓
APPROVE

MEDIUM
 ↓
REVIEW

HIGH
 ↓
REVIEW / BLOCK
```

The initial implementation will keep high-risk predictions configurable rather than hard-coding automatic blocking.

This separation allows the ML model to answer:

> "How suspicious is this transaction?"

while the business policy answers:

> "What should the system do with this risk?"

---

# 📥 API Contract

## Predict Transaction

### Endpoint

```http
POST /api/v1/fraud/predict
```

### Request

```json
{
  "transaction_id": "TXN10001",
  "customer_id": "C1029",
  "amount": 25000,
  "currency": "INR",
  "merchant_category": "ELECTRONICS",
  "payment_method": "CARD",
  "device_id": "DEV998",
  "location": "Mumbai",
  "timestamp": "2026-09-25T13:45:32"
}
```

### Response

```json
{
  "transaction_id": "TXN10001",
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "decision": "REVIEW",
  "model_predictions": {
    "xgboost": 0.72,
    "lstm": 0.84,
    "transformer": 0.91
  },
  "reason_codes": [
    "UNUSUAL_TRANSACTION_AMOUNT",
    "NEW_DEVICE",
    "HIGH_TRANSACTION_VELOCITY"
  ],
  "model_version": "1.0.0"
}
```

---

# 🩺 Health Check

```http
GET /api/v1/health
```

Example:

```json
{
  "status": "UP",
  "service": "fraud-detection-api",
  "version": "1.0.0"
}
```

---

# 🤖 Model Information

```http
GET /api/v1/model/info
```

Expected response:

```json
{
  "model_version": "1.0.0",
  "models": [
    "xgboost",
    "lstm",
    "transformer"
  ]
}
```

---

# 📊 Evaluation Strategy

Fraud detection is an imbalanced classification problem.

Therefore, accuracy will **not** be used as the only evaluation metric.

Primary metrics:

```text
Precision
Recall
F1 Score
PR-AUC
ROC-AUC
Confusion Matrix
False Positive Rate
False Negative Rate
```

Operational metrics:

```text
Inference Latency
Requests Per Second
Error Rate
Model Loading Time
```

Business-oriented evaluation will eventually consider the relative cost of:

```text
False Positive
False Negative
Manual Review
Fraud Loss
```

---

# 🔬 Model Experimentation

The project will progressively compare:

```text
Baseline
   │
   ├── Logistic Regression
   │
   ├── Random Forest
   │
   └── XGBoost
            │
            ▼
       Sequence Models
            │
            ├── LSTM
            │
            └── Transformer
            │
            ▼
          Ensemble
```

The goal is to understand **why** a model performs differently, not simply report the highest metric.

---

# 🗂️ Repository Structure

```text
fraud-detection-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   └── external/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_sequence_generation.ipynb
│   ├── 05_lstm.ipynb
│   ├── 06_transformer.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── preprocessing/
│   └── utils/
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── services/
│   ├── models/
│   ├── ensemble/
│   ├── preprocessing/
│   ├── config/
│   └── utils/
│
├── training/
│
├── artifacts/
│   ├── xgboost/
│   ├── lstm/
│   └── transformer/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
│
├── monitoring/
├── scripts/
│
└── docs/
    ├── architecture.md
    └── requirements.md
```

---

# 🔄 ML Training Pipeline

```text
Raw Dataset
     │
     ▼
Data Validation
     │
     ▼
Data Cleaning
     │
     ▼
EDA
     │
     ▼
Feature Engineering
     │
     ├─────────────────┐
     ▼                 ▼
Tabular Features   Sequences
     │                 │
     ▼                 ├──────────────┐
  XGBoost              ▼              ▼
                     LSTM        Transformer
     │                 │              │
     └─────────────────┼──────────────┘
                       ▼
                 Model Evaluation
                       │
                       ▼
                 Model Selection
                       │
                       ▼
                Model Artifacts
```

---

# 🚀 Inference Pipeline

```text
API Request
     │
     ▼
Schema Validation
     │
     ▼
Feature Engineering
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
 XGBoost           LSTM        Transformer
     │               │               │
     └───────────────┼───────────────┘
                     ▼
                Risk Engine
                     │
                     ▼
               Final Prediction
                     │
                     ▼
                 API Response
```

Training and inference are intentionally separated.

The production API will **load trained model artifacts** rather than retraining models during requests.

---

# 🧪 Testing Strategy

The project will contain three levels of tests.

## Unit Tests

Test individual components:

```text
Feature engineering
Preprocessing
Sequence creation
Risk scoring
Model wrappers
```

## Integration Tests

Test:

```text
Feature pipeline
Model pipeline
Ensemble pipeline
Database interactions
```

## API Tests

Test:

```text
POST /api/v1/fraud/predict
GET /api/v1/health
GET /api/v1/model/info
```

Run:

```bash
pytest
```

---

# 🐳 Containerization

The application will eventually be containerized using Docker.

Target architecture:

```text
                 Docker Compose
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Flask API     PostgreSQL       Redis
        │
        ├── XGBoost
        ├── LSTM
        └── Transformer
```

---

# 📈 MLOps

MLflow will be introduced for experiment tracking and model lifecycle management.

We will track:

```text
Experiment
Dataset Version
Hyperparameters
Training Metrics
Validation Metrics
Model Artifacts
Model Version
```

Conceptually:

```text
Training Run
     │
     ▼
   MLflow
     │
     ├── Experiment
     ├── Metrics
     ├── Parameters
     └── Artifacts
              │
              ▼
        Model Registry
              │
              ▼
          Production
```

---

# 📡 Monitoring

The production service will eventually monitor:

### Application metrics

```text
Request count
Error rate
Latency
Throughput
```

### ML metrics

```text
Prediction distribution
Fraud prediction rate
Score distribution
Feature drift
Prediction drift
Model performance
```

### Logging

Each prediction should have structured logs containing information such as:

```text
request_id
transaction_id
model_version
prediction
risk_score
latency
timestamp
```

Sensitive financial/customer information should not be unnecessarily written to application logs.

---

# 🔐 Security Considerations

The system will follow basic production security practices:

* No secrets committed to Git.
* Environment variables for credentials.
* Input validation.
* API authentication where required.
* Rate limiting.
* Secure HTTP headers.
* Sanitized error responses.
* Structured logging.
* No unnecessary sensitive transaction/customer information in logs.

Example:

```text
.env
```

will remain outside version control.

Only:

```text
.env.example
```

will be committed.

---

# 📦 Configuration

Configuration is centralized in:

```text
configs/config.yaml
```

Example:

```yaml
project:
  name: "AI Fraud Detection & Risk Intelligence Platform"
  version: "1.0.0"

data:
  target_column: "is_fraud"
  test_size: 0.20
  validation_size: 0.10
  random_state: 42

models:
  xgboost:
    enabled: true

  lstm:
    enabled: true
    sequence_length: 20

  transformer:
    enabled: true
    sequence_length: 20

ensemble:
  xgboost_weight: 0.40
  lstm_weight: 0.30
  transformer_weight: 0.30

risk:
  low_threshold: 0.30
  high_threshold: 0.70
```

These values are initial configuration values and will be validated through experimentation.

---

# 💻 Local Development

## 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fraud-detection-ai
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment

Copy:

```text
.env.example
```

to:

```text
.env
```

and configure local environment variables.

## 5. Run Tests

```bash
pytest
```

## 6. Run Flask Application

Development execution will be documented here once the Flask application is implemented.

---

# 🛠️ Development Roadmap

## Phase 1 — Project Foundation

* [x] Define business problem
* [x] Define model architecture
* [x] Define API contract
* [x] Define repository structure
* [ ] Select dataset
* [ ] Define data contract

## Phase 2 — Data Engineering

* [ ] Dataset ingestion
* [ ] Data validation
* [ ] EDA
* [ ] Missing-value handling
* [ ] Outlier analysis
* [ ] Feature engineering
* [ ] Class imbalance analysis

## Phase 3 — Machine Learning

* [ ] Logistic Regression baseline
* [ ] Random Forest baseline
* [ ] XGBoost
* [ ] Hyperparameter tuning
* [ ] Threshold analysis
* [ ] Model evaluation

## Phase 4 — Sequence Modeling

* [ ] Transaction sequence generation
* [ ] Sequence preprocessing
* [ ] LSTM
* [ ] GRU experimentation
* [ ] LSTM evaluation

## Phase 5 — Transformer

* [ ] Transaction embeddings
* [ ] Positional encoding
* [ ] Self-attention
* [ ] Transformer encoder
* [ ] Transformer training
* [ ] Transformer evaluation

## Phase 6 — Ensemble

* [ ] Model probability calibration
* [ ] Weighted ensemble
* [ ] Risk engine
* [ ] Decision policy
* [ ] Explainability

## Phase 7 — Flask API

* [ ] Flask application factory
* [ ] REST endpoints
* [ ] Request validation
* [ ] Model loading
* [ ] Prediction service
* [ ] Error handling
* [ ] API logging

## Phase 8 — Testing

* [ ] Unit tests
* [ ] Integration tests
* [ ] API tests
* [ ] Test coverage
* [ ] Edge-case testing

## Phase 9 — Deployment

* [ ] Dockerfile
* [ ] Docker Compose
* [ ] Production WSGI server
* [ ] Environment configuration
* [ ] Health checks

## Phase 10 — MLOps

* [ ] MLflow tracking
* [ ] Model registry
* [ ] Model versioning
* [ ] Experiment tracking
* [ ] Data/model lineage

## Phase 11 — Monitoring

* [ ] Structured logging
* [ ] API metrics
* [ ] Model metrics
* [ ] Data drift
* [ ] Prediction drift
* [ ] Alerting

## Phase 12 — CI/CD

* [ ] GitHub Actions
* [ ] Automated tests
* [ ] Code quality checks
* [ ] Docker build
* [ ] Deployment pipeline

---

# 🧭 Engineering Principles

This project follows several production ML principles.

### 1. Separate training from inference

```text
Training ≠ Inference
```

Models are trained offline and loaded by the serving application.

### 2. Configuration over hardcoding

Model parameters, thresholds and paths should be configurable.

### 3. Reproducibility

Experiments should be reproducible using controlled random seeds, versioned configurations and tracked artifacts.

### 4. Evaluation beyond accuracy

Fraud detection requires metrics that reflect class imbalance and operational costs.

### 5. Model explainability

Where appropriate, the system should provide model-attribution information or reason codes without presenting them as causal explanations.

### 6. Observability

A production ML system must be observable at both:

```text
Application Level
+
Model Level
```

### 7. Fail safely

Model/service failures should result in controlled errors rather than exposing internal implementation details.

---

# ⚠️ Limitations

This repository is a **research/engineering portfolio project** and should not be connected directly to real financial transaction authorization systems without appropriate security, compliance, validation, monitoring and operational controls.

The models will be trained using publicly available or appropriately licensed datasets rather than real customer financial information.

Model performance on a public dataset should not be interpreted as production fraud-detection performance.

---

# 🔮 Future Improvements

Potential future extensions include:

* Real-time event streaming
* Kafka integration
* Online feature computation
* Feature store
* Redis-based low-latency features
* PostgreSQL transaction storage
* Model serving optimization
* GPU inference experimentation
* Advanced Transformer architectures
* Online learning
* Drift-triggered retraining
* Champion/challenger models
* Automated model promotion
* Cloud deployment
* Kubernetes
* Distributed inference
* Advanced observability

---

# 🎓 Learning Objectives

This project is designed to demonstrate practical understanding of:

### Machine Learning

* Classification
* Imbalanced datasets
* Feature engineering
* Ensemble methods
* XGBoost
* Model evaluation
* Threshold optimization

### Deep Learning

* Neural networks
* Sequence modeling
* RNN
* LSTM
* GRU
* Attention
* Transformers
* Positional encoding
* Self-attention

### Software Engineering

* Python
* Flask
* REST APIs
* Modular architecture
* Configuration management
* Exception handling
* Logging
* Testing

### MLOps

* MLflow
* Model versioning
* Experiment tracking
* Docker
* CI/CD
* Monitoring

---

# 👨‍💻 Development Philosophy

The project will be developed incrementally.

We will **first make each component correct**, then make it production-ready.

```text
Correctness
    ↓
Modularity
    ↓
Testing
    ↓
Performance
    ↓
Observability
    ↓
Deployment
```

We will avoid adding infrastructure simply for the appearance of complexity.

Every technology introduced into the project should solve a concrete engineering problem.

---

# 📜 License

This project will use an appropriate open-source license once the repository is finalized.

---

# 👤 Author

**Soumya Ranjan Rout**

Machine Learning / AI Engineer — Project Portfolio

---

# ⭐ Project Status

```text
🚧 ACTIVE DEVELOPMENT

Current Phase:
Project Foundation

Next Phase:
Dataset Selection + Data Contract
```

---
