# Architecture

## System Overview

```mermaid
flowchart TD
    subgraph Offline["Offline / Batch Pipeline"]
        A[Synthetic Data Generator] --> B[Data Validation<br/>schema/range/business rules]
        B --> C[EDA Report Generation]
        B --> D[Preprocessing<br/>impute · winsorize · encode · scale · SMOTE]
        D --> E[Feature Engineering<br/>risk scores · categorical buckets]
        E --> F[Model Training<br/>6 models x Optuna x 5-fold CV]
        F --> G[MLflow Tracking and Model Registry]
        F --> H[Model Evaluation<br/>metrics · SHAP · calibration]
        G --> I[(models/best_model.joblib<br/>models/preprocessor.joblib)]
    end

    subgraph Online["Online Serving"]
        I --> J[FastAPI Inference Service]
        J --> K[/predict/]
        J --> L[/predict-batch/]
        J --> M[/metrics - Prometheus/]
        J --> N[/health · /model-info · /version/]
    end

    subgraph Observability["Observability and Retraining"]
        M --> O[Prometheus]
        O --> P[Grafana Dashboards]
        Q[Evidently Drift Monitor] -->|drift/perf trigger| R[Auto-Retraining Orchestrator]
        R --> F
        J --> Q
    end

    subgraph Delivery["DevSecMLOps Delivery"]
        S[GitHub Actions CI/CD] --> T[Lint: Ruff]
        S --> U[Security: Bandit/Safety/Trivy/Gitleaks]
        S --> V[Test Suite: pytest, 90%+ coverage]
        S --> W[Docker Build and Push to GHCR]
        W --> X[Terraform: ECS Fargate + ALB]
        X --> J
    end
```

## Request Flow: `/predict`

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI (/predict)
    participant Val as Pydantic Validation
    participant Pred as ReadmissionPredictor
    participant FE as FeatureEngineer
    participant Pre as Preprocessor (ColumnTransformer)
    participant Model as RandomForestClassifier

    Client->>API: POST /predict {patient features}
    API->>Val: validate schema, ranges, enums, BP consistency
    Val-->>API: 422 on failure / validated payload
    API->>Pred: predict_single(patient)
    Pred->>FE: transform() - derive risk features
    FE->>Pre: transform() - impute/encode/scale
    Pre->>Model: predict_proba(X)
    Model-->>Pred: probability
    Pred-->>API: prediction, label, probability, risk_level
    API-->>Client: 200 PredictionResponse
    API->>API: increment Prometheus counters
```

## Component Responsibilities

| Layer | Module(s) | Responsibility |
|---|---|---|
| Data Generation | `src/data_generation/` | Synthetic dataset with realistic risk structure |
| Validation | `src/validation/validate_data.py` | Schema/range/business-rule gate before downstream use |
| EDA | `src/validation/eda.py` | Exploratory visualizations & summary stats |
| Feature Engineering | `src/feature_engineering/` | Clinically-motivated derived features |
| Preprocessing | `src/preprocessing/` | Cleaning, imputation, encoding, scaling, SMOTE |
| Training | `src/training/train.py` | CV + Optuna tuning across 6 model families, MLflow tracking |
| Evaluation | `src/evaluation/evaluate.py` | Metrics, curves, calibration, SHAP |
| Inference | `src/inference/predictor.py` | Loads artifacts once, serves single/batch predictions |
| API | `src/api/` | FastAPI app, Pydantic schemas, Prometheus instrumentation |
| Monitoring | `src/monitoring/drift_detection.py` | Evidently AI drift analysis |
| Retraining | `src/retraining/retrain_trigger.py` | Drift/performance-triggered retraining orchestration |
| Utils | `src/utils/` | Config loading, structured Loguru logging |

## Why These Design Choices

- **ColumnTransformer + joblib bundle**: the fitted preprocessor, feature
  engineer, and feature-name list are persisted together
  (`models/preprocessor.joblib`) so inference-time transformation is
  guaranteed identical to training-time transformation — a common source of
  training/serving skew if handled separately.
- **SMOTE only on the training fold**: applied after the train/test split so
  the held-out test set reflects the true (imbalanced) population and
  evaluation metrics aren't optimistically biased.
- **MLflow with a SQLite backend**: zero external dependencies for local/CI
  runs; swap `training.mlflow_tracking_uri` in `configs/config.yaml` for a
  remote tracking server in a real multi-user deployment.
- **Native validation engine over a live Great Expectations context**: the
  rule set (schema/dtype/missing/duplicate/range/categorical/binary/business
  rule/target checks) is a pandas-based superset of what a GE
  ExpectationSuite enforces, avoiding GE's heavier context/store
  bootstrapping for a single-table pipeline while remaining portable to one.
