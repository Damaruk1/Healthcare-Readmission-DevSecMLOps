# Developer Guide

## Prerequisites

- Python 3.12
- Docker + Docker Compose (for containerized runs)
- ~2 GB free disk (venv + model artifacts + MLflow store)

## Local Setup

```bash
git clone <this-repo>
cd Healthcare-Readmission-DevSecMLOps
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pre-commit install              # optional but recommended
```

## Running the Full Pipeline Locally

Each stage is an independently runnable script, in this order:

```bash
export PYTHONPATH=.

python src/data_generation/generate_data.py     # -> data/raw/healthcare_readmission.csv
python src/validation/validate_data.py          # -> reports/validation/*.{json,html}; fails fast (exit 1) if checks fail
python src/validation/eda.py                    # -> reports/eda/*.{png,csv}
python src/preprocessing/preprocess.py          # -> data/processed/{train,test}.csv, models/preprocessor.joblib
python src/training/train.py                    # -> models/best_model.joblib, models/model_metadata.json, MLflow runs
python src/evaluation/evaluate.py               # -> reports/evaluation/*.{png,json}
python src/monitoring/drift_detection.py        # -> reports/drift/{drift_report.html,drift_summary.json}
```

Then serve the API:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Adjusting the Training Search Space

`configs/config.yaml` → `training.optuna_trials` and `training.cv_folds`
control tuning cost. The defaults (`optuna_trials: 20`, `cv_folds: 5`,
6 models) take several minutes; for fast local iteration, drop to
`optuna_trials: 3`, `cv_folds: 3`, and trim `training.models` to 1–2 entries.

## Inspecting Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000 to browse runs, compare metrics, and view
the registered model versions.

## Project Structure

```
src/
  data_generation/   synthetic dataset generator
  validation/        data validation suite + EDA report generator
  preprocessing/      cleaning, imputation, encoding, scaling, SMOTE
  feature_engineering/ derived clinical-risk features
  training/           multi-model CV + Optuna tuning + MLflow tracking
  evaluation/          metrics, curves, SHAP, feature importance
  inference/           model-loading + prediction service
  api/                 FastAPI app + Pydantic schemas
  monitoring/          Evidently AI drift detection
  retraining/          drift/performance-triggered retraining orchestration
  utils/               config loader, structured logging
configs/config.yaml    single source of truth for all paths/hyperparameters
tests/                 unit + integration + API tests (pytest, 90%+ coverage)
docker/, Dockerfile,
docker-compose.yml     container definitions
terraform/             AWS ECS Fargate reference infrastructure
monitoring/, dashboards/ Prometheus + Grafana configs
.github/workflows/     CI/CD pipeline
```

## Code Style & Quality Gates

```bash
ruff check src/ tests/ --fix     # lint + autofix
ruff format src/ tests/          # format
bandit -c pyproject.toml -r src/ # security static analysis
safety scan -r requirements.txt --key <SAFETY_API_KEY>   # dependency CVE scan (needs API key)
pytest tests/ -v                 # full test suite + coverage report (reports/coverage/)
```

All four run in CI on every push/PR (see `.github/workflows/ci-cd.yml`).

## Adding a New Model to the Comparison

1. Add an `_objective_<name>(trial, X, y, cv)` Optuna objective function in
   `src/training/train.py` following the existing pattern.
2. Register it in `MODEL_REGISTRY` with its estimator class.
3. Add its key to `training.models` in `configs/config.yaml`.

No other code changes needed — CV, tuning, MLflow logging, and best-model
selection all iterate over `training.models` generically.

## Extending the API

New endpoints go in `src/api/main.py`; request/response contracts go in
`src/api/schemas.py` as Pydantic models. The Prometheus counters
(`PREDICTION_COUNTER`, `REQUEST_LATENCY`, `API_ERROR_COUNTER`) are
module-level — reuse them from new endpoints for consistent observability.

## Common Gotchas

- Pandas' `read_csv` treats the literal string `"None"` as `NaN` by default
  — this is why the `alcohol_consumption` category is named `"Non-drinker"`
  rather than `"None"` (see `src/data_generation/generate_data.py`). Keep
  this in mind if you add new categorical values.
- `run_preprocessing()` always writes to the paths in `configs/config.yaml`
  (`data.processed_train_path` / `data.processed_test_path`), even when
  called from a `ModelTrainer` configured with different *model* output
  paths — this is intentional (processed data is shared across training
  runs) but worth knowing if you're sandboxing an experiment.
