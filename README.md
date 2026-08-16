# Healthcare Patient Readmission Risk Prediction — DevSecMLOps

An end-to-end, production-shaped MLOps + DevSecOps project: predicts whether
a hospital patient will be readmitted within 30 days of discharge, using a
synthetic dataset and a full pipeline from data generation through a served,
observable, auto-retraining API.

> **Educational project — 100% synthetic data. Never use for real clinical
> decisions.** See [`docs/HEALTHCARE_DISCLAIMER.md`](docs/HEALTHCARE_DISCLAIMER.md).

## What's Actually Here (and verified working end-to-end)

- **Synthetic data generator** — 5,000+ rows, 27 columns, realistic risk
  correlations, injected missingness/outliers/duplicates, ~22% class imbalance.
- **Data validation** — 66 automated checks (schema, dtype, missing values,
  duplicates, ranges, categorical/binary domains, business rules, target
  sanity), JSON + HTML reports.
- **EDA** — correlation heatmap, distributions, missing-value heatmap, class
  imbalance, pairplot, boxplots, violin plots, summary statistics.
- **Feature engineering** — BMI/age/glucose/BP categories, hospital
  utilization score, disease burden score, medication burden, a composite
  readmission risk index.
- **Preprocessing** — dedup, IQR winsorization, median/mode imputation,
  one-hot + ordinal-style encoding, `StandardScaler`, stratified train/test
  split, SMOTE (training fold only), all inside a `ColumnTransformer`.
- **Training** — Logistic Regression, Random Forest, Gradient Boosting,
  XGBoost, LightGBM, CatBoost, each with 5-fold CV + Optuna tuning, tracked
  in MLflow, best model auto-selected and registered.
- **Evaluation** — accuracy/precision/recall/F1/ROC AUC/PR AUC, confusion
  matrix, ROC/PR curves, calibration curve, SHAP explainability, feature
  importance.
- **FastAPI service** — `/`, `/health`, `/predict`, `/predict-batch`,
  `/metrics`, `/model-info`, `/version`, `/retrain`, `/drift-report` (+html),
  `/docs`. Pydantic-validated inputs, Prometheus-instrumented.
- **Drift detection** — Evidently AI, HTML + JSON reports.
- **Auto-retraining** — triggers on drift-threshold breach or performance
  degradation, re-runs training, re-registers the model.
- **Docker** — multi-stage production API image + a separate pipeline image;
  `docker compose up` brings up API + Prometheus + Grafana.
- **CI/CD** — GitHub Actions: lint → security scan → tests → Docker
  build/Trivy scan/smoke-test → push to GHCR → deploy → post-deploy smoke
  tests.
- **Security** — Bandit (0 issues on 1,918 lines), Ruff (0 issues),
  Safety, Trivy, gitleaks, pre-commit hooks.
- **Monitoring** — Prometheus metrics (prediction count, latency, CPU, RAM,
  model version, API errors), a provisioned Grafana dashboard.
- **Tests** — 66 tests, **91% coverage** (`pytest --cov`), unit + integration
  + API + pipeline-entrypoint tests.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

python src/data_generation/generate_data.py
python src/validation/validate_data.py
python src/preprocessing/preprocess.py
python src/training/train.py
python src/evaluation/evaluate.py

uvicorn src.api.main:app --reload
# -> http://localhost:8000/docs
```

Or with Docker Compose (see `docs/DEPLOYMENT_GUIDE.md` for the full sequence):
```bash
docker compose up --build
```

## Example Prediction

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age":67,"gender":"Male","bmi":31.5,"systolic_bp":145,"diastolic_bp":92,"heart_rate":88,
  "blood_glucose":178,"hba1c":8.2,"cholesterol":235,"number_of_medications":7,
  "previous_admissions":3,"length_of_stay":8,"emergency_visits_last_year":2,
  "chronic_disease_count":4,"diabetes":1,"hypertension":1,"heart_disease":1,"kidney_disease":0,
  "smoking_status":"Former","alcohol_consumption":"Low","physical_activity_level":"Low",
  "discharge_destination":"Home","follow_up_scheduled":0,"insurance_type":"Private","admission_type":"Emergency"
}'
```
```json
{"prediction":1,"prediction_label":"Readmitted within 30 days","probability":0.7407,"risk_level":"High","model_version":"1.0.0","timestamp":"2026-08-06T09:07:07Z"}
```

## Model Performance (current artifact)

Best model: **Random Forest** — Test ROC AUC **0.791**, PR AUC 0.539,
Accuracy 0.799. Full comparison across all 6 models and per-metric detail in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System + request-flow diagrams, component responsibilities, design rationale |
| [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) | Every endpoint, request/response schemas, curl examples |
| [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md) | Docker Compose, Terraform/ECS, CI/CD-driven deployment |
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Local setup, running each pipeline stage, extending the project |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Model details, training data, performance, limitations |
| [`docs/HEALTHCARE_DISCLAIMER.md`](docs/HEALTHCARE_DISCLAIMER.md) | Scope and limits of this project — read before using |

## Tech Stack

Python 3.12 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost ·
Optuna · MLflow · Great Expectations-style validation · FastAPI · Pydantic ·
Docker · Docker Compose · GitHub Actions · Terraform · Prometheus · Grafana ·
Loguru · Evidently AI · pytest · Bandit · Safety · Trivy · Ruff · pre-commit ·
SHAP · Matplotlib · Seaborn

## License

Educational/portfolio use. No warranty. See `docs/HEALTHCARE_DISCLAIMER.md`
for the scope restrictions on this project.
