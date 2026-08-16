# API Documentation

Interactive OpenAPI docs are always available at `/docs` (Swagger UI) and
`/redoc` while the API is running. This document is a static reference.

Base URL (local): `http://localhost:8000`

## Authentication

None — this is a demo/educational API with no auth layer. Add an API-key or
OAuth2 dependency (`fastapi.security`) before exposing this beyond a private
network.

## Endpoints

### `GET /`
Root info + disclaimer. No parameters.

### `GET /health`
Liveness/readiness probe.
```json
{"status": "ok", "model_loaded": true, "timestamp": "2026-08-06T09:04:17Z"}
```
`status` is `"degraded"` if the model artifacts haven't been trained/loaded yet.

### `POST /predict`
Single-patient prediction.

**Request body:**
```json
{
  "age": 67, "gender": "Male", "bmi": 31.5,
  "systolic_bp": 145, "diastolic_bp": 92, "heart_rate": 88,
  "blood_glucose": 178, "hba1c": 8.2, "cholesterol": 235,
  "number_of_medications": 7, "previous_admissions": 3, "length_of_stay": 8,
  "emergency_visits_last_year": 2, "chronic_disease_count": 4,
  "diabetes": 1, "hypertension": 1, "heart_disease": 1, "kidney_disease": 0,
  "smoking_status": "Former", "alcohol_consumption": "Low",
  "physical_activity_level": "Low", "discharge_destination": "Home",
  "follow_up_scheduled": 0, "insurance_type": "Private", "admission_type": "Emergency"
}
```

**Response `200`:**
```json
{
  "prediction": 1,
  "prediction_label": "Readmitted within 30 days",
  "probability": 0.7407,
  "risk_level": "High",
  "model_version": "1.0.0",
  "timestamp": "2026-08-06T09:07:07Z"
}
```

**Risk level buckets:** `Low` (<0.25), `Moderate` (0.25–0.49), `High`
(0.50–0.74), `Very High` (≥0.75).

**Errors:** `422` on schema/validation failure (e.g. `diastolic_bp >=
systolic_bp`, out-of-range values, invalid enum values, missing fields);
`503` if the model isn't trained/loaded yet.

### `POST /predict-batch`
Batch prediction from an uploaded CSV.

- **Form field:** `file` — a CSV with the same columns as the `/predict`
  body (plus an optional `patient_id` column; one is auto-generated per row
  if absent).
- **Response:** a downloadable CSV (`Content-Disposition: attachment`) with
  columns `patient_id, prediction, prediction_label, probability, risk_level,
  model_version, prediction_timestamp`.
- **Errors:** `400` if the file isn't a `.csv` or fails to parse; `503` if
  the model isn't ready.

### `GET /metrics`
Prometheus exposition-format metrics (see `docs/DEPLOYMENT_GUIDE.md` for the
metric names and what they track).

### `GET /model-info`
Current model name, version, training timestamp, test ROC AUC, comparison
results across all trained model families, and feature count.

### `GET /version`
API version, model version, project name.

### `GET /drift-report`
JSON summary of the most recent drift analysis (`drift_detected`,
`report_available`, `report_path`, `generated_at`). Returns
`report_available: false` if no drift run has happened yet.

### `GET /drift-report/html`
Renders the full Evidently AI HTML drift report, if one has been generated.

### `POST /retrain`
Triggers a background retraining check.
- `?force=true` — always retrains immediately, ignoring drift/performance
  triggers.
- Default (`force=false`) — only retrains if drift exceeds
  `monitoring.drift_threshold` or performance has degraded beyond
  `monitoring.performance_degradation_threshold` (see `configs/config.yaml`).
- Returns `202`-style acceptance immediately; the job runs in the background.

## Example: `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @patient.json

curl -X POST http://localhost:8000/predict-batch \
  -F "file=@patients.csv" \
  -o predictions.csv
```
