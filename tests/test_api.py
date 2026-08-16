"""Integration tests for the FastAPI application.

These tests exercise the real API against whatever model artifacts exist on
disk (models/best_model.joblib + models/preprocessor.joblib). Run the
training pipeline first (`python src/training/train.py`) for the
prediction-dependent tests to exercise the full code path; if artifacts are
absent, those tests assert the API's documented 503 behavior instead of
failing outright.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.inference.predictor import get_predictor

client = TestClient(app)


@pytest.fixture(scope="module")
def model_ready() -> bool:
    return get_predictor().is_ready


class TestGeneralEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "message" in body
        assert "disclaimer" in body

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert "model_loaded" in body

    def test_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.json()["info"]["title"]


class TestPredictEndpoint:
    def test_predict_valid_payload(self, sample_patient_payload, model_ready):
        response = client.post("/predict", json=sample_patient_payload)
        if not model_ready:
            assert response.status_code == 503
            return
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk_level"] in {"Low", "Moderate", "High", "Very High"}
        assert "model_version" in body
        assert "timestamp" in body

    def test_predict_rejects_invalid_age(self, sample_patient_payload):
        payload = dict(sample_patient_payload)
        payload["age"] = 5  # below minimum of 18
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_inconsistent_blood_pressure(self, sample_patient_payload):
        payload = dict(sample_patient_payload)
        payload["diastolic_bp"] = payload["systolic_bp"] + 10
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_invalid_enum(self, sample_patient_payload):
        payload = dict(sample_patient_payload)
        payload["gender"] = "NotAValidGender"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_missing_field(self, sample_patient_payload):
        payload = dict(sample_patient_payload)
        del payload["bmi"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_predict_batch_valid_csv(self, sample_patient_payload, model_ready):
        import pandas as pd

        df = pd.DataFrame([sample_patient_payload, sample_patient_payload])
        df.insert(0, "patient_id", ["PT_TEST_1", "PT_TEST_2"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        response = client.post("/predict-batch", files={"file": ("patients.csv", buf.getvalue(), "text/csv")})
        if not model_ready:
            assert response.status_code == 503
            return
        assert response.status_code == 200
        assert "patient_id" in response.text
        assert "prediction_label" in response.text

    def test_predict_batch_rejects_non_csv(self):
        response = client.post("/predict-batch", files={"file": ("patients.txt", "not,a,csv", "text/plain")})
        assert response.status_code == 400


class TestObservabilityEndpoints:
    def test_metrics_endpoint_returns_prometheus_format(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_model_info_endpoint(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert "model_name" in response.json()

    def test_version_endpoint(self):
        response = client.get("/version")
        assert response.status_code == 200
        body = response.json()
        assert "api_version" in body
        assert "model_version" in body

    def test_drift_report_endpoint_handles_absence_gracefully(self):
        response = client.get("/drift-report")
        assert response.status_code == 200
        body = response.json()
        assert "drift_detected" in body
        assert "report_available" in body


class TestRetrainEndpoint:
    def test_retrain_returns_accepted(self):
        response = client.post("/retrain")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "accepted"
        assert "triggered_at" in body
