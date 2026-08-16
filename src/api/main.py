"""FastAPI application: Healthcare Patient Readmission Risk Prediction API.

Exposes single/batch prediction, health, metrics, model-info, version,
retrain-trigger, and drift-report endpoints. Instrumented with Prometheus
metrics and structured Loguru logging.

DISCLAIMER: This API serves predictions from a model trained purely on
synthetic data, for educational/demo purposes only. It must never be used
for real clinical decision-making.
"""

from __future__ import annotations

import io
import json
import time
from datetime import UTC, datetime

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import (
    DriftReportResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
    RetrainResponse,
    VersionResponse,
)
from src.inference.predictor import get_predictor
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()

app = FastAPI(
    title=cfg.api.title,
    version=cfg.api.version,
    description=(
        "Predicts 30-day hospital readmission risk from patient clinical/demographic data. "
        "**Educational use only — trained entirely on synthetic data. "
        "Never use for real clinical decisions.**"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "readmission_predictions_total", "Total number of predictions served", ["endpoint", "prediction_label"]
)
REQUEST_LATENCY = Histogram("readmission_request_latency_seconds", "Request latency in seconds", ["endpoint"])
API_ERROR_COUNTER = Counter("readmission_api_errors_total", "Total API errors", ["endpoint", "error_type"])
MODEL_VERSION_GAUGE_LABEL = cfg.project.version


@app.middleware("http")
async def add_process_time_and_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001 - top-level safety net, re-raised after logging
        API_ERROR_COUNTER.labels(endpoint=request.url.path, error_type=type(exc).__name__).inc()
        logger.exception(f"Unhandled exception on {request.url.path}")
        raise
    duration = time.perf_counter() - start_time
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
    return response


# ---------------------------------------------------------------------------
# Root & health
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "message": "Healthcare Patient Readmission Risk Prediction API",
        "docs": "/docs",
        "disclaimer": "Educational use only. Trained on synthetic data. Never use for real clinical decisions.",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    predictor = get_predictor()
    return HealthResponse(
        status="ok" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures):
    predictor = get_predictor()
    if not predictor.is_ready:
        API_ERROR_COUNTER.labels(endpoint="/predict", error_type="ModelNotReady").inc()
        raise HTTPException(status_code=503, detail="Model is not trained/loaded yet. Run the training pipeline first.")

    try:
        result = predictor.predict_single(patient.model_dump())
    except Exception as exc:
        API_ERROR_COUNTER.labels(endpoint="/predict", error_type=type(exc).__name__).inc()
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    PREDICTION_COUNTER.labels(endpoint="/predict", prediction_label=result["prediction_label"]).inc()
    logger.info(f"Prediction served: {result['prediction_label']} (p={result['probability']})")
    return PredictionResponse(**result)


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model is not trained/loaded yet. Run the training pipeline first.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for batch prediction.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        API_ERROR_COUNTER.labels(endpoint="/predict-batch", error_type="CSVParseError").inc()
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    try:
        result_df = predictor.predict_batch(df)
    except Exception as exc:
        API_ERROR_COUNTER.labels(endpoint="/predict-batch", error_type=type(exc).__name__).inc()
        logger.error(f"Batch prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    for label, count in result_df["prediction_label"].value_counts().items():
        PREDICTION_COUNTER.labels(endpoint="/predict-batch", prediction_label=label).inc(count)
    logger.info(f"Batch prediction served for {len(result_df)} rows")

    buffer = io.StringIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"},
    )


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Observability"])
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Observability"])
def model_info():
    predictor = get_predictor()
    meta = predictor.metadata
    return ModelInfoResponse(
        model_name=meta.get("best_model_name", "unknown"),
        model_version=meta.get("model_version", cfg.project.version),
        trained_at=meta.get("trained_at"),
        test_roc_auc=meta.get("best_test_roc_auc"),
        all_model_results=meta.get("all_results"),
        feature_count=len(predictor.feature_names) if predictor.feature_names else None,
    )


@app.get("/version", response_model=VersionResponse, tags=["Observability"])
def version():
    predictor = get_predictor()
    return VersionResponse(
        api_version=cfg.api.version,
        model_version=predictor.metadata.get("model_version", cfg.project.version),
        project_name=cfg.project.name,
    )


@app.get("/drift-report", response_model=DriftReportResponse, tags=["Observability"])
def drift_report():
    summary_path = resolve_path("reports/drift/drift_summary.json")
    if not summary_path.exists():
        return DriftReportResponse(
            drift_detected=False,
            report_available=False,
            report_path=None,
            generated_at=None,
            message="No drift report has been generated yet. POST to a retraining/monitoring job first.",
        )

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    return DriftReportResponse(
        drift_detected=summary.get("drift_detected", False),
        report_available=True,
        report_path=summary.get("html_report_path"),
        generated_at=summary.get("generated_at"),
        message="Drift report available.",
    )


@app.get("/drift-report/html", response_class=HTMLResponse, tags=["Observability"])
def drift_report_html():
    html_path = resolve_path("reports/drift/drift_report.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="No drift HTML report has been generated yet.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
def _run_retraining_job() -> None:
    from src.retraining.retrain_trigger import RetrainingOrchestrator

    logger.info("Background retraining job started")
    orchestrator = RetrainingOrchestrator(cfg)
    result = orchestrator.maybe_retrain()
    logger.info(f"Background retraining job finished: {result}")

    global _predictor_needs_reload
    _predictor_needs_reload = True


@app.post("/retrain", response_model=RetrainResponse, tags=["Retraining"])
def retrain(background_tasks: BackgroundTasks, force: bool = False):
    if force:
        from src.training.train import ModelTrainer

        background_tasks.add_task(lambda: ModelTrainer(cfg).train_all())
        message = "Forced retraining job started in the background."
    else:
        background_tasks.add_task(_run_retraining_job)
        message = (
            "Retraining evaluation started in the background (will only retrain if drift/performance triggers fire)."
        )

    logger.info(message)
    return RetrainResponse(status="accepted", message=message, triggered_at=datetime.now(UTC))
