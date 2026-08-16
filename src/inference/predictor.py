"""Inference service.

Loads the trained model + fitted preprocessing pipeline once and exposes a
single/batch prediction interface used by the FastAPI layer. Encapsulates
risk-level bucketing so it's consistent between single and batch prediction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def risk_level_from_probability(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "High"
    return "Very High"


class ReadmissionPredictor:
    """Loads model + preprocessing artifacts and serves predictions."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_config()
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = resolve_path(self.cfg.training.model_output_path)
        preprocessor_path = resolve_path(self.cfg.training.preprocessor_output_path)
        metadata_path = resolve_path(self.cfg.training.metadata_output_path)

        if not model_path.exists() or not preprocessor_path.exists():
            logger.warning(
                f"Model or preprocessor artifact missing at {model_path} / {preprocessor_path}. "
                "API will report not-ready until training has been run."
            )
            return

        self.model = joblib.load(model_path)
        bundle = joblib.load(preprocessor_path)
        self.preprocessor = bundle["preprocessor"]
        self.feature_engineer = bundle["feature_engineer"]
        self.feature_names = bundle["feature_names"]

        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(f"Loaded model ({self.metadata.get('best_model_name', 'unknown')}) and preprocessor")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        engineered = self.feature_engineer.transform(df)
        return self.preprocessor.transform(engineered)

    def predict_single(self, patient: dict[str, Any]) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded. Train the model before requesting predictions.")

        df = pd.DataFrame([patient])
        X = self._transform(df)
        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)

        return {
            "prediction": pred,
            "prediction_label": "Readmitted within 30 days" if pred == 1 else "Not readmitted within 30 days",
            "probability": round(proba, 4),
            "risk_level": risk_level_from_probability(proba),
            "model_version": self.metadata.get("model_version", self.cfg.project.version),
            "timestamp": datetime.now(UTC),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded. Train the model before requesting predictions.")

        id_col = df["patient_id"] if "patient_id" in df.columns else pd.Series([f"ROW_{i}" for i in range(len(df))])
        feature_df = df.drop(columns=[c for c in ["patient_id"] if c in df.columns])

        X = self._transform(feature_df)
        probas = self.model.predict_proba(X)[:, 1]
        preds = (probas >= 0.5).astype(int)
        now = datetime.now(UTC)

        result = pd.DataFrame(
            {
                "patient_id": id_col.values,
                "prediction": preds,
                "prediction_label": [
                    "Readmitted within 30 days" if p == 1 else "Not readmitted within 30 days" for p in preds
                ],
                "probability": np.round(probas, 4),
                "risk_level": [risk_level_from_probability(p) for p in probas],
                "model_version": self.metadata.get("model_version", self.cfg.project.version),
                "prediction_timestamp": now,
            }
        )
        return result


_predictor_singleton: ReadmissionPredictor | None = None


def get_predictor() -> ReadmissionPredictor:
    """Return a process-wide singleton predictor instance (loaded once)."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = ReadmissionPredictor()
    return _predictor_singleton
