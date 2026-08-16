"""Integration tests covering the inference layer and end-to-end pipeline wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.predictor import ReadmissionPredictor, risk_level_from_probability
from src.utils.config import load_config, resolve_path


class TestRiskLevelBucketing:
    @pytest.mark.parametrize(
        "probability,expected",
        [(0.05, "Low"), (0.30, "Moderate"), (0.60, "High"), (0.90, "Very High")],
    )
    def test_bucket_boundaries(self, probability, expected):
        assert risk_level_from_probability(probability) == expected


class TestReadmissionPredictor:
    def test_predictor_reports_ready_when_artifacts_exist(self):
        cfg = load_config()
        model_path = resolve_path(cfg.training.model_output_path)
        preprocessor_path = resolve_path(cfg.training.preprocessor_output_path)
        predictor = ReadmissionPredictor(cfg)
        expected_ready = model_path.exists() and preprocessor_path.exists()
        assert predictor.is_ready == expected_ready

    def test_predict_single_raises_when_not_ready(self, monkeypatch, sample_patient_payload):
        predictor = ReadmissionPredictor.__new__(ReadmissionPredictor)
        predictor.model = None
        predictor.preprocessor = None
        with pytest.raises(RuntimeError):
            predictor.predict_single(sample_patient_payload)

    def test_predict_single_end_to_end(self, sample_patient_payload):
        predictor = ReadmissionPredictor()
        if not predictor.is_ready:
            pytest.skip("Model artifacts not present; run training pipeline first.")
        result = predictor.predict_single(sample_patient_payload)
        assert result["prediction"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0
        assert result["risk_level"] in {"Low", "Moderate", "High", "Very High"}


class TestConfigLoading:
    def test_config_loads_and_has_expected_sections(self):
        cfg = load_config()
        for section in ["project", "data", "preprocessing", "training", "monitoring", "api", "logging"]:
            assert section in cfg

    def test_resolve_path_returns_absolute_path(self):
        path = resolve_path("configs/config.yaml")
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert path.exists()
