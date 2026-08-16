"""Tests for drift detection and the auto-retraining trigger orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.monitoring.drift_detection import DriftMonitor
from src.retraining.retrain_trigger import RetrainingOrchestrator
from src.utils.config import load_config, resolve_path


class TestDriftMonitor:
    def test_run_drift_report_on_identical_data_shows_no_drift(self, synthetic_df: pd.DataFrame, tmp_path: Path):
        cfg = load_config()
        monitor = DriftMonitor(cfg)
        numeric_df = synthetic_df.drop(columns=["patient_id"])
        summary = monitor.run_drift_report(numeric_df, numeric_df.copy(), tmp_path, target_col=cfg.data.target_column)

        assert summary["drift_detected"] is False
        assert (tmp_path / "drift_report.html").exists()
        assert (tmp_path / "drift_summary.json").exists()

    def test_run_drift_report_with_shifted_distribution(self, synthetic_df: pd.DataFrame, tmp_path: Path):
        cfg = load_config()
        monitor = DriftMonitor(cfg)
        reference = synthetic_df.drop(columns=["patient_id"])
        shifted = reference.copy()
        shifted["age"] = shifted["age"] + 40  # large synthetic shift to exercise the drifted path
        shifted["blood_glucose"] = shifted["blood_glucose"] * 2

        summary = monitor.run_drift_report(reference, shifted, tmp_path, target_col=cfg.data.target_column)
        assert "drift_share" in summary
        assert isinstance(summary["drift_detected"], bool)


class TestRetrainingOrchestrator:
    def test_check_performance_trigger_fires_on_degradation(self, tmp_path: Path):
        cfg = load_config()
        orchestrator = RetrainingOrchestrator(cfg)

        metadata_path = resolve_path(cfg.training.metadata_output_path)
        if not metadata_path.exists():
            pytest.skip("Model metadata not present; run the training pipeline first.")

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)
        baseline = metadata["best_test_roc_auc"]

        degraded_score = baseline - (cfg.monitoring.performance_degradation_threshold + 0.05)
        assert orchestrator.check_performance_trigger(degraded_score) is True

    def test_check_performance_trigger_does_not_fire_on_stable_score(self):
        cfg = load_config()
        orchestrator = RetrainingOrchestrator(cfg)

        metadata_path = resolve_path(cfg.training.metadata_output_path)
        if not metadata_path.exists():
            pytest.skip("Model metadata not present; run the training pipeline first.")

        import json

        with open(metadata_path) as f:
            metadata = json.load(f)
        baseline = metadata["best_test_roc_auc"]

        assert orchestrator.check_performance_trigger(baseline) is False

    def test_maybe_retrain_skips_when_no_trigger(self):
        cfg = load_config()
        orchestrator = RetrainingOrchestrator(cfg)

        with patch.object(orchestrator, "check_drift_trigger", return_value=False):
            result = orchestrator.maybe_retrain(current_roc_auc=None)

        assert result["retrained"] is False

    def test_maybe_retrain_invokes_training_when_drift_triggered(self):
        cfg = load_config()
        orchestrator = RetrainingOrchestrator(cfg)

        fake_result = {"best_model_name": "logistic_regression", "best_test_auc": 0.81}
        with (
            patch.object(orchestrator, "check_drift_trigger", return_value=True),
            patch("src.retraining.retrain_trigger.ModelTrainer") as MockTrainer,
        ):
            MockTrainer.return_value.train_all.return_value = fake_result
            result = orchestrator.maybe_retrain(current_roc_auc=None)

        assert result["retrained"] is True
        assert result["reason"] == "drift"
        assert result["new_best_model"] == "logistic_regression"
