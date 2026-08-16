"""Tests for the model evaluation module using existing trained artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.evaluation.evaluate import ModelEvaluator
from src.utils.config import load_config, resolve_path


@pytest.fixture(scope="module")
def trained_model_and_test_data():
    cfg = load_config()
    model_path = resolve_path(cfg.training.model_output_path)
    test_path = resolve_path(cfg.data.processed_test_path)
    if not model_path.exists() or not test_path.exists():
        pytest.skip("Trained model / processed test set not present; run the training pipeline first.")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    target_col = cfg.data.target_column
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    feature_names = test_df.drop(columns=[target_col]).columns.tolist()
    return model, X_test, y_test, feature_names


class TestModelEvaluator:
    def test_compute_metrics_returns_expected_keys(self, trained_model_and_test_data, tmp_path: Path):
        model, X_test, y_test, feature_names = trained_model_and_test_data
        evaluator = ModelEvaluator(model, feature_names, tmp_path)
        metrics = evaluator.compute_metrics(X_test, y_test)
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_run_full_evaluation_creates_all_figures(self, trained_model_and_test_data, tmp_path: Path):
        model, X_test, y_test, feature_names = trained_model_and_test_data
        evaluator = ModelEvaluator(model, feature_names, tmp_path)
        evaluator.run_full_evaluation(X_test, y_test)

        expected_files = [
            "confusion_matrix.png",
            "roc_curve.png",
            "pr_curve.png",
            "calibration_curve.png",
            "evaluation_metrics.json",
        ]
        for filename in expected_files:
            assert (tmp_path / filename).exists(), f"missing evaluation artifact: {filename}"
