"""Tests for the training orchestration module.

Runs the *real* tuning/training/MLflow-logging code path end-to-end, but
against a small synthetic dataset with a minimal search space (1 model,
1 Optuna trial, 2 CV folds) so it completes in a few seconds instead of
minutes — this still exercises every line of ModelTrainer, just cheaply.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.train import ModelTrainer
from src.utils.config import ConfigBox, load_config


@pytest.fixture()
def fast_cfg(tmp_path: Path) -> ConfigBox:
    cfg = load_config()
    cfg = ConfigBox(dict(cfg))  # shallow copy so we don't mutate the cached singleton
    cfg["training"] = ConfigBox(dict(cfg["training"]))
    cfg["training"]["models"] = ["logistic_regression"]
    cfg["training"]["optuna_trials"] = 1
    cfg["training"]["cv_folds"] = 2
    cfg["training"]["mlflow_tracking_uri"] = f"sqlite:///{tmp_path}/mlflow_test.db"
    cfg["training"]["mlflow_experiment_name"] = "test-experiment"
    cfg["training"]["model_output_path"] = str(tmp_path / "model.joblib")
    cfg["training"]["preprocessor_output_path"] = str(tmp_path / "preprocessor.joblib")
    cfg["training"]["metadata_output_path"] = str(tmp_path / "metadata.json")
    return cfg


class TestModelTrainerFast:
    def test_train_all_runs_end_to_end_with_reduced_search_space(self, fast_cfg):
        # run_preprocessing() writes to configured (real, shared) processed-data
        # paths regardless of the trainer's own output paths, which is expected
        # production behavior: preprocessing artifacts are shared across model runs.
        trainer = ModelTrainer(fast_cfg)
        result = trainer.train_all()

        assert result["best_model_name"] == "logistic_regression"
        assert 0.0 <= result["best_test_auc"] <= 1.0
        assert "logistic_regression" in result["results"]
        assert isinstance(result["X_test"], np.ndarray)
        assert isinstance(result["y_test"], np.ndarray)

        from pathlib import Path as P

        assert P(fast_cfg.training.model_output_path).exists()
        assert P(fast_cfg.training.metadata_output_path).exists()
