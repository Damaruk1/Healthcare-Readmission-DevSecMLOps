"""Auto-retraining trigger.

Checks whether drift has exceeded the configured threshold or whether live
performance has degraded beyond tolerance versus the metadata-recorded
training-time metric, and if so, re-runs the full training pipeline and
re-registers the new model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.monitoring.drift_detection import DriftMonitor
from src.training.train import ModelTrainer
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrainingOrchestrator:
    """Decides whether retraining is warranted and executes it if so."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_config()

    def check_drift_trigger(self) -> bool:
        import pandas as pd

        reference_df = pd.read_csv(resolve_path(self.cfg.monitoring.drift_reference_path))
        current_df = pd.read_csv(resolve_path(self.cfg.data.processed_test_path))

        monitor = DriftMonitor(self.cfg)
        summary = monitor.run_drift_report(
            reference_df, current_df, resolve_path("reports/drift"), target_col=self.cfg.data.target_column
        )
        return bool(summary["drift_detected"])

    def check_performance_trigger(self, current_roc_auc: float) -> bool:
        metadata_path = resolve_path(self.cfg.training.metadata_output_path)
        if not metadata_path.exists():
            return False
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        baseline_auc = metadata.get("best_test_roc_auc", 0.0)
        degradation = baseline_auc - current_roc_auc
        triggered = degradation > self.cfg.monitoring.performance_degradation_threshold
        if triggered:
            logger.warning(
                f"Performance degradation trigger: baseline={baseline_auc:.4f}, "
                f"current={current_roc_auc:.4f}, degradation={degradation:.4f}"
            )
        return triggered

    def maybe_retrain(self, current_roc_auc: float | None = None) -> dict[str, Any]:
        drift_triggered = self.check_drift_trigger()
        perf_triggered = self.check_performance_trigger(current_roc_auc) if current_roc_auc is not None else False

        if not (drift_triggered or perf_triggered):
            logger.info("No retraining trigger condition met. Skipping retraining.")
            return {
                "retrained": False,
                "reason": "no trigger condition met",
                "drift_triggered": drift_triggered,
                "performance_triggered": perf_triggered,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        reason = "drift" if drift_triggered else "performance_degradation"
        logger.info(f"Retraining triggered due to: {reason}. Starting full training pipeline...")

        trainer = ModelTrainer(self.cfg)
        result = trainer.train_all()

        return {
            "retrained": True,
            "reason": reason,
            "new_best_model": result["best_model_name"],
            "new_best_test_auc": result["best_test_auc"],
            "timestamp": datetime.now(UTC).isoformat(),
        }


def main() -> None:
    orchestrator = RetrainingOrchestrator()
    result = orchestrator.maybe_retrain()
    logger.info(f"Retraining check result: {result}")


if __name__ == "__main__":
    main()
