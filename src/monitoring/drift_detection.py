"""Drift detection module using Evidently AI.

Compares a reference dataset (training distribution) against current
production data to detect feature drift, prediction drift, and overall data
drift. Generates an HTML report and a compact JSON summary consumed by the
`/drift-report` API endpoint and the auto-retraining trigger.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    """Runs Evidently AI drift analysis between reference and current data."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_config()

    def _build_dataset(self, df: pd.DataFrame, target_col: str | None) -> Dataset:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

        definition = DataDefinition(
            numerical_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )
        return Dataset.from_pandas(df, data_definition=definition)

    def run_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_dir: Path,
        target_col: str | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        common_cols = [c for c in reference_df.columns if c in current_df.columns]
        ref = reference_df[common_cols].copy()
        cur = current_df[common_cols].copy()

        ref_dataset = self._build_dataset(ref, target_col)
        cur_dataset = self._build_dataset(cur, target_col)

        report = Report([DataDriftPreset()])
        result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

        html_path = output_dir / "drift_report.html"
        result.save_html(str(html_path))

        result_dict = result.dict()
        drift_share = None
        try:
            for metric in result_dict.get("metrics", []):
                if "DriftedColumnsCount" in str(metric.get("metric_name", "")):
                    value = metric.get("value", {})
                    drift_share = value.get("share") if isinstance(value, dict) else value
                    break
        except Exception as exc:
            logger.warning(f"Could not extract drift share from report: {exc}")
            drift_share = None

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "reference_rows": len(ref),
            "current_rows": len(cur),
            "columns_analyzed": common_cols,
            "drift_share": drift_share,
            "drift_threshold": self.cfg.monitoring.drift_threshold,
            "drift_detected": bool(drift_share is not None and drift_share > self.cfg.monitoring.drift_threshold),
            "html_report_path": str(html_path),
        }

        json_path = output_dir / "drift_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Drift report generated: drift_detected={summary['drift_detected']} (share={drift_share})")
        return summary


def main() -> None:
    cfg = load_config()
    reference_path = resolve_path(cfg.monitoring.drift_reference_path)
    reference_df = pd.read_csv(reference_path)

    # In absence of live production traffic, use the held-out test split as a
    # stand-in "current" batch so the report is runnable end-to-end out of the box.
    current_path = resolve_path(cfg.data.processed_test_path)
    current_df = pd.read_csv(current_path)

    monitor = DriftMonitor(cfg)
    output_dir = resolve_path("reports/drift")
    monitor.run_drift_report(reference_df, current_df, output_dir, target_col=cfg.data.target_column)


if __name__ == "__main__":
    main()
