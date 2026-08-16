"""Tests for the EDA report generator — verifies every artifact is produced."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.eda import run_eda


class TestRunEda:
    def test_all_expected_artifacts_created(self, synthetic_df: pd.DataFrame, tmp_path: Path):
        run_eda(synthetic_df, tmp_path)
        expected_files = [
            "correlation_heatmap.png",
            "distributions.png",
            "missing_value_heatmap.png",
            "class_imbalance.png",
            "pairplot.png",
            "boxplots.png",
            "violin_plots.png",
            "summary_statistics.csv",
        ]
        for filename in expected_files:
            assert (tmp_path / filename).exists(), f"missing EDA artifact: {filename}"
            assert (tmp_path / filename).stat().st_size > 0

    def test_summary_statistics_contains_all_columns(self, synthetic_df: pd.DataFrame, tmp_path: Path):
        run_eda(synthetic_df, tmp_path)
        summary = pd.read_csv(tmp_path / "summary_statistics.csv", index_col=0)
        assert set(synthetic_df.columns).issubset(set(summary.index))
