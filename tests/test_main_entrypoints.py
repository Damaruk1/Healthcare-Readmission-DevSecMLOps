"""Tests exercising each module's CLI entrypoint (main()) against real project data.

These call the same code path used when running each script directly
(`python src/.../module.py`), giving coverage of the file I/O / orchestration
glue that pure unit tests around the underlying functions don't reach.
Skipped gracefully if the prerequisite artifacts from an earlier pipeline
stage aren't present on disk.
"""

from __future__ import annotations

import pytest

from src.utils.config import load_config, resolve_path


class TestValidateDataMain:
    def test_main_runs_against_real_raw_data(self):
        cfg = load_config()
        if not resolve_path(cfg.data.raw_path).exists():
            pytest.skip("Raw dataset not present; run data generation first.")
        from src.validation.validate_data import main

        main()  # raises SystemExit(1) if validation fails — success is silent return
        assert resolve_path(cfg.data.validated_flag_path).exists()


class TestEdaMain:
    def test_main_runs_against_real_raw_data(self):
        cfg = load_config()
        if not resolve_path(cfg.data.raw_path).exists():
            pytest.skip("Raw dataset not present; run data generation first.")
        from src.validation.eda import main

        main()
        assert resolve_path("reports/eda/correlation_heatmap.png").exists()


class TestDriftDetectionMain:
    def test_main_runs_against_processed_splits(self):
        cfg = load_config()
        if not resolve_path(cfg.data.processed_test_path).exists():
            pytest.skip("Processed test split not present; run preprocessing first.")
        from src.monitoring.drift_detection import main

        main()
        assert resolve_path("reports/drift/drift_summary.json").exists()
