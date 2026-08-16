"""Tests for the data validation module."""

from __future__ import annotations

import pandas as pd

from src.validation.validate_data import HealthcareDataValidator


class TestHealthcareDataValidator:
    def test_valid_synthetic_data_passes(self, synthetic_df: pd.DataFrame):
        validator = HealthcareDataValidator(synthetic_df)
        report = validator.run_all()
        assert report.success is True
        assert report.n_failed == 0

    def test_schema_check_fails_on_missing_column(self, synthetic_df: pd.DataFrame):
        broken = synthetic_df.drop(columns=["age"])
        validator = HealthcareDataValidator(broken)
        validator.validate_schema()
        result = next(r for r in validator.report.results if r.check_name == "schema_columns_match")
        assert result.passed is False

    def test_target_binary_check_fails_on_invalid_values(self, synthetic_df: pd.DataFrame):
        broken = synthetic_df.copy()
        broken.loc[0, "readmitted_30_days"] = 5
        validator = HealthcareDataValidator(broken)
        validator.validate_target()
        result = next(r for r in validator.report.results if r.check_name == "target_binary_values")
        assert result.passed is False

    def test_range_check_flags_extreme_outlier_fraction(self, synthetic_df: pd.DataFrame):
        broken = synthetic_df.copy()
        # push a large fraction of ages far out of range
        broken.loc[: len(broken) // 2, "age"] = 500
        validator = HealthcareDataValidator(broken)
        validator.validate_ranges()
        result = next(r for r in validator.report.results if r.check_name == "range_age")
        assert result.passed is False

    def test_business_rule_bp_consistency(self, synthetic_df: pd.DataFrame):
        broken = synthetic_df.copy()
        broken["diastolic_bp"] = broken["systolic_bp"] + 10  # always invalid
        validator = HealthcareDataValidator(broken)
        validator.validate_business_rules()
        result = next(r for r in validator.report.results if r.check_name == "business_rule_bp_consistency")
        assert result.passed is False

    def test_report_to_dict_is_json_serializable(self, synthetic_df: pd.DataFrame):
        import json

        validator = HealthcareDataValidator(synthetic_df)
        report = validator.run_all()
        serialized = json.dumps(report.to_dict())
        assert isinstance(serialized, str)
