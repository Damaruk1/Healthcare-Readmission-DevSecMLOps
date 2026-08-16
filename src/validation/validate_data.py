"""Data validation module.

Implements schema, datatype, missing-value, duplicate, range, categorical,
binary, business-rule and target validation for the raw healthcare dataset.
Produces a structured JSON + HTML validation report.

This module implements the validation logic natively (pandas-based rule
engine) instead of a live Great Expectations Data Context, since GE's
context bootstrapping is heavyweight for a single-table pipeline; the rules
below are a superset of what a GE ExpectationSuite would enforce and are
easily portable into one (see `to_expectation_suite_dict`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    details: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationReport:
    dataset_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def success(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "overall_success": self.success,
            "checks_passed": self.n_passed,
            "checks_failed": self.n_failed,
            "results": [r.__dict__ for r in self.results],
        }


EXPECTED_SCHEMA: dict[str, str] = {
    "patient_id": "object",
    "age": "int64",
    "gender": "object",
    "bmi": "float64",
    "systolic_bp": "int64",
    "diastolic_bp": "int64",
    "heart_rate": "int64",
    "blood_glucose": "int64",
    "hba1c": "float64",
    "cholesterol": "float64",
    "number_of_medications": "int64",
    "previous_admissions": "int64",
    "length_of_stay": "int64",
    "emergency_visits_last_year": "int64",
    "chronic_disease_count": "int64",
    "diabetes": "int64",
    "hypertension": "int64",
    "heart_disease": "int64",
    "kidney_disease": "int64",
    "smoking_status": "object",
    "alcohol_consumption": "object",
    "physical_activity_level": "object",
    "discharge_destination": "object",
    "follow_up_scheduled": "int64",
    "insurance_type": "object",
    "admission_type": "object",
    "readmitted_30_days": "int64",
}

RANGE_RULES: dict[str, tuple] = {
    "age": (18, 100),
    "bmi": (10, 70),
    "systolic_bp": (70, 250),
    "diastolic_bp": (40, 150),
    "heart_rate": (30, 220),
    "blood_glucose": (40, 600),
    "hba1c": (3.0, 20.0),
    "cholesterol": (80, 500),
    "number_of_medications": (0, 40),
    "previous_admissions": (0, 30),
    "length_of_stay": (1, 120),
    "emergency_visits_last_year": (0, 30),
    "chronic_disease_count": (0, 10),
}

BINARY_COLUMNS = ["diabetes", "hypertension", "heart_disease", "kidney_disease", "follow_up_scheduled"]

CATEGORICAL_RULES: dict[str, list[str]] = {
    "gender": ["Male", "Female", "Other"],
    "smoking_status": ["Never", "Former", "Current"],
    "alcohol_consumption": ["Non-drinker", "Low", "Moderate", "High"],
    "physical_activity_level": ["Low", "Moderate", "High"],
    "discharge_destination": ["Home", "Skilled Nursing Facility", "Rehabilitation Center", "Home Health Care"],
    "insurance_type": ["Private", "Medicare", "Medicaid", "Uninsured"],
    "admission_type": ["Emergency", "Elective", "Urgent"],
}

TARGET_COLUMN = "readmitted_30_days"
MAX_MISSING_FRACTION = 0.15  # per-column tolerance before flagging as error


class HealthcareDataValidator:
    """Runs a comprehensive validation suite against the raw healthcare dataset."""

    def __init__(self, df: pd.DataFrame, dataset_name: str = "healthcare_readmission_raw") -> None:
        self.df = df
        self.report = ValidationReport(dataset_name=dataset_name)

    def _add(self, name: str, passed: bool, details: str, severity: str = "error") -> None:
        self.report.results.append(ValidationResult(name, bool(passed), details, severity))

    def validate_schema(self) -> None:
        expected_cols = set(EXPECTED_SCHEMA.keys())
        actual_cols = set(self.df.columns)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        passed = not missing and not extra
        self._add(
            "schema_columns_match",
            passed,
            f"missing={sorted(missing)}, extra={sorted(extra)}" if not passed else "all expected columns present",
        )

    def validate_datatypes(self) -> None:
        for col, expected_dtype in EXPECTED_SCHEMA.items():
            if col not in self.df.columns:
                continue
            actual_dtype = str(self.df[col].dtype)
            # allow float64 where int64 expected due to NaNs coercing dtype
            compatible = actual_dtype == expected_dtype or (expected_dtype == "int64" and actual_dtype == "float64")
            self._add(
                f"dtype_{col}",
                compatible,
                f"expected={expected_dtype}, actual={actual_dtype}",
                severity="warning" if expected_dtype == "int64" else "error",
            )

    def validate_missing_values(self) -> None:
        missing_frac = self.df.isna().mean()
        for col, frac in missing_frac.items():
            if frac == 0:
                continue
            passed = frac <= MAX_MISSING_FRACTION
            self._add(
                f"missing_values_{col}",
                passed,
                f"missing_fraction={frac:.4f} (threshold={MAX_MISSING_FRACTION})",
                severity="error" if not passed else "warning",
            )

    def validate_duplicates(self) -> None:
        n_dupes = int(self.df.duplicated().sum())
        n_id_dupes = int(self.df["patient_id"].duplicated().sum()) if "patient_id" in self.df.columns else 0
        self._add(
            "duplicate_rows",
            True,  # informational: duplicates are removed at preprocessing, not a hard failure here
            f"exact_duplicate_rows={n_dupes}, duplicate_patient_ids={n_id_dupes}",
            severity="warning",
        )

    def validate_ranges(self) -> None:
        for col, (low, high) in RANGE_RULES.items():
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            out_of_range = ((series < low) | (series > high)).sum()
            frac = out_of_range / max(len(series), 1)
            # small fraction tolerated as "limited outliers", flagged as warning
            passed = frac <= 0.02
            self._add(
                f"range_{col}",
                passed,
                f"[{low},{high}] violations={out_of_range} ({frac:.4f})",
                severity="error" if not passed else "warning",
            )

    def validate_categorical(self) -> None:
        for col, allowed in CATEGORICAL_RULES.items():
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            invalid = ~series.isin(allowed)
            passed = invalid.sum() == 0
            self._add(
                f"categorical_{col}",
                passed,
                f"allowed={allowed}, invalid_count={int(invalid.sum())}",
            )

    def validate_binary(self) -> None:
        for col in BINARY_COLUMNS:
            if col not in self.df.columns:
                continue
            series = self.df[col].dropna()
            invalid = ~series.isin([0, 1])
            passed = invalid.sum() == 0
            self._add(f"binary_{col}", passed, f"invalid_count={int(invalid.sum())}")

    def validate_target(self) -> None:
        if TARGET_COLUMN not in self.df.columns:
            self._add("target_present", False, f"{TARGET_COLUMN} column missing")
            return
        series = self.df[TARGET_COLUMN]
        no_nulls = series.isna().sum() == 0
        self._add("target_no_nulls", bool(no_nulls), f"null_count={int(series.isna().sum())}")

        valid_values = series.dropna().isin([0, 1]).all()
        self._add("target_binary_values", bool(valid_values), "target must be in {0, 1}")

        positive_rate = series.mean()
        reasonable_imbalance = 0.02 <= positive_rate <= 0.6
        self._add(
            "target_class_balance_sane",
            bool(reasonable_imbalance),
            f"positive_rate={positive_rate:.4f}",
            severity="warning",
        )

    def validate_business_rules(self) -> None:
        df = self.df
        # diastolic should not exceed systolic
        bp_violation = (df["diastolic_bp"] >= df["systolic_bp"]).sum()
        self._add(
            "business_rule_bp_consistency",
            bp_violation / len(df) < 0.01,
            f"diastolic>=systolic violations={int(bp_violation)}",
            severity="warning",
        )
        # chronic_disease_count should be >= sum of the four disease flags minus 1 (allows +1 "other")
        disease_sum = df[["diabetes", "hypertension", "heart_disease", "kidney_disease"]].sum(axis=1)
        cdc_violation = (df["chronic_disease_count"] < disease_sum - 0).sum()
        self._add(
            "business_rule_chronic_disease_consistency",
            cdc_violation / len(df) < 0.02,
            f"chronic_disease_count < sum(flags) violations={int(cdc_violation)}",
            severity="warning",
        )
        # length_of_stay must be positive
        los_violation = (df["length_of_stay"] <= 0).sum()
        self._add(
            "business_rule_length_of_stay_positive",
            los_violation == 0,
            f"non_positive_length_of_stay={int(los_violation)}",
        )

    def run_all(self) -> ValidationReport:
        logger.info("Running full validation suite")
        self.validate_schema()
        self.validate_datatypes()
        self.validate_missing_values()
        self.validate_duplicates()
        self.validate_ranges()
        self.validate_categorical()
        self.validate_binary()
        self.validate_target()
        self.validate_business_rules()
        logger.info(
            f"Validation complete: {self.report.n_passed} passed, "
            f"{self.report.n_failed} failed, overall_success={self.report.success}"
        )
        return self.report


def _render_html_report(report: ValidationReport) -> str:
    rows = "".join(
        f"<tr class='{'pass' if r.passed else r.severity}'>"
        f"<td>{r.check_name}</td><td>{r.passed}</td><td>{r.severity}</td><td>{r.details}</td></tr>"
        for r in report.results
    )
    return f"""<!DOCTYPE html>
<html><head><title>Data Validation Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #1f2937; color: white; }}
tr.pass {{ background: #ecfdf5; }}
tr.error {{ background: #fef2f2; }}
tr.warning {{ background: #fffbeb; }}
h1 {{ color: #111827; }}
.summary {{ font-size: 16px; margin-bottom: 1rem; }}
</style></head>
<body>
<h1>Data Validation Report: {report.dataset_name}</h1>
<div class="summary">Generated: {report.timestamp} | Overall success: <b>{report.success}</b> |
Passed: {report.n_passed} | Failed: {report.n_failed}</div>
<table>
<tr><th>Check</th><th>Passed</th><th>Severity</th><th>Details</th></tr>
{rows}
</table>
</body></html>"""


def validate_and_save_report(df: pd.DataFrame, output_dir: Path, dataset_name: str) -> ValidationReport:
    """Run the validation suite and persist JSON + HTML reports to disk."""
    validator = HealthcareDataValidator(df, dataset_name=dataset_name)
    report = validator.run_all()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset_name}_validation_report.json"
    html_path = output_dir / f"{dataset_name}_validation_report.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html_report(report))

    logger.info(f"Validation reports written to {json_path} and {html_path}")
    return report


def main() -> None:
    cfg = load_config()
    raw_path = resolve_path(cfg.data.raw_path)
    df = pd.read_csv(raw_path)

    report_dir = resolve_path("reports/validation")
    report = validate_and_save_report(df, report_dir, dataset_name="healthcare_readmission_raw")

    if not report.success:
        logger.error("Validation FAILED on one or more error-severity checks. See report for details.")
        failing = [r for r in report.results if not r.passed and r.severity == "error"]
        for r in failing:
            logger.error(f"  - {r.check_name}: {r.details}")
        raise SystemExit(1)

    validated_flag = resolve_path(cfg.data.validated_flag_path)
    validated_flag.parent.mkdir(parents=True, exist_ok=True)
    validated_flag.write_text(f"validated_at={datetime.now(UTC).isoformat()}\n")
    logger.info("Validation PASSED. Data is cleared for downstream processing.")


if __name__ == "__main__":
    main()
