"""Tests for the feature engineering module."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.feature_engineer import HealthcareFeatureEngineer


class TestHealthcareFeatureEngineer:
    def test_transform_adds_expected_columns(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        for col in [
            "bmi_category",
            "age_group",
            "glucose_category",
            "bp_category",
            "hospital_utilization_score",
            "disease_burden_score",
            "medication_burden",
            "readmission_risk_index",
        ]:
            assert col in out.columns

    def test_original_columns_preserved(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        assert set(synthetic_df.columns).issubset(set(out.columns))

    def test_readmission_risk_index_bounded(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        assert out["readmission_risk_index"].min() >= 0
        assert out["readmission_risk_index"].max() <= 100

    def test_bmi_category_values_valid(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        valid_categories = {"Underweight", "Normal", "Overweight", "Obese_I", "Obese_II_Plus"}
        observed = set(out["bmi_category"].dropna().astype(str).unique())
        assert observed.issubset(valid_categories)

    def test_hospital_utilization_score_non_negative(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        assert (out["hospital_utilization_score"] >= 0).all()

    def test_row_count_unchanged(self, synthetic_df: pd.DataFrame):
        engineer = HealthcareFeatureEngineer()
        out = engineer.fit_transform(synthetic_df)
        assert len(out) == len(synthetic_df)
