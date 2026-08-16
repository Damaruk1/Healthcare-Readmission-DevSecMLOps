"""Unit tests for the synthetic data generator."""

from __future__ import annotations

import pandas as pd

from src.data_generation.generate_data import SyntheticHealthcareDataGenerator


class TestSyntheticHealthcareDataGenerator:
    def test_generate_returns_dataframe(self, synthetic_df: pd.DataFrame):
        assert isinstance(synthetic_df, pd.DataFrame)
        assert len(synthetic_df) > 0

    def test_expected_columns_present(self, synthetic_df: pd.DataFrame):
        expected = {
            "patient_id",
            "age",
            "gender",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "blood_glucose",
            "hba1c",
            "cholesterol",
            "number_of_medications",
            "previous_admissions",
            "length_of_stay",
            "emergency_visits_last_year",
            "chronic_disease_count",
            "diabetes",
            "hypertension",
            "heart_disease",
            "kidney_disease",
            "smoking_status",
            "alcohol_consumption",
            "physical_activity_level",
            "discharge_destination",
            "follow_up_scheduled",
            "insurance_type",
            "admission_type",
            "readmitted_30_days",
        }
        assert expected.issubset(set(synthetic_df.columns))

    def test_target_is_binary(self, synthetic_df: pd.DataFrame):
        assert set(synthetic_df["readmitted_30_days"].unique()).issubset({0, 1})

    def test_class_imbalance_present(self, synthetic_df: pd.DataFrame):
        positive_rate = synthetic_df["readmitted_30_days"].mean()
        assert 0.05 < positive_rate < 0.5, "expected a realistic minority-class imbalance"

    def test_missing_values_present(self, synthetic_df: pd.DataFrame):
        assert synthetic_df.isna().sum().sum() > 0

    def test_patient_id_mostly_unique(self, synthetic_df: pd.DataFrame):
        # duplicates are intentionally injected in small numbers
        dup_fraction = synthetic_df["patient_id"].duplicated().mean()
        assert dup_fraction < 0.05

    def test_age_within_expected_bounds(self, synthetic_df: pd.DataFrame):
        assert synthetic_df["age"].min() >= 18
        assert synthetic_df["age"].max() <= 100

    def test_binary_columns_are_binary(self, synthetic_df: pd.DataFrame):
        for col in ["diabetes", "hypertension", "heart_disease", "kidney_disease", "follow_up_scheduled"]:
            assert set(synthetic_df[col].dropna().unique()).issubset({0, 1})

    def test_higher_risk_factors_correlate_with_readmission(self, synthetic_df: pd.DataFrame):
        grouped = synthetic_df.groupby("readmitted_30_days")["previous_admissions"].mean()
        assert grouped.loc[1] > grouped.loc[0], "readmitted patients should have more prior admissions on average"

    def test_reproducible_with_same_seed(self):
        gen1 = SyntheticHealthcareDataGenerator(n_rows=50, random_seed=99)
        gen2 = SyntheticHealthcareDataGenerator(n_rows=50, random_seed=99)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)
