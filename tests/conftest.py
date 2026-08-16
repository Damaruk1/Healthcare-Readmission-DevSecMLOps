"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generation.generate_data import SyntheticHealthcareDataGenerator  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A small, fast-to-generate synthetic dataset for tests."""
    generator = SyntheticHealthcareDataGenerator(n_rows=300, random_seed=7)
    return generator.generate()


@pytest.fixture()
def sample_patient_payload() -> dict:
    return {
        "age": 67,
        "gender": "Male",
        "bmi": 31.5,
        "systolic_bp": 145,
        "diastolic_bp": 92,
        "heart_rate": 88,
        "blood_glucose": 178,
        "hba1c": 8.2,
        "cholesterol": 235,
        "number_of_medications": 7,
        "previous_admissions": 3,
        "length_of_stay": 8,
        "emergency_visits_last_year": 2,
        "chronic_disease_count": 4,
        "diabetes": 1,
        "hypertension": 1,
        "heart_disease": 1,
        "kidney_disease": 0,
        "smoking_status": "Former",
        "alcohol_consumption": "Low",
        "physical_activity_level": "Low",
        "discharge_destination": "Home",
        "follow_up_scheduled": 0,
        "insurance_type": "Private",
        "admission_type": "Emergency",
    }


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
