"""Feature engineering module.

Derives clinically-motivated composite features (BMI category, age group,
hospital utilization score, disease burden, medication burden, glucose
category, blood pressure category and an aggregate readmission risk index)
from the raw/cleaned dataset. Implemented as a scikit-learn compatible
transformer so it can be composed inside a Pipeline / ColumnTransformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthcareFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived clinical-risk features to a healthcare readmission DataFrame.

    This transformer is stateless (no fitting required beyond validating
    input columns) so `fit` is a no-op, making it safe to place before the
    imputation/scaling steps of the preprocessing pipeline.
    """

    def __init__(self) -> None:
        self.engineered_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> HealthcareFeatureEngineer:
        return self

    @staticmethod
    def _bmi_category(bmi: pd.Series) -> pd.Series:
        bins = [0, 18.5, 25, 30, 35, np.inf]
        labels = ["Underweight", "Normal", "Overweight", "Obese_I", "Obese_II_Plus"]
        return pd.cut(bmi, bins=bins, labels=labels)

    @staticmethod
    def _age_group(age: pd.Series) -> pd.Series:
        bins = [0, 30, 45, 60, 75, np.inf]
        labels = ["18-30", "31-45", "46-60", "61-75", "76+"]
        return pd.cut(age, bins=bins, labels=labels)

    @staticmethod
    def _glucose_category(glucose: pd.Series) -> pd.Series:
        bins = [0, 100, 125, 200, np.inf]
        labels = ["Normal", "Prediabetic", "Diabetic", "Severely_High"]
        return pd.cut(glucose, bins=bins, labels=labels)

    @staticmethod
    def _bp_category(systolic: pd.Series, diastolic: pd.Series) -> pd.Series:
        conditions = [
            (systolic < 120) & (diastolic < 80),
            (systolic < 130) & (diastolic < 80),
            (systolic < 140) | (diastolic < 90),
            (systolic >= 140) | (diastolic >= 90),
        ]
        labels = ["Normal", "Elevated", "Hypertension_Stage1", "Hypertension_Stage2"]
        return pd.Series(np.select(conditions, labels, default="Hypertension_Stage2"), index=systolic.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["bmi_category"] = self._bmi_category(df["bmi"])
        df["age_group"] = self._age_group(df["age"])
        df["glucose_category"] = self._glucose_category(df["blood_glucose"])
        df["bp_category"] = self._bp_category(df["systolic_bp"], df["diastolic_bp"])

        # Hospital Utilization Score: weighted combination of historical utilization signals
        df["hospital_utilization_score"] = (
            2.0 * df["previous_admissions"] + 1.5 * df["emergency_visits_last_year"] + 0.5 * df["length_of_stay"]
        )

        # Disease Burden Score: count + severity-weighted chronic conditions
        df["disease_burden_score"] = (
            df["chronic_disease_count"]
            + 1.5 * df["diabetes"]
            + 1.5 * df["heart_disease"]
            + 1.2 * df["kidney_disease"]
            + 1.0 * df["hypertension"]
        )

        # Medication Burden: polypharmacy proxy
        df["medication_burden"] = pd.cut(
            df["number_of_medications"],
            bins=[-1, 2, 5, 9, np.inf],
            labels=["Low", "Moderate", "High", "Polypharmacy"],
        )

        # Readmission Risk Index: aggregate normalized composite score (0-100 scale, heuristic)
        norm_prev_adm = df["previous_admissions"].clip(0, 10) / 10
        norm_er_visits = df["emergency_visits_last_year"].clip(0, 10) / 10
        norm_disease_burden = df["disease_burden_score"].clip(0, 10) / 10
        norm_meds = df["number_of_medications"].clip(0, 20) / 20
        no_follow_up_penalty = (1 - df["follow_up_scheduled"]).astype(float)

        df["readmission_risk_index"] = (
            100
            * (
                0.30 * norm_prev_adm
                + 0.20 * norm_er_visits
                + 0.30 * norm_disease_burden
                + 0.10 * norm_meds
                + 0.10 * no_follow_up_penalty
            )
        ).round(2)

        self.engineered_columns_ = [
            "bmi_category",
            "age_group",
            "glucose_category",
            "bp_category",
            "hospital_utilization_score",
            "disease_burden_score",
            "medication_burden",
            "readmission_risk_index",
        ]
        logger.info(f"Feature engineering added columns: {self.engineered_columns_}")
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features or []) + self.engineered_columns_)
