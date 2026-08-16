"""Synthetic healthcare readmission dataset generator.

Generates a fully synthetic (no real patient data) dataset that models
realistic relationships between clinical/demographic features and 30-day
readmission risk. Includes missing values, class imbalance, limited outliers
and mixed categorical features by design, as required for a realistic MLOps
pipeline demo.

DISCLAIMER: This data is entirely synthetic and randomly generated. It must
never be used for real clinical decision-making.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticHealthcareDataGenerator:
    """Generates a synthetic patient readmission dataset with realistic risk structure."""

    def __init__(self, n_rows: int = 5000, random_seed: int = 42) -> None:
        self.n_rows = n_rows
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def _generate_demographics(self) -> pd.DataFrame:
        age = np.clip(self.rng.normal(58, 18, self.n_rows), 18, 95).round().astype(int)
        gender = self.rng.choice(["Male", "Female", "Other"], self.n_rows, p=[0.48, 0.49, 0.03])
        bmi = np.clip(self.rng.normal(28, 6, self.n_rows), 14, 55).round(1)
        return pd.DataFrame({"age": age, "gender": gender, "bmi": bmi})

    def _generate_vitals_and_labs(self, age: np.ndarray, bmi: np.ndarray) -> pd.DataFrame:
        age_factor = (age - 18) / (95 - 18)
        bmi_factor = (bmi - 14) / (55 - 14)

        systolic_bp = (
            np.clip(self.rng.normal(120 + 25 * age_factor + 10 * bmi_factor, 15, self.n_rows), 85, 220)
            .round()
            .astype(int)
        )
        diastolic_bp = (
            np.clip(self.rng.normal(75 + 10 * age_factor + 6 * bmi_factor, 10, self.n_rows), 50, 130)
            .round()
            .astype(int)
        )
        heart_rate = np.clip(self.rng.normal(78, 13, self.n_rows), 45, 160).round().astype(int)
        blood_glucose = np.clip(self.rng.normal(110 + 40 * bmi_factor, 35, self.n_rows), 60, 400).round().astype(int)
        hba1c = np.clip(self.rng.normal(5.7 + 2.0 * bmi_factor, 1.1, self.n_rows), 4.0, 14.0).round(1)
        cholesterol = np.clip(self.rng.normal(195 + 25 * bmi_factor, 35, self.n_rows), 100, 350).round().astype(int)

        return pd.DataFrame(
            {
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
                "blood_glucose": blood_glucose,
                "hba1c": hba1c,
                "cholesterol": cholesterol,
            }
        )

    def _generate_clinical_history(self, age: np.ndarray) -> pd.DataFrame:
        age_factor = (age - 18) / (95 - 18)

        number_of_medications = np.clip(self.rng.poisson(2 + 6 * age_factor, self.n_rows), 0, 20)
        previous_admissions = np.clip(self.rng.poisson(0.5 + 2.5 * age_factor, self.n_rows), 0, 12)
        length_of_stay = np.clip(self.rng.poisson(2 + 4 * age_factor, self.n_rows), 1, 30)
        emergency_visits_last_year = np.clip(self.rng.poisson(0.3 + 1.7 * age_factor, self.n_rows), 0, 10)

        diabetes = self.rng.binomial(1, np.clip(0.08 + 0.35 * age_factor, 0, 0.9), self.n_rows)
        hypertension = self.rng.binomial(1, np.clip(0.10 + 0.5 * age_factor, 0, 0.9), self.n_rows)
        heart_disease = self.rng.binomial(1, np.clip(0.05 + 0.4 * age_factor, 0, 0.85), self.n_rows)
        kidney_disease = self.rng.binomial(1, np.clip(0.03 + 0.25 * age_factor, 0, 0.7), self.n_rows)

        chronic_disease_count = diabetes + hypertension + heart_disease + kidney_disease
        # occasionally patients have chronic conditions beyond the flagged four
        chronic_disease_count = chronic_disease_count + self.rng.binomial(1, 0.1, self.n_rows)

        return pd.DataFrame(
            {
                "number_of_medications": number_of_medications,
                "previous_admissions": previous_admissions,
                "length_of_stay": length_of_stay,
                "emergency_visits_last_year": emergency_visits_last_year,
                "chronic_disease_count": chronic_disease_count,
                "diabetes": diabetes,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "kidney_disease": kidney_disease,
            }
        )

    def _generate_lifestyle_and_admin(self) -> pd.DataFrame:
        smoking_status = self.rng.choice(["Never", "Former", "Current"], self.n_rows, p=[0.55, 0.30, 0.15])
        alcohol_consumption = self.rng.choice(
            ["Non-drinker", "Low", "Moderate", "High"], self.n_rows, p=[0.40, 0.35, 0.18, 0.07]
        )
        physical_activity_level = self.rng.choice(["Low", "Moderate", "High"], self.n_rows, p=[0.45, 0.38, 0.17])
        discharge_destination = self.rng.choice(
            ["Home", "Skilled Nursing Facility", "Rehabilitation Center", "Home Health Care"],
            self.n_rows,
            p=[0.65, 0.12, 0.10, 0.13],
        )
        follow_up_scheduled = self.rng.binomial(1, 0.68, self.n_rows)
        insurance_type = self.rng.choice(
            ["Private", "Medicare", "Medicaid", "Uninsured"], self.n_rows, p=[0.42, 0.33, 0.18, 0.07]
        )
        admission_type = self.rng.choice(["Emergency", "Elective", "Urgent"], self.n_rows, p=[0.55, 0.25, 0.20])
        return pd.DataFrame(
            {
                "smoking_status": smoking_status,
                "alcohol_consumption": alcohol_consumption,
                "physical_activity_level": physical_activity_level,
                "discharge_destination": discharge_destination,
                "follow_up_scheduled": follow_up_scheduled,
                "insurance_type": insurance_type,
                "admission_type": admission_type,
            }
        )

    def _compute_readmission_target(self, df: pd.DataFrame) -> np.ndarray:
        """Compute readmission probability via a weighted clinical risk score + noise."""
        risk_score = (
            0.015 * (df["age"] - 50)
            + 0.020 * (df["blood_glucose"] - 110)
            + 0.35 * (df["hba1c"] - 5.7)
            + 0.55 * df["chronic_disease_count"]
            + 0.60 * df["previous_admissions"]
            + 0.18 * df["length_of_stay"]
            + 0.45 * df["emergency_visits_last_year"]
            + 0.12 * df["number_of_medications"]
            + 0.9 * (1 - df["follow_up_scheduled"])
            + 0.4 * df["diabetes"]
            + 0.3 * df["heart_disease"]
            + 0.3 * df["kidney_disease"]
        )
        # standardize then squash through a logistic function, centered to
        # produce a realistic minority-class imbalance (~15-20% positive rate)
        z = (risk_score - risk_score.mean()) / risk_score.std()
        prob = 1 / (1 + np.exp(-(1.15 * z - 1.55)))
        prob = np.clip(prob, 0.01, 0.97)
        target = self.rng.binomial(1, prob)
        return target

    def _inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce realistic missingness (MCAR-ish) in a subset of columns."""
        missing_cols = {
            "bmi": 0.03,
            "cholesterol": 0.05,
            "hba1c": 0.07,
            "alcohol_consumption": 0.04,
            "physical_activity_level": 0.06,
            "insurance_type": 0.02,
        }
        for col, frac in missing_cols.items():
            mask = self.rng.random(self.n_rows) < frac
            df.loc[mask, col] = np.nan
        return df

    def _inject_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Introduce a small number of extreme-but-plausible outliers."""
        n_outliers = max(1, int(self.n_rows * 0.01))
        idx = self.rng.choice(self.n_rows, n_outliers, replace=False)
        df.loc[idx, "blood_glucose"] = self.rng.integers(380, 450, n_outliers)
        df.loc[idx[: n_outliers // 2], "length_of_stay"] = self.rng.integers(45, 60, len(idx[: n_outliers // 2]))
        return df

    def _inject_duplicates(self, df: pd.DataFrame, n_duplicates: int = 15) -> pd.DataFrame:
        dup_rows = df.sample(n=n_duplicates, random_state=self.random_seed)
        return pd.concat([df, dup_rows], ignore_index=True)

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic dataset and return it as a DataFrame."""
        logger.info(f"Generating {self.n_rows} synthetic patient records")

        demo = self._generate_demographics()
        vitals = self._generate_vitals_and_labs(demo["age"].values, demo["bmi"].values)
        history = self._generate_clinical_history(demo["age"].values)
        lifestyle = self._generate_lifestyle_and_admin()

        df = pd.concat([demo, vitals, history, lifestyle], axis=1)
        df.insert(0, "patient_id", [f"PT{100000 + i}" for i in range(self.n_rows)])

        target = self._compute_readmission_target(df)
        df["readmitted_30_days"] = target

        df = self._inject_outliers(df)
        df = self._inject_missing_values(df)
        df = self._inject_duplicates(df)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        positive_rate = df["readmitted_30_days"].mean()
        logger.info(f"Generated dataset shape={df.shape}, positive_rate={positive_rate:.3f}")
        return df


def main() -> None:
    cfg = load_config()
    generator = SyntheticHealthcareDataGenerator(n_rows=cfg.data.n_rows, random_seed=cfg.project.random_seed)
    df = generator.generate()

    output_path = resolve_path(cfg.data.raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved synthetic dataset to {output_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
