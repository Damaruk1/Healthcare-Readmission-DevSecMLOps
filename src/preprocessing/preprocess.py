"""Data preprocessing pipeline.

Handles duplicate removal, missing value imputation, outlier treatment
(IQR-based winsorization), categorical encoding, feature scaling, the
train/test split, and SMOTE class balancing on the training fold only.

The fitted preprocessing pipeline (ColumnTransformer) is persisted alongside
the trained model so inference-time transformation exactly matches training.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering.feature_engineer import HealthcareFeatureEngineer
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "patient_id"
TARGET_COLUMN = "readmitted_30_days"

BASE_NUMERIC_FEATURES = [
    "age",
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
]
BINARY_FEATURES = ["diabetes", "hypertension", "heart_disease", "kidney_disease", "follow_up_scheduled"]
BASE_CATEGORICAL_FEATURES = [
    "gender",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity_level",
    "discharge_destination",
    "insurance_type",
    "admission_type",
]
ENGINEERED_NUMERIC_FEATURES = ["hospital_utilization_score", "disease_burden_score", "readmission_risk_index"]
ENGINEERED_CATEGORICAL_FEATURES = ["bmi_category", "age_group", "glucose_category", "bp_category", "medication_burden"]


@dataclass
class PreprocessingArtifacts:
    pipeline: Pipeline
    feature_names: list[str]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COLUMN])
    df = df.drop_duplicates(subset=[ID_COLUMN])
    logger.info(f"Removed {before - len(df)} duplicate rows ({before} -> {len(df)})")
    return df.reset_index(drop=True)


def winsorize_outliers(df: pd.DataFrame, columns: list[str], multiplier: float = 1.5) -> pd.DataFrame:
    """Cap outliers to the IQR fence rather than dropping rows (preserves sample size)."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped:
            logger.info(f"Winsorized {n_capped} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")
    return df


def build_preprocessing_pipeline(cfg) -> ColumnTransformer:
    """Build a ColumnTransformer covering imputation + encoding + scaling for all feature groups."""
    numeric_features = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    categorical_features = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.preprocessing.numeric_impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.preprocessing.categorical_impute_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "numeric":
            names.extend(cols)
        elif name == "categorical":
            encoder: OneHotEncoder = trans.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(cols).tolist())
        elif name == "binary":
            names.extend(cols)
    return names


def run_preprocessing(cfg=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Full pipeline: load -> clean -> engineer -> split -> fit-transform -> SMOTE.

    Returns X_train, X_test, y_train, y_test (as numpy arrays) and output feature names.
    Also persists the fitted preprocessor and processed CSVs to disk.
    """
    cfg = cfg or load_config()
    raw_path = resolve_path(cfg.data.raw_path)
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data: {df.shape}")

    df = remove_duplicates(df)
    df = winsorize_outliers(df, BASE_NUMERIC_FEATURES, cfg.preprocessing.outlier_iqr_multiplier)

    engineer = HealthcareFeatureEngineer()
    df = engineer.fit_transform(df)

    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.project.random_seed, stratify=y
    )
    logger.info(f"Train/test split: train={X_train_raw.shape}, test={X_test_raw.shape}")

    preprocessor = build_preprocessing_pipeline(cfg)
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)
    feature_names = get_output_feature_names(preprocessor)

    logger.info(f"Class distribution before SMOTE: {y_train.value_counts(normalize=True).to_dict()}")
    smote = SMOTE(random_state=cfg.preprocessing.smote_random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_transformed, y_train)
    logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_res).value_counts(normalize=True).to_dict()}")

    # Persist artifacts
    preprocessor_path = resolve_path(cfg.training.preprocessor_output_path)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessor": preprocessor, "feature_engineer": engineer, "feature_names": feature_names}, preprocessor_path
    )
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")

    train_out = pd.DataFrame(X_train_res, columns=feature_names)
    train_out[TARGET_COLUMN] = y_train_res.values if hasattr(y_train_res, "values") else y_train_res
    test_out = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_out[TARGET_COLUMN] = y_test.values

    train_path = resolve_path(cfg.data.processed_train_path)
    test_path = resolve_path(cfg.data.processed_test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    logger.info(f"Saved processed train/test CSVs to {train_path} and {test_path}")

    return (
        np.asarray(X_train_res),
        np.asarray(X_test_transformed),
        np.asarray(y_train_res),
        np.asarray(y_test),
        feature_names,
    )


def main() -> None:
    run_preprocessing()


if __name__ == "__main__":
    main()
