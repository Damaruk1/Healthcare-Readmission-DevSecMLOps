"""Pydantic request/response schemas for the Readmission Risk API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class SmokingStatusEnum(str, Enum):
    never = "Never"
    former = "Former"
    current = "Current"


class AlcoholConsumptionEnum(str, Enum):
    non_drinker = "Non-drinker"
    low = "Low"
    moderate = "Moderate"
    high = "High"


class PhysicalActivityEnum(str, Enum):
    low = "Low"
    moderate = "Moderate"
    high = "High"


class DischargeDestinationEnum(str, Enum):
    home = "Home"
    snf = "Skilled Nursing Facility"
    rehab = "Rehabilitation Center"
    home_health = "Home Health Care"


class InsuranceTypeEnum(str, Enum):
    private = "Private"
    medicare = "Medicare"
    medicaid = "Medicaid"
    uninsured = "Uninsured"


class AdmissionTypeEnum(str, Enum):
    emergency = "Emergency"
    elective = "Elective"
    urgent = "Urgent"


class RiskLevel(str, Enum):
    low = "Low"
    moderate = "Moderate"
    high = "High"
    very_high = "Very High"


class PatientFeatures(BaseModel):
    """Single patient's clinical/demographic input features for prediction."""

    age: int = Field(..., ge=18, le=100, description="Patient age in years")
    gender: GenderEnum
    bmi: float = Field(..., ge=10, le=70)
    systolic_bp: int = Field(..., ge=70, le=250)
    diastolic_bp: int = Field(..., ge=40, le=150)
    heart_rate: int = Field(..., ge=30, le=220)
    blood_glucose: int = Field(..., ge=40, le=600)
    hba1c: float = Field(..., ge=3.0, le=20.0)
    cholesterol: float = Field(..., ge=80, le=500)
    number_of_medications: int = Field(..., ge=0, le=40)
    previous_admissions: int = Field(..., ge=0, le=30)
    length_of_stay: int = Field(..., ge=1, le=120)
    emergency_visits_last_year: int = Field(..., ge=0, le=30)
    chronic_disease_count: int = Field(..., ge=0, le=10)
    diabetes: int = Field(..., ge=0, le=1)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    kidney_disease: int = Field(..., ge=0, le=1)
    smoking_status: SmokingStatusEnum
    alcohol_consumption: AlcoholConsumptionEnum
    physical_activity_level: PhysicalActivityEnum
    discharge_destination: DischargeDestinationEnum
    follow_up_scheduled: int = Field(..., ge=0, le=1)
    insurance_type: InsuranceTypeEnum
    admission_type: AdmissionTypeEnum

    @field_validator("diastolic_bp")
    @classmethod
    def diastolic_lower_than_systolic(cls, v: int, info) -> int:
        systolic = info.data.get("systolic_bp")
        if systolic is not None and v >= systolic:
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    risk_level: RiskLevel
    model_version: str
    timestamp: datetime


class BatchPredictionRow(BaseModel):
    patient_id: str
    prediction: int
    prediction_label: str
    probability: float
    risk_level: RiskLevel
    model_version: str
    prediction_timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: str | None
    test_roc_auc: float | None
    all_model_results: dict | None
    feature_count: int | None


class VersionResponse(BaseModel):
    api_version: str
    model_version: str
    project_name: str


class RetrainResponse(BaseModel):
    status: str
    message: str
    triggered_at: datetime


class DriftReportResponse(BaseModel):
    drift_detected: bool
    report_available: bool
    report_path: str | None
    generated_at: str | None
    message: str
