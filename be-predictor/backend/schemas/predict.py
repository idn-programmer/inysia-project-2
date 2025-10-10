from __future__ import annotations

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = Field(default=None, pattern="^(Male|Female)$")
    pulseRate: Optional[int] = None
    sbp: Optional[int] = None
    dbp: Optional[int] = None
    glucose: Optional[float] = None  # mg/dL expected
    heightCm: Optional[float] = None
    weightKg: Optional[float] = None
    bmi: Optional[float] = None
    familyDiabetes: Optional[bool] = None
    hypertensive: Optional[bool] = None
    familyHypertension: Optional[bool] = None
    cardiovascular: Optional[bool] = None
    stroke: Optional[bool] = None
    userId: Optional[int] = None


class PredictResponse(BaseModel):
    risk: int
    model_version: str


class PredictionOut(BaseModel):
    id: str
    risk: int
    model_version: str
    created_at: str


