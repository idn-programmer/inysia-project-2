from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List, Optional

from ..schemas.predict import PredictRequest, PredictResponse, PredictionOut
from ..schemas.common import HistoryQuery
from ..services.ml_service import predict as ml_predict
from ..db.session import get_db
from ..db import models as orm
from .auth import get_current_user


router = APIRouter()


def _normalize(req: PredictRequest) -> dict:
    features = {
        "age": req.age,
        "gender": req.gender,
        "pulse_rate": req.pulseRate,
        "systolic_bp": req.sbp,
        "diastolic_bp": req.dbp,
        "glucose": req.glucose,
        "height_cm": req.heightCm,
        "weight_kg": req.weightKg,
        "bmi": req.bmi,
        "familyDiabetes": req.familyDiabetes,
        "hypertensive": req.hypertensive,
        "familyHypertension": req.familyHypertension,
        "cardiovascular": req.cardiovascular,
        "stroke": req.stroke,
    }
    # Compute BMI if not provided but height/weight exist
    if (req.bmi is None or req.bmi == 0) and req.heightCm and req.weightKg:
        m = float(req.heightCm) / 100.0
        if m > 0:
            features["bmi"] = round(float(req.weightKg) / (m * m), 1)
    return features


def get_token_from_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract token from Authorization header."""
    if not authorization:
        return None
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        return token
    except ValueError:
        return None


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest, 
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    features = _normalize(payload)
    risk, version = ml_predict(features)

    # Get current user if token is provided
    current_user = None
    token = get_token_from_header(authorization)
    if token:
        try:
            current_user = get_current_user(token, db)
        except HTTPException:
            # If token is invalid, continue without user
            pass

    # persist
    pred = orm.Prediction(
        user_id=current_user.id if current_user else 1,  # Default to user_id 1 if no auth
        age=payload.age,
        gender=payload.gender,
        pulse_rate=payload.pulseRate,
        systolic_bp=payload.sbp,
        diastolic_bp=payload.dbp,
        glucose=payload.glucose,
        height=payload.heightCm,
        weight=payload.weightKg,
        bmi=features.get("bmi"),
        family_diabetes=payload.familyDiabetes or False,
        hypertensive=payload.hypertensive or False,
        family_hypertension=payload.familyHypertension or False,
        cardiovascular_disease=payload.cardiovascular or False,
        stroke=payload.stroke or False,
        risk_score=risk,
    )
    db.add(pred)
    db.commit()
    return PredictResponse(risk=risk, model_version=version)


@router.get("/history", response_model=List[PredictionOut])
def history(
    user_id: int | None = None,
    limit: int = 50,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    q = db.query(orm.Prediction).order_by(orm.Prediction.created_at.desc())
    # If auth header present, restrict to that user
    token = get_token_from_header(authorization)
    if token:
        try:
            current_user = get_current_user(token, db)
            q = q.filter(orm.Prediction.user_id == current_user.id)
        except HTTPException:
            pass
    elif user_id:
        q = q.filter(orm.Prediction.user_id == user_id)
    rows = q.limit(limit).all()
    return [
        PredictionOut(
            id=str(r.id),
            risk=int(r.risk_score or 0),
            model_version="v1.0.0",
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


