from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os

import joblib

from ..config import get_settings


@dataclass
class ModelArtifact:
    pipeline: Any | None
    version: str


_artifact: ModelArtifact | None = None


def load_model() -> ModelArtifact:
    global _artifact
    if _artifact is not None:
        return _artifact
    settings = get_settings()
    model_path = settings.model_path
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        version = getattr(pipeline, "model_version", "v1")
        _artifact = ModelArtifact(pipeline=pipeline, version=version)
        return _artifact
    _artifact = ModelArtifact(pipeline=None, version="fallback")
    return _artifact


def _fallback_risk(features: Dict[str, Any]) -> int:
    glucose = float(features.get("glucose", 0) or 0)
    bmi = float(features.get("bmi", 0) or 0)
    hypertensive = bool(features.get("hypertensive") or False)
    family_diabetes = bool(features.get("familyDiabetes") or False)
    cardiovascular = bool(features.get("cardiovascular") or False)
    stroke = bool(features.get("stroke") or False)

    score = 0.0
    # glucose mg/dL reference ~90
    score += max(0.0, min(60.0, (glucose - 90.0) * 0.6))
    # BMI reference ~22
    score += max(0.0, min(25.0, (bmi - 22.0) * 1.2))
    if hypertensive:
        score += 10.0
    if family_diabetes:
        score += 10.0
    if cardiovascular:
        score += 8.0
    if stroke:
        score += 8.0
    risk = int(round(max(0.0, min(100.0, score))))
    return risk


def predict(features: Dict[str, Any]) -> Tuple[int, str]:
    artifact = load_model()
    if artifact.pipeline is None:
        return _fallback_risk(features), artifact.version
    # Expect pipeline to handle dict -> vector via a ColumnTransformer in training
    prob = artifact.pipeline.predict_proba([features])[0][1]
    risk = int(round(prob * 100))
    return risk, getattr(artifact.pipeline, "model_version", artifact.version)


