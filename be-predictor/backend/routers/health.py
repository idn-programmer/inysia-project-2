from fastapi import APIRouter
from ..services.ml_service import load_model

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/model-info")
def model_info():
    """Get information about the loaded ML model"""
    artifact = load_model()
    
    return {
        "model_loaded": artifact.model is not None,
        "model_version": artifact.version,
        "model_type": type(artifact.model).__name__ if artifact.model else "None",
        "optimal_threshold": artifact.optimal_threshold,
        "features_count": len(artifact.feature_names),
        "shap_available": artifact.shap_explainer is not None,
        "scaler_available": artifact.scaler is not None,
        "label_encoder_available": artifact.label_encoder is not None
    }


