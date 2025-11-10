from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os
import json
import pickle

import joblib
import pandas as pd
import numpy as np

from ..config import get_settings


@dataclass
class ModelArtifact:
    model: Any | None
    scaler: Any | None
    label_encoder: Any | None
    shap_explainer: Any | None
    feature_names: list[str]
    optimal_threshold: float
    version: str


_artifact: ModelArtifact | None = None


def load_model() -> ModelArtifact:
    global _artifact
    if _artifact is not None:
        return _artifact
    
    settings = get_settings()
    model_dir = os.path.dirname(settings.model_path)
    
    # Try to load LightGBM model and all artifacts
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    explainer_path = os.path.join(model_dir, "shap_explainer.pkl")
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    threshold_path = os.path.join(model_dir, "optimal_threshold.json")
    
    if os.path.exists(model_path):
        try:
            model = pickle.load(open(model_path, 'rb'))
            scaler = pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) else None
            label_encoder = pickle.load(open(le_path, 'rb')) if os.path.exists(le_path) else None
            shap_explainer = pickle.load(open(explainer_path, 'rb')) if os.path.exists(explainer_path) else None
            
            feature_names = []
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
            
            # Load optimal threshold
            optimal_threshold = 0.5  # Default threshold
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    optimal_threshold = threshold_data.get("threshold", 0.5)
            
            _artifact = ModelArtifact(
                model=model,
                scaler=scaler,
                label_encoder=label_encoder,
                shap_explainer=shap_explainer,
                feature_names=feature_names,
                optimal_threshold=optimal_threshold,
                version="v2.0.1-lightgbm-optimized-20251110"
            )
            print(f"✓ Loaded LightGBM optimized model with threshold {optimal_threshold:.4f}")
            return _artifact
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Fallback
    _artifact = ModelArtifact(
        model=None,
        scaler=None,
        label_encoder=None,
        shap_explainer=None,
        feature_names=[],
        optimal_threshold=0.5,
        version="fallback"
    )
    print("⚠ Using fallback predictor (model not trained yet)")
    return _artifact


def _prepare_features(features: Dict[str, Any], artifact: ModelArtifact) -> pd.DataFrame:
    """Prepare features for prediction - convert from API format to model format."""
    # Map API field names to dataset column names
    feature_mapping = {
        'age': 'age',
        'gender': 'gender',
        'pulseRate': 'pulse_rate',
        'sbp': 'systolic_bp',
        'dbp': 'diastolic_bp',
        'glucose': 'glucose',
        'heightCm': 'height',
        'weightKg': 'weight',
        'bmi': 'bmi',
        'familyDiabetes': 'family_diabetes',
        'hypertensive': 'hypertensive',
        'familyHypertension': 'family_hypertension',
        'cardiovascular': 'cardiovascular_disease',
        'stroke': 'stroke'
        }
    
    # Convert to DataFrame with proper column names
    row_data = {}
    for api_key, dataset_key in feature_mapping.items():
        value = features.get(api_key)
        # Convert booleans to int (0 or 1)
        if isinstance(value, bool):
            value = int(value)
        row_data[dataset_key] = value
    
    df = pd.DataFrame([row_data])
    
    # Ensure we have all expected features
    if artifact.feature_names:
        for feat in artifact.feature_names:
            if feat not in df.columns:
                df[feat] = 0  # Default value for missing features
        # Reorder columns to match training
        df = df[artifact.feature_names]
    
    # Encode gender if label encoder exists
    if artifact.label_encoder is not None and 'gender' in df.columns:
        try:
            df['gender'] = artifact.label_encoder.transform(df['gender'].fillna('Male'))
        except:
            # If transform fails, use 0 for Male, 1 for Female
            df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    
    # Fill NaN values with median/mean
    df = df.fillna(df.median(numeric_only=True))
    
    # Scale features if scaler exists
    if artifact.scaler is not None:
        df[artifact.feature_names] = artifact.scaler.transform(df[artifact.feature_names])
    
    return df


def _fallback_risk(features: Dict[str, Any]) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """Fallback risk calculation when model isn't loaded."""
    glucose = float(features.get("glucose", 0) or 0)
    bmi = float(features.get("bmi", 0) or 0)
    hypertensive = bool(features.get("hypertensive") or False)
    family_diabetes = bool(features.get("familyDiabetes") or False)
    cardiovascular = bool(features.get("cardiovascular") or False)
    stroke = bool(features.get("stroke") or False)

    score = 0.0
    shap_contributions = {}
    
    # glucose mg/dL reference ~90
    glucose_contrib = max(0.0, min(60.0, (glucose - 90.0) * 0.6))
    score += glucose_contrib
    shap_contributions["glucose"] = glucose_contrib
    
    # BMI reference ~22
    bmi_contrib = max(0.0, min(25.0, (bmi - 22.0) * 1.2))
    score += bmi_contrib
    shap_contributions["bmi"] = bmi_contrib
    
    if hypertensive:
        score += 10.0
        shap_contributions["hypertensive"] = 10.0
    if family_diabetes:
        score += 10.0
        shap_contributions["familyDiabetes"] = 10.0
    if cardiovascular:
        score += 8.0
        shap_contributions["cardiovascular"] = 8.0
    if stroke:
        score += 8.0
        shap_contributions["stroke"] = 8.0
    
    risk = int(round(max(0.0, min(100.0, score))))
    
    # Simple global importance for fallback
    global_importance = {
        "glucose": 0.25,
        "bmi": 0.20,
        "familyDiabetes": 0.15,
        "hypertensive": 0.15,
        "cardiovascular": 0.12,
        "stroke": 0.08,
        "age": 0.05
    }
    
    return risk, shap_contributions, global_importance


def get_global_feature_importance(artifact: ModelArtifact) -> Dict[str, float]:
    """Get global feature importance from the model."""
    if artifact.model is None or not hasattr(artifact.model, 'feature_importances_'):
        return {}
    
    importances = artifact.model.feature_importances_
    feature_names = artifact.feature_names if artifact.feature_names else [f"feature_{i}" for i in range(len(importances))]
    
    # Convert to dict and normalize to percentages
    importance_dict = dict(zip(feature_names, importances * 100))
    return importance_dict


def predict(features: Dict[str, Any]) -> Tuple[int, str, Dict[str, float], Dict[str, float]]:
    """
    Predict diabetes risk and return SHAP values.
    
    Returns:
        Tuple of (risk_score, model_version, shap_values, global_importance)
    """
    artifact = load_model()
    
    if artifact.model is None:
        risk, shap_values, global_importance = _fallback_risk(features)
        return risk, artifact.version, shap_values, global_importance
    
    try:
        # Prepare features
        df = _prepare_features(features, artifact)
        
        # Get prediction probability
        prob = artifact.model.predict_proba(df)[0][1]
        
        # Use optimal threshold for binary classification
        binary_prediction = 1 if prob >= artifact.optimal_threshold else 0
        
        # Convert probability to risk percentage with threshold scaling
        if prob >= artifact.optimal_threshold:
            # Scale from threshold to 100%
            risk = int(round((prob - artifact.optimal_threshold) / (1 - artifact.optimal_threshold) * 100))
        else:
            # Scale from 0% to threshold
            risk = int(round(prob / artifact.optimal_threshold * 50))  # Scale to 0-50%
        
        # Ensure risk is within 0-100 range
        risk = max(0, min(100, risk))
        
        # Calculate SHAP values
        shap_values = {}
        if artifact.shap_explainer is not None:
            try:
                shap_vals = artifact.shap_explainer.shap_values(df)
                # For binary classification, shap_values might be a list [class0, class1]
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Use class 1 (diabetic) values
                
                # Convert to numpy array and flatten if needed
                shap_vals = np.array(shap_vals)
                if shap_vals.ndim > 1:
                    shap_vals = shap_vals.flatten()
                
                # Convert to dict with readable names
                feature_names = artifact.feature_names if artifact.feature_names else df.columns.tolist()
                
                # Map back to API field names for consistency
                reverse_mapping = {
                    'age': 'age',
                    'gender': 'gender',
                    'pulse_rate': 'pulseRate',
                    'systolic_bp': 'sbp',
                    'diastolic_bp': 'dbp',
                    'glucose': 'glucose',
                    'height': 'heightCm',
                    'weight': 'weightKg',
                    'bmi': 'bmi',
                    'family_diabetes': 'familyDiabetes',
                    'hypertensive': 'hypertensive',
                    'family_hypertension': 'familyHypertension',
                    'cardiovascular_disease': 'cardiovascular',
                    'cardiovascular': 'cardiovascular',
                    'stroke': 'stroke'
                }
                
                for i, feat_name in enumerate(feature_names):
                    if i < len(shap_vals):
                        api_name = reverse_mapping.get(feat_name, feat_name)
                        shap_values[api_name] = float(shap_vals[i])
            except Exception as e:
                print(f"SHAP calculation error: {e}")
                import traceback
                traceback.print_exc()
                # Return empty dict - Pydantic expects Dict[str, float]
                shap_values = {}
        
        # Get global feature importance
        global_importance = get_global_feature_importance(artifact)
        
        # Add threshold information to global importance for debugging
        global_importance["_threshold_used"] = artifact.optimal_threshold
        global_importance["_binary_prediction"] = binary_prediction
        
        return risk, artifact.version, shap_values, global_importance
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fall back to simple prediction
        risk, shap_values, global_importance = _fallback_risk(features)
        return risk, "fallback", shap_values, global_importance
