"""
Test Model Production - Validasi model dengan data yang sama seperti training
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from backend.services.ml_service import predict, load_model

def test_model_with_training_data():
    """Test model dengan data training yang sama"""
    
    print("🔍 Testing Model Production dengan Data Training")
    print("="*60)
    
    # Load training data
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "Data" / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    
    print(f"📊 Loading training data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
    df = df[df['age'] >= 60]
    
    print(f"✓ Dataset loaded: {len(df)} records")
    print(f"✓ Diabetic distribution: {df['diabetic'].value_counts().to_dict()}")
    
    # Load model artifact
    print("\n🤖 Loading model artifact...")
    artifact = load_model()
    print(f"✓ Model loaded: {artifact.model is not None}")
    print(f"✓ Model type: {type(artifact.model).__name__ if artifact.model else 'None'}")
    print(f"✓ Version: {artifact.version}")
    print(f"✓ Optimal threshold: {artifact.optimal_threshold}")
    print(f"✓ Features: {artifact.feature_names}")
    
    if artifact.model is None:
        print("❌ Model tidak ter-load!")
        return
    
    # Test dengan beberapa sample
    print("\n🧪 Testing dengan sample data...")
    
    # Ambil 5 sample diabetes dan 5 sample non-diabetes
    diabetes_samples = df[df['diabetic'] == 1].head(5)
    non_diabetes_samples = df[df['diabetic'] == 0].head(5)
    
    print(f"\n📋 Testing {len(diabetes_samples)} diabetes samples:")
    for idx, row in diabetes_samples.iterrows():
        # Convert to API format
        api_data = {
            'age': int(row['age']),
            'gender': 'Male' if row['gender'] == 'Male' else 'Female',
            'pulseRate': float(row['pulse_rate']),
            'sbp': float(row['systolic_bp']),
            'dbp': float(row['diastolic_bp']),
            'glucose': float(row['glucose']),
            'heightCm': float(row['height']),
            'weightKg': float(row['weight']),
            'bmi': float(row['bmi']),
            'familyDiabetes': bool(row['family_diabetes']),
            'hypertensive': bool(row['hypertensive']),
            'familyHypertension': bool(row['family_hypertension']),
            'cardiovascular': bool(row['cardiovascular_disease']),
            'stroke': bool(row['stroke'])
        }
        
        # Get prediction
        risk, version, shap_values, global_importance = predict(api_data)
        
        print(f"  Sample {idx}: Actual={row['diabetic']}, Predicted Risk={risk}%, Threshold={artifact.optimal_threshold:.4f}")
        print(f"    Features: glucose={row['glucose']}, bmi={row['bmi']:.1f}, hypertensive={row['hypertensive']}")
    
    print(f"\n📋 Testing {len(non_diabetes_samples)} non-diabetes samples:")
    for idx, row in non_diabetes_samples.iterrows():
        # Convert to API format
        api_data = {
            'age': int(row['age']),
            'gender': 'Male' if row['gender'] == 'Male' else 'Female',
            'pulseRate': float(row['pulse_rate']),
            'sbp': float(row['systolic_bp']),
            'dbp': float(row['diastolic_bp']),
            'glucose': float(row['glucose']),
            'heightCm': float(row['height']),
            'weightKg': float(row['weight']),
            'bmi': float(row['bmi']),
            'familyDiabetes': bool(row['family_diabetes']),
            'hypertensive': bool(row['hypertensive']),
            'familyHypertension': bool(row['family_hypertension']),
            'cardiovascular': bool(row['cardiovascular_disease']),
            'stroke': bool(row['stroke'])
        }
        
        # Get prediction
        risk, version, shap_values, global_importance = predict(api_data)
        
        print(f"  Sample {idx}: Actual={row['diabetic']}, Predicted Risk={risk}%, Threshold={artifact.optimal_threshold:.4f}")
        print(f"    Features: glucose={row['glucose']}, bmi={row['bmi']:.1f}, hypertensive={row['hypertensive']}")
    
    # Test dengan data ekstrem
    print(f"\n🔥 Testing dengan data ekstrem:")
    
    # High risk case
    high_risk_data = {
        'age': 70,
        'gender': 'Male',
        'pulseRate': 90,
        'sbp': 180,
        'dbp': 110,
        'glucose': 200,
        'heightCm': 170,
        'weightKg': 90,
        'bmi': 31.1,
        'familyDiabetes': True,
        'hypertensive': True,
        'familyHypertension': True,
        'cardiovascular': True,
        'stroke': True
    }
    
    risk, version, shap_values, global_importance = predict(high_risk_data)
    print(f"  High Risk Case: Risk={risk}%, Threshold={artifact.optimal_threshold:.4f}")
    
    # Low risk case
    low_risk_data = {
        'age': 65,
        'gender': 'Female',
        'pulseRate': 70,
        'sbp': 120,
        'dbp': 80,
        'glucose': 90,
        'heightCm': 160,
        'weightKg': 55,
        'bmi': 21.5,
        'familyDiabetes': False,
        'hypertensive': False,
        'familyHypertension': False,
        'cardiovascular': False,
        'stroke': False
    }
    
    risk, version, shap_values, global_importance = predict(low_risk_data)
    print(f"  Low Risk Case: Risk={risk}%, Threshold={artifact.optimal_threshold:.4f}")

if __name__ == "__main__":
    test_model_with_training_data()




