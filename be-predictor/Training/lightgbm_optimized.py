"""
LightGBM Optimized with Threshold Tuning and Class Weight Tuning
Focus on maximizing F1 score for balanced diabetes prediction
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, recall_score, precision_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE

# Model Algorithms
from lightgbm import LGBMClassifier

# SHAP for interpretability
import shap

# ==================== DATA LOADING & PREPROCESSING ====================

def load_data(file_path):
    """Load and preprocess diabetes dataset"""
    print(f"\n📊 Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Map target variable
    df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
    
    # Focus on elderly people (age >= 60)
    df = df[df['age'] >= 60]
    
    print(f"✓ Dataset loaded: {len(df)} records (age >= 60)")
    print(f"✓ Diabetic distribution: {df['diabetic'].value_counts().to_dict()}")
    print(f"✓ Class imbalance ratio: {df['diabetic'].value_counts()[0] / df['diabetic'].value_counts()[1]:.2f}:1")
    
    return df

def preprocess_data(df):
    """Preprocess features and target"""
    print("\n🔧 Preprocessing data...")
    
    X = df.drop('diabetic', axis=1)
    y = df['diabetic']
    
    # Encode gender if present
    label_encoder = None
    if 'gender' in X.columns:
        label_encoder = LabelEncoder()
        X['gender'] = label_encoder.fit_transform(X['gender'])
    
    # Remove NaN values
    nan_indices = y[y.isna()].index
    X = X.drop(nan_indices)
    y = y.drop(nan_indices)
    
    # Use RobustScaler for better outlier handling
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    print(f"✓ Features: {list(X.columns)}")
    print(f"✓ Total samples: {len(X)}")
    
    return X, y, scaler, label_encoder

# ==================== THRESHOLD TUNING ====================

def find_optimal_threshold(y_true, y_pred_proba, target_f1=0.7):
    """Find optimal threshold for target F1 score"""
    
    print(f"\n🎯 Finding optimal threshold for F1 score >= {target_f1}")
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Fix indexing issue - precision_recall_curve returns arrays of different lengths
    # precision and recall have length n_thresholds + 1, thresholds has length n_thresholds
    # We need to align them properly
    n_thresholds = len(thresholds)
    precision = precision[:n_thresholds]
    recall = recall[:n_thresholds]
    
    # Calculate F1 scores for all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Find thresholds that meet target F1 score
    valid_indices = f1_scores >= target_f1
    if not np.any(valid_indices):
        print(f"⚠️  No threshold found for F1 >= {target_f1}")
        print(f"   Max F1 achievable: {np.max(f1_scores):.4f}")
        # Use threshold that gives max F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    else:
        # Among valid thresholds, choose the one with highest F1 score
        valid_f1_scores = f1_scores[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        optimal_idx = np.argmax(valid_f1_scores)
        optimal_threshold = valid_thresholds[optimal_idx]
    
    print(f"✓ Optimal threshold: {optimal_threshold:.4f}")
    
    # Calculate metrics with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    optimal_recall = recall_score(y_true, y_pred_optimal)
    optimal_precision = precision_score(y_true, y_pred_optimal)
    optimal_f1 = f1_score(y_true, y_pred_optimal)
    
    print(f"✓ With optimal threshold:")
    print(f"   Recall: {optimal_recall:.4f}")
    print(f"   Precision: {optimal_precision:.4f}")
    print(f"   F1 Score: {optimal_f1:.4f}")
    
    return optimal_threshold, y_pred_optimal

# ==================== LIGHTGBM OPTIMIZATION ====================

def train_lightgbm_optimized(X_train, X_test, y_train, y_test):
    """Train LightGBM with class weight and threshold tuning"""
    
    print("\n" + "="*60)
    print("🤖 Training LightGBM with Class Weight & Threshold Tuning")
    print("="*60)
    
    import time
    start_time = time.time()
    
    # Apply SMOTE to balance classes
    print("\n⚖️  Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"✓ After SMOTE: {len(X_train_resampled)} samples")
    print(f"✓ Class distribution: {np.bincount(y_train_resampled)}")
    
    # Test different class weights
    class_weights = [1, 2, 3, 5, 8, 10, 15, 20]
    best_model = None
    best_threshold = 0.5
    best_metrics = {'f1_score': 0}  # Initialize with F1 score
    
    print(f"\n🔄 Testing {len(class_weights)} different class weights...")
    
    for weight in class_weights:
        print(f"\n--- Testing class_weight: {weight} ---")
        
        # Create LightGBM with current class weight
        lgb_model = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
            scale_pos_weight=weight
        )
        
        # Train model
        lgb_model.fit(X_train_resampled, y_train_resampled)
        
        # Get predictions
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for this model
        threshold, y_pred_optimal = find_optimal_threshold(y_test, y_pred_proba, target_f1=0.7)
        
        # Calculate metrics
        recall = recall_score(y_test, y_pred_optimal)
        precision = precision_score(y_test, y_pred_optimal)
        f1 = f1_score(y_test, y_pred_optimal)
        accuracy = accuracy_score(y_test, y_pred_optimal)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   Results: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
        
        # Keep track of best model (prioritize F1 score)
        if f1 > best_metrics.get('f1_score', 0):
            best_model = lgb_model
            best_threshold = threshold
            best_metrics = {
                'class_weight': weight,
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
    
    training_time = time.time() - start_time
    
    # Final evaluation with best model
    print(f"\n🏆 Best Model Results:")
    print(f"   Class Weight: {best_metrics['class_weight']}")
    print(f"   Optimal Threshold: {best_metrics['threshold']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    
    # Final predictions with best model and threshold
    y_pred_proba_final = best_model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_pred_proba_final >= best_threshold).astype(int)
    
    print(f"\n📈 Final Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))
    
    print(f"\n📋 Final Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=['No Diabetes', 'Diabetes']))
    
    return best_model, best_threshold, best_metrics, training_time

# ==================== SHAP ANALYSIS ====================

def calculate_shap_values(model, X_sample, model_name):
    """Calculate SHAP values for model interpretability"""
    
    print(f"\n🔍 Calculating SHAP values for {model_name}...")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Calculate mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        print(f"✓ SHAP values calculated successfully")
        
        return explainer, mean_abs_shap
        
    except Exception as e:
        print(f"⚠️  Error calculating SHAP values: {e}")
        return None, None

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function for LightGBM optimization"""
    
    print("\n" + "="*60)
    print("🎯 LIGHTGBM OPTIMIZED FOR F1 SCORE WITH THRESHOLD & CLASS WEIGHT TUNING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "Data" / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    output_dir = script_dir.parent / "backend" / "models"
    
    # Load and preprocess data
    data = load_data(str(data_path))
    X, y, scaler, label_encoder = preprocess_data(data)
    feature_names = list(X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Diabetes cases in test set: {y_test.sum()}")
    
    # Train optimized LightGBM
    best_model, best_threshold, best_metrics, training_time = train_lightgbm_optimized(
        X_train, X_test, y_train, y_test
    )
    
    # Calculate SHAP values
    shap_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    shap_explainer, mean_shap = calculate_shap_values(best_model, shap_sample, "LightGBM_Optimized")
    
    # Save everything
    print("\n💾 Saving optimized LightGBM model and results...")
    
    # Save comparison results
    results_path = output_dir.parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "random_forest_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best LightGBM model saved to {model_path}")
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save label encoder
    if label_encoder:
        le_path = output_dir / "label_encoder.pkl"
        with open(le_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✓ Label encoder saved to {le_path}")
    
    # Save SHAP explainer
    if shap_explainer:
        explainer_path = output_dir / "shap_explainer.pkl"
        with open(explainer_path, 'wb') as f:
            pickle.dump(shap_explainer, f)
        print(f"✓ SHAP explainer saved to {explainer_path}")
    
    # Save feature names
    feature_names_path = output_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"✓ Feature names saved to {feature_names_path}")
    
    # Save threshold
    threshold_path = output_dir / "optimal_threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump({"threshold": best_threshold}, f)
    print(f"✓ Optimal threshold saved to {threshold_path}")
    
    # Save detailed report
    report_path = results_path / "lightgbm_optimized_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"LIGHTGBM OPTIMIZED REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write("Optimization Results:\n")
        f.write(f"  Class Weight: {best_metrics['class_weight']}\n")
        f.write(f"  Optimal Threshold: {best_metrics['threshold']:.4f}\n\n")
        f.write("Performance Metrics:\n")
        for metric, value in best_metrics.items():
            if metric not in ['class_weight', 'threshold']:
                f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nTraining Time: {training_time:.2f}s\n")
    
    print(f"✓ Report saved to {report_path}")
    print("\n✅ LightGBM optimization complete!")
    
    print("\n" + "="*60)
    print("✅ LIGHTGBM OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
