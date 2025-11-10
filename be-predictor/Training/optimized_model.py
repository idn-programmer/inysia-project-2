"""
Optimized ML Model for Diabetes Prediction
Simplified version with better performance and fixed XGBoost warnings
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
    make_scorer
)
from imblearn.over_sampling import SMOTE

# Model Algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

# ==================== OPTIMIZED MODEL CONFIGURATIONS ====================

def get_optimized_models():
    """Define optimized models with fixed XGBoost parameters"""
    
    models = {
        'XGBoost_Optimized': {
            'model': XGBClassifier(
                random_state=42, 
                n_jobs=-1,
                eval_metric='logloss'
                # Removed use_label_encoder to fix warning
            ),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'scale_pos_weight': [2, 3, 5],
                'min_child_weight': [1, 3, 5]
            }
        },
        'LightGBM_Optimized': {
            'model': LGBMClassifier(
                random_state=42, 
                n_jobs=-1, 
                verbose=-1,
                force_col_wise=True
            ),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [5, 10, 20],
                'scale_pos_weight': [2, 3, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'CatBoost_Optimized': {
            'model': CatBoostClassifier(
                random_state=42, 
                verbose=0,
                thread_count=-1
            ),
            'params': {
                'iterations': [200, 300, 500],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'scale_pos_weight': [2, 3, 5]
            }
        },
        'RandomForest_Optimized': {
            'model': RandomForestClassifier(
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'params': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            }
        }
    }
    
    return models

# ==================== MODEL TRAINING & EVALUATION ====================

def train_and_evaluate_model(name, model_config, X_train, X_test, y_train, y_test):
    """Train a model with optimized techniques"""
    
    print(f"\n{'='*60}")
    print(f"🤖 Training: {name}")
    print(f"{'='*60}")
    
    import time
    start_time = time.time()
    
    # Apply SMOTE to balance classes
    print("\n⚖️  Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"✓ After SMOTE: {len(X_train_resampled)} samples")
    print(f"✓ Class distribution: {np.bincount(y_train_resampled)}")
    
    # Setup GridSearchCV with F1 scoring
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    # Use StratifiedKFold for better cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model_config['model'],
        param_grid=model_config['params'],
        cv=cv,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    print(f"\n⏳ Training {name}...")
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = grid_search.best_estimator_
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    metrics = {
        'model_name': name,
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'best_cv_score': grid_search.best_score_,
        'training_time': training_time,
        'best_params': grid_search.best_params_
    }
    
    # Print results
    print(f"\n📊 Results for {name}:")
    print(f"  F1 Score (weighted): {metrics['f1_score']:.4f}")
    print(f"  F1 Score (macro):    {metrics['f1_macro']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"  CV F1 Score:         {metrics['best_cv_score']:.4f}")
    print(f"  Training Time:       {training_time:.2f}s")
    
    print(f"\n🔧 Best Parameters:")
    for param, value in metrics['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    return best_model, metrics

# ==================== SHAP ANALYSIS ====================

def calculate_shap_values(model, X_sample, model_name):
    """Calculate SHAP values with improved error handling"""
    
    print(f"\n🔍 Calculating SHAP values for {model_name}...")
    
    try:
        # Use TreeExplainer for tree-based models
        if model_name in ['XGBoost_Optimized', 'LightGBM_Optimized', 'CatBoost_Optimized', 'RandomForest_Optimized']:
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use Explainer
            explainer = shap.Explainer(model, X_sample)
        
        # Calculate SHAP values on sample
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
    """Main execution function for optimized model training"""
    
    print("\n" + "="*60)
    print("🎯 OPTIMIZED DIABETES PREDICTION MODEL TRAINING")
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
    
    # Get optimized models
    models = get_optimized_models()
    
    # Train and evaluate all models
    all_results = []
    trained_models = {}
    
    for name, config in models.items():
        try:
            model, metrics = train_and_evaluate_model(
                name, config, X_train, X_test, y_train, y_test
            )
            
            trained_models[name] = model
            all_results.append(metrics)
            
        except Exception as e:
            print(f"\n❌ Error training {name}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Display results
        print("\n" + "="*60)
        print("📊 FINAL OPTIMIZED MODEL COMPARISON RESULTS")
        print("="*60)
        
        print("\n🏆 Model Rankings (by F1 Score):")
        print("-" * 60)
        
        for idx, row in results_df.iterrows():
            print(f"\n{idx + 1}. {row['model_name']}")
            print(f"   F1 Score (weighted): {row['f1_score']:.4f}")
            print(f"   F1 Score (macro):    {row['f1_macro']:.4f}")
            print(f"   Recall:              {row['recall']:.4f}")
            print(f"   Precision:           {row['precision']:.4f}")
            print(f"   ROC-AUC:             {row['roc_auc']:.4f}")
            print(f"   Training Time:       {row['training_time']:.2f}s")
        
        # Best model
        best_model_name = results_df.iloc[0]['model_name']
        best_f1 = results_df.iloc[0]['f1_score']
        best_recall = results_df.iloc[0]['recall']
        best_precision = results_df.iloc[0]['precision']
        
        print("\n" + "="*60)
        print(f"🥇 BEST OPTIMIZED MODEL: {best_model_name}")
        print("="*60)
        print(f"✓ F1 Score (weighted): {best_f1:.4f}")
        print(f"✓ F1 Score (macro):    {results_df.iloc[0]['f1_macro']:.4f}")
        print(f"✓ Recall:              {best_recall:.4f}")
        print(f"✓ Precision:           {best_precision:.4f}")
        print(f"✓ ROC-AUC:             {results_df.iloc[0]['roc_auc']:.4f}")
        
        # Check success criteria
        print("\n📋 Success Criteria Check:")
        print(f"  F1 Score >= 0.90:     {'✅ PASS' if best_f1 >= 0.90 else '❌ FAIL'} ({best_f1:.4f})")
        print(f"  Recall >= 0.80:       {'✅ PASS' if best_recall >= 0.80 else '❌ FAIL'} ({best_recall:.4f})")
        print(f"  Precision >= 0.80:    {'✅ PASS' if best_precision >= 0.80 else '❌ FAIL'} ({best_precision:.4f})")
        
        # Get best model and calculate SHAP
        best_model = trained_models[best_model_name]
        shap_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        shap_explainer, mean_shap = calculate_shap_values(best_model, shap_sample, best_model_name)
        
        # Save everything
        print("\n💾 Saving optimized model and results...")
        
        # Save comparison results
        results_path = output_dir.parent / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(results_path / "optimized_model_comparison.csv", index=False)
        print(f"✓ Results saved to {results_path / 'optimized_model_comparison.csv'}")
        
        # Save best model
        model_path = output_dir / "random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Best optimized model saved to {model_path}")
        
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
        
        # Save detailed report
        report_path = results_path / "optimized_model_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"OPTIMIZED MODEL REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n\n")
            f.write("Performance Metrics:\n")
            best_row = results_df.iloc[0]
            for col in ['f1_score', 'f1_macro', 'recall', 'precision', 'accuracy', 'roc_auc']:
                f.write(f"  {col}: {best_row[col]:.4f}\n")
            f.write(f"\nBest Parameters:\n")
            for param, value in best_row['best_params'].items():
                f.write(f"  {param}: {value}\n")
        
        print(f"✓ Report saved to {report_path}")
        print("\n✅ All optimized model artifacts saved successfully!")
    
    print("\n" + "="*60)
    print("✅ OPTIMIZED MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()




