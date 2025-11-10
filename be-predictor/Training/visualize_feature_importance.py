"""
Visualisasi Feature Importance dari Model LightGBM
Menampilkan weights/importance dari setiap variabel yang ditemukan oleh model
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# Model Algorithms
from lightgbm import LGBMClassifier

# Set style untuk visualisasi yang lebih menarik
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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

# ==================== MODEL LOADING/TRAINING ====================

def load_or_train_model(X_train, X_test, y_train, y_test):
    """Load existing model or train a new one"""
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir.parent / "backend" / "models" / "random_forest_model.pkl"
    
    model = None
    if model_path.exists():
        try:
            print(f"\n📥 Loading existing model from: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("🔄 Training new model...")
            model = None
    
    if model is None:
        print("\n🤖 Training new LightGBM model...")
        
        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Train LightGBM
        model = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
            scale_pos_weight=10
        )
        model.fit(X_train_resampled, y_train_resampled)
        print("✓ Model trained successfully")
    
    return model

# ==================== EVALUATION ====================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc
    }
    
    return metrics, y_pred, y_pred_proba

# ==================== VISUALIZATION ====================

def visualize_feature_importance(model, feature_names):
    """Visualize feature importance dengan warna yang menarik"""
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'Fitur': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Create figure with custom colors
    plt.figure(figsize=(12, 8))
    
    # Use viridis color scheme (bisa diganti dengan 'plasma', 'inferno', 'magma', atau custom)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(coef_df)))
    
    # Create horizontal bar plot
    bars = plt.barh(coef_df['Fitur'], coef_df['Importance'], color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(coef_df.iterrows()):
        plt.text(row['Importance'] + max(importances) * 0.01, i, 
                f'{row["Importance"]:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
    plt.ylabel('Fitur', fontsize=14, fontweight='bold')
    plt.title('Pengaruh Fitur terhadap Diabetes (Feature Importance - LightGBM)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "feature_importance_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualisasi disimpan ke: {output_path}")
    
    plt.show()
    
    return coef_df

def visualize_prediction_distribution(y_test, y_pred_proba):
    """Visualize distribution of prediction probabilities"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram of probabilities
    axes[0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Diabetes', 
                 color='#2ecc71', edgecolor='black')
    axes[0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Diabetes', 
                 color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribusi Probabilitas Prediksi', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Box plot
    data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
    bp = axes[1].boxplot(data_to_plot, labels=['No Diabetes', 'Diabetes'], 
                         patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_ylabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Box Plot Probabilitas Prediksi', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "prediction_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualisasi distribusi disimpan ke: {output_path}")
    
    plt.show()

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("📊 VISUALISASI FEATURE IMPORTANCE - LIGHTGBM MODEL")
    print("="*60)
    
    # Paths
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "Data" / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    
    # 1️⃣ Load data
    df = load_data(str(data_path))
    
    # 2️⃣ Preview data
    print("\n📋 Preview dataset:")
    print(df.head())
    
    # 3️⃣ Preprocess data
    X, y, scaler, label_encoder = preprocess_data(df)
    feature_names = list(X.columns)
    
    # 4️⃣ Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # 5️⃣ Load or train model
    model = load_or_train_model(X_train, X_test, y_train, y_test)
    
    # 6️⃣ Evaluate model
    print("\n📈 Evaluating model...")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    print(f"\n📊 Model Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # 7️⃣ Visualize feature importance
    print("\n🎨 Creating feature importance visualization...")
    importance_df = visualize_feature_importance(model, feature_names)
    
    print("\n📋 Feature Importance (sorted by importance):")
    print(importance_df.to_string(index=False))
    
    # 8️⃣ Visualize prediction distribution
    print("\n📊 Creating prediction distribution visualization...")
    visualize_prediction_distribution(y_test, y_pred_proba)
    
    print("\n" + "="*60)
    print("✅ VISUALISASI SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()


