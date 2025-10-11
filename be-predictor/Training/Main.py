import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
    df = df[df['age'] >= 60]  # Focus on elderly people
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop('diabetic', axis=1)
    y = df['diabetic']
    if 'gender' in X.columns:
        le = LabelEncoder()
        X['gender'] = le.fit_transform(X['gender'])
    nan_indices = y[y.isna()].index
    X = X.drop(nan_indices)
    y = y.drop(nan_indices)
    # Feature scaling for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, y, scaler, le if 'gender' in df.columns else None

# Train and tune model
def train_and_tune_model(X, y, class_weight={0: 1, 1: 3}):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    rf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight)
    random_search = RandomizedSearchCV(
        estimator=rf_classifier, 
        param_distributions=param_dist, 
        scoring='recall', 
        cv=5, 
        n_iter=30, 
        n_jobs=-1, 
        random_state=42
    )
    random_search.fit(X_train_res, y_train_res)
    best_rf_model = random_search.best_estimator_

    y_pred_best_rf = best_rf_model.predict(X_test)
    y_pred_proba_tuned_rf = best_rf_model.predict_proba(X_test)
    confidence_scores = np.max(y_pred_proba_tuned_rf, axis=1)
    average_confidence = np.mean(confidence_scores) * 100

    # Evaluation
    print("Best hyperparameters found:", random_search.best_params_)
    print("Best cross-validation recall score:", random_search.best_score_)
    print("\nConfusion Matrix (Tuned Random Forest):")
    print(confusion_matrix(y_test, y_pred_best_rf))
    print("\nClassification Report (Tuned Random Forest):")
    print(classification_report(y_test, y_pred_best_rf))
    print("\nAccuracy Score (Tuned Random Forest):")
    print(accuracy_score(y_test, y_pred_best_rf))
    print(f"Average confidence score for predictions: {average_confidence:.2f}%")

    # Threshold evaluation (default 0.5)
    y_pred_tuned_rf_threshold = (y_pred_proba_tuned_rf[:, 1] >= 0.49).astype(int)
    print("\nConfusion Matrix (Tuned Random Forest with 0.49 Threshold):")
    print(confusion_matrix(y_test, y_pred_tuned_rf_threshold))
    print("\nClassification Report (Tuned Random Forest with 0.49 Threshold):")
    print(classification_report(y_test, y_pred_tuned_rf_threshold))
    print("\nAccuracy Score (Tuned Random Forest with 0.49 Threshold):")
    print(accuracy_score(y_test, y_pred_tuned_rf_threshold))

    return best_rf_model, X_train_res, X_test

# Train SHAP explainer
def train_shap_explainer(model, X_sample):
    print("\nTraining SHAP explainer...")
    # Use a sample of training data for faster SHAP computation
    # Using TreeExplainer which is optimized for tree-based models
    explainer = shap.TreeExplainer(model)
    print("SHAP explainer trained successfully")
    return explainer

# Save model and artifacts
def save_model(model, scaler, label_encoder, explainer, feature_names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main model
    model_path = output_dir / "random_forest_model.pkl"
    pickle.dump(model, open(model_path, 'wb'))
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    pickle.dump(scaler, open(scaler_path, 'wb'))
    print(f"Scaler saved to {scaler_path}")
    
    # Save label encoder
    if label_encoder:
        le_path = output_dir / "label_encoder.pkl"
        pickle.dump(label_encoder, open(le_path, 'wb'))
        print(f"Label encoder saved to {le_path}")
    
    # Save SHAP explainer
    explainer_path = output_dir / "shap_explainer.pkl"
    pickle.dump(explainer, open(explainer_path, 'wb'))
    print(f"SHAP explainer saved to {explainer_path}")
    
    # Save feature names
    feature_names_path = output_dir / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"Feature names saved to {feature_names_path}")

# Load model for inference
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

# Main function for training and saving model
if __name__ == "__main__":
    # Use relative path
    script_dir = Path(__file__).resolve().parent  # Gets Training folder
    data_path = script_dir / "Data" / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    
    print(f"Loading data from: {data_path}")
    data = load_data(str(data_path))
    print(f"Dataset loaded: {len(data)} records (age >= 60)")
    
    X, y, scaler, label_encoder = preprocess_data(data)
    feature_names = list(X.columns)
    print(f"Features: {feature_names}")
    
    # Load class weight from file if exists, else use default
    try:
        with open('class_weight.json', 'r') as f:
            class_weight = json.load(f)
            class_weight = {int(k): v for k, v in class_weight.items()}
    except FileNotFoundError:
        class_weight = {0: 1, 1: 3}
    
    best_rf_model, X_train_sample, X_test = train_and_tune_model(X, y, class_weight=class_weight)
    
    # Train SHAP explainer
    # Use a subset of training data for efficiency
    shap_sample_size = min(100, len(X_train_sample))
    X_shap_sample = X_train_sample.sample(n=shap_sample_size, random_state=42)
    explainer = train_shap_explainer(best_rf_model, X_shap_sample)
    
    # Save everything
    output_dir = script_dir.parent / "backend" / "models"
    save_model(best_rf_model, scaler, label_encoder, explainer, feature_names, output_dir)
    
    # Save class weight to file
    class_weight_path = output_dir / "class_weight.json"
    with open(class_weight_path, 'w') as f:
        json.dump(class_weight, f)
    print(f"Class weights saved to {class_weight_path}")
    
    print("\n✓ Training complete! All artifacts saved.")
