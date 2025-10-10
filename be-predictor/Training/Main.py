import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names to expected
    df = df.rename(
        columns={
            "pulse_rate": "pulseRate",
            "systolic_bp": "sbp",
            "diastolic_bp": "dbp",
            "height": "height_m",
            "weight": "weightKg",
            "family_diabetes": "familyDiabetes",
            "family_hypertension": "familyHypertension",
            "cardiovascular_disease": "cardiovascular",
        }
    )

    # Convert units
    # glucose from mmol/L to mg/dL if values look like mmol range
    # Heuristic: if median < 30 assume mmol/L and convert by *18
    if df["glucose"].median() < 30:
        df["glucose"] = df["glucose"] * 18.0

    # convert height to cm
    if "height_m" in df.columns:
        df["heightCm"] = df["height_m"] * 100.0
    else:
        df["heightCm"] = np.nan

    # weight column already in kg as weightKg
    if "weightKg" not in df.columns and "weight" in df.columns:
        df["weightKg"] = df["weight"]

    # Ensure BMI present or compute
    if "bmi" not in df.columns or df["bmi"].isna().all():
        m = (df["heightCm"] / 100.0).replace(0, np.nan)
        df["bmi"] = df["weightKg"] / (m * m)

    # Binary target
    df["target"] = (df["diabetic"].astype(str).str.lower() == "yes").astype(int)

    # Gender to expected spelling
    df["gender"] = df["gender"].astype(str)

    return df


def build_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = [
        "age",
        "pulseRate",
        "sbp",
        "dbp",
        "glucose",
        "heightCm",
        "weightKg",
        "bmi",
        "familyDiabetes",
        "hypertensive",
        "familyHypertension",
        "cardiovascular",
        "stroke",
    ]
    categorical_features = ["gender"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], remainder="drop"
    )

    model = LogisticRegression(max_iter=1000)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    # Attach metadata for serving
    pipe.model_version = "v1.0.0"
    pipe.feature_names = numeric_features + categorical_features
    return pipe


def main():
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "Training" / "Data" / "DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    df = load_dataset(str(csv_path))

    features = [
        "age",
        "gender",
        "pulseRate",
        "sbp",
        "dbp",
        "glucose",
        "heightCm",
        "weightKg",
        "bmi",
        "familyDiabetes",
        "hypertensive",
        "familyHypertension",
        "cardiovascular",
        "stroke",
    ]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline(features)
    pipe.fit(X_train, y_train)

    out_dir = root / "backend" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.joblib"
    joblib.dump(pipe, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, recall_score, precision_score, f1_score

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
    df = df[df['age'] >= 60]
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
    return X, y

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
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    rf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight)
    random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, scoring='recall', cv=5, n_iter=30, n_jobs=-1, random_state=42)
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
    print("\nConfusion Matrix (Tuned Random Forest with 0.5 Threshold):")
    print(confusion_matrix(y_test, y_pred_tuned_rf_threshold))
    print("\nClassification Report (Tuned Random Forest with 0.5 Threshold):")
    print(classification_report(y_test, y_pred_tuned_rf_threshold))
    print("\nAccuracy Score (Tuned Random Forest with 0.5 Threshold):")
    print(accuracy_score(y_test, y_pred_tuned_rf_threshold))

    return best_rf_model


# Save model
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    print(f"Best model saved to {filename}")

# Predict risk
def predict_risk(model, user_data):
    # user_data should be a 2D numpy array or DataFrame matching training features
    risk_prediction = model.predict(user_data)
    risk_proba = model.predict_proba(user_data)
    return risk_prediction, risk_proba

# Load model for inference
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

# Main function for training and saving model
if __name__ == "__main__":
    data = load_data(r'D:\inysia project\DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv')
    X, y = preprocess_data(data)
    # Load class weight from file if exists, else use default
    try:
        with open('class_weight.json', 'r') as f:
            class_weight = json.load(f)
            class_weight = {int(k): v for k, v in class_weight.items()}
    except FileNotFoundError:
        class_weight = {0: 1, 1: 3}
    best_rf_model = train_and_tune_model(X, y, class_weight=class_weight)
    save_model(best_rf_model, 'best_random_forest_model.pkl')
    # Save class weight to file
    with open('class_weight.json', 'w') as f:
        json.dump(class_weight, f)