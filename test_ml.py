#!/usr/bin/env python3
"""
Test script for the Student Dropout Prediction System
Tests the core ML pipeline, data preprocessing, and SVM evaluation.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

REQUIRED_COLUMNS_PREDICTION = [
    "Marital status", "Course", "Daytime/evening attendance", "Previous qualification", 
    "Previous qualification (grade)", "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", 
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", 
    "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", 
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)", 
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"
]
REQUIRED_COLUMNS_TRAINING = REQUIRED_COLUMNS_PREDICTION + ["Target"]

def load_and_validate_data(filename):
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(filename)
        print(f"[+] Dataset loaded successfully: {df.shape}")

        missing_columns = [col for col in REQUIRED_COLUMNS_TRAINING if col not in df.columns]
        if missing_columns:
            print(f"[-] Missing columns: {missing_columns}")
            return None

        print("[+] All required columns present")
        return df
    except Exception as e:
        print(f"[-] Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """Preprocess dataset for SVM training."""
    print("\n[*] Preprocessing data...")
    processed_df = df.copy()

    # Normalize target
    processed_df['Target'] = processed_df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

    # Encode categorical columns
    label_encoders = {}
    for col in sorted(processed_df.select_dtypes(include=['object']).columns):
        if col != 'Target':
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le

    X = processed_df.drop('Target', axis=1)
    # Filter only relevant training columns
    cols_to_use = [c for c in REQUIRED_COLUMNS_PREDICTION if c in X.columns]
    X = X[cols_to_use]
    y = processed_df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[+] Data preprocessed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled, y, scaler, label_encoders, processed_df

def train_and_evaluate_model(X, y):
    """Train SVM model and evaluate performance."""
    print("\n[*] Training SVM model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training set: {X_train.shape[0]} samples")
    print(f"    Test set:     {X_test.shape[0]} samples")

    model = SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[+] Model Performance:")
    print(f"    Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"    Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"    Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"\n    Confusion Matrix:\n{cm}")

    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def main():
    print("=" * 60)
    print("Student Dropout Prediction System - Model Pipeline Test")
    print("=" * 60)

    data_file = 'Data.csv'
    df = load_and_validate_data(data_file)
    if df is None:
        print("[-] Test failed: Could not load data.")
        sys.exit(1)

    X, y, scaler, label_encoders, processed_df = preprocess_data(df)
    model, metrics = train_and_evaluate_model(X, y)

    # Validate predictions on test sample
    probabilities = model.predict_proba(X[:5])[:, 1]
    print("\n[+] Sample predictions (first 5 samples dropout probability):")
    for idx, prob in enumerate(probabilities):
        print(f"    Sample {idx + 1}: {prob * 100:.2f}%")

    print("\n[+] Core ML pipeline test passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
