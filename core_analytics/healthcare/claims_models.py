"""
Healthcare Claim Rejection-Risk Classification Model
Provides:
  - Scikit-learn Pipeline with ColumnTransformer for clean feature encoding and imputation
  - Strict prevention of target leakage:
      * EXCLUDES: `claim_status`, `denial_reason`, `paid_amount`, adjudication timestamps
      * INCLUDES: pre-adjudication features: `billed_amount`, `claim_type`, `diagnosis_code`,
        `procedure_code`, member plan type, member risk score, member age
  - Class-imbalance aware training (RandomForest / LogisticRegression with class_weight='balanced')
  - Evaluates standard metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC
  - Continuous rejection risk probabilities with calibrated risk bands:
      * Low Risk: p < 0.25
      * Medium Risk: 0.25 <= p < 0.60
      * High Risk: p >= 0.60
  - Strict separation of predicted risk from actual historical claim outcomes
  - Self-contained persistence support (e.g. claims_rejection_model.joblib) without touching student model_cache.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

DEFAULT_MODEL_PATH = "claims_rejection_model.joblib"

class ClaimRejectionRiskModel:
    """Classifies healthcare claims by pre-adjudication rejection risk."""

    # Explicit features available PRIOR to adjudication (No target leakage)
    NUMERIC_FEATURES = ['billed_amount', 'member_risk_score', 'member_age']
    CATEGORICAL_FEATURES = ['claim_type', 'diagnosis_code', 'procedure_code', 'plan_type']

    # Explicitly forbidden leakage columns
    LEAKAGE_COLUMNS = ['claim_status', 'denial_reason', 'paid_amount']

    def __init__(self, random_state: int = 42, n_estimators: int = 100):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.pipeline: Optional[Pipeline] = None
        self.evaluation_metrics: Dict[str, Any] = {}

    def _prepare_training_data(self, claims_df: pd.DataFrame, members_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Merges pre-adjudication member details (e.g., plan, risk score, age)
        and extracts binary rejection target (1 = Denied, 0 = Paid / Processed).
        Strictly excludes leakage features.
        """
        # Deduplicate claims to prevent synthetic twin leakage
        clean_claims = claims_df.drop_duplicates(subset=['claim_id']).copy()

        # Target definition: 1 if Denied, 0 otherwise
        y = clean_claims['claim_status'].apply(lambda x: 1 if str(x).strip().lower() == 'denied' else 0)

        # Merge pre-adjudication member demographic & benefit info
        m_subset = members_df[['member_id', 'plan_type', 'risk_score', 'date_of_birth']].copy()
        m_subset = m_subset.rename(columns={'risk_score': 'member_risk_score'})

        # Calculate member age from DOB
        dob = pd.to_datetime(m_subset['date_of_birth'], errors='coerce')
        now_dt = pd.to_datetime('2023-06-01') # Fixed benchmark date
        m_subset['member_age'] = ((now_dt - dob).dt.days // 365.25).fillna(45.0)

        merged = pd.merge(clean_claims, m_subset, on='member_id', how='left')

        # Drop leakage columns and identifiers
        drop_cols = self.LEAKAGE_COLUMNS + ['claim_id', 'member_id', 'provider_id', 'service_start_date', 'service_end_date', 'date_of_birth']
        X = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

        return X, y

    def train_and_evaluate(self, claims_df: pd.DataFrame, members_df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Trains model pipeline with reproducible train/test split.
        Imputers and encoders are fitted strictly on training data.
        """
        X, y = self._prepare_training_data(claims_df, members_df)

        if len(X) < 20:
            raise ValueError(f"Insufficient samples ({len(X)}) to train rejection model.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Build ColumnTransformer
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, [f for f in self.NUMERIC_FEATURES if f in X.columns]),
            ('cat', cat_transformer, [f for f in self.CATEGORICAL_FEATURES if f in X.columns])
        ])

        # Classifier with balanced class weights to address denial rarity (~15%)
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight='balanced',
            max_depth=6
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        self.pipeline.fit(X_train, y_train)

        # Evaluate on holdout test set
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = None

        self.evaluation_metrics = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'denial_ratio_train_pct': round(float(y_train.mean()) * 100, 2),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm,
            'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
            'threshold_selection_note': 'Uses balanced class weighting to counter class imbalance without arbitrary cutoff distortions.'
        }
        return self.evaluation_metrics

    def predict_rejection_risk(self, claims_df: pd.DataFrame, members_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates continuous rejection probability and categorical risk bands:
          - Low: p < 0.25
          - Medium: 0.25 <= p < 0.60
          - High: p >= 0.60
        Strictly preserves actual claim_status in a separate column to distinguish
        predicted risk from observed outcome.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline has not been trained or loaded.")

        X, _ = self._prepare_training_data(claims_df, members_df)
        probs = self.pipeline.predict_proba(X)[:, 1]

        results_df = claims_df.drop_duplicates(subset=['claim_id']).copy()
        results_df['rejection_probability'] = np.round(probs, 4)

        # Categorical risk bands
        def assign_band(p):
            if p < 0.25:
                return 'Low'
            elif p < 0.60:
                return 'Medium'
            else:
                return 'High'

        results_df['predicted_risk_band'] = results_df['rejection_probability'].apply(assign_band)
        # Clearly separate predicted risk from observed outcome
        results_df = results_df.rename(columns={'claim_status': 'actual_observed_status'})

        return results_df[['claim_id', 'member_id', 'billed_amount', 'actual_observed_status', 'rejection_probability', 'predicted_risk_band']]

    def save_model(self, file_path: str = DEFAULT_MODEL_PATH):
        """Persists the trained pipeline without overwriting student model_cache.joblib."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained pipeline.")
        joblib.dump({
            'pipeline': self.pipeline,
            'metrics': self.evaluation_metrics
        }, file_path)

    def load_model(self, file_path: str = DEFAULT_MODEL_PATH):
        """Loads a persisted claims rejection pipeline."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file '{file_path}' not found.")
        payload = joblib.load(file_path)
        self.pipeline = payload['pipeline']
        self.evaluation_metrics = payload.get('metrics', {})
