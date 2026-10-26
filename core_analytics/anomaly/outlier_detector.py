"""
Reusable Anomaly and Outlier Detection Engine
Provides statistical and machine-learning outlier detection:
  1. Tukey's Interquartile Range (IQR) Fences
  2. Modified Z-Score based on Median Absolute Deviation (MAD)
  3. Scikit-learn Isolation Forest for multivariate anomaly scoring

Core Design Principles:
  - Explicit and deterministic preprocessing.
  - Fits parameters/models only on training/reference data, reusable for scoring unseen records.
  - Safe handling of constant columns, missing values, small samples, and non-numeric inputs.
  - Structured results contract with record identifiers, anomaly scores, flags, methods,
    and feature-level contribution/explanation.
  - Strict compliance with ethical guidelines: anomalies represent statistical variations
    requiring review and are NEVER labeled as confirmed fraud.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyMethod(Enum):
    IQR = "IQR"
    MODIFIED_Z_SCORE = "MODIFIED_Z_SCORE"
    ISOLATION_FOREST = "ISOLATION_FOREST"

@dataclass
class AnomalyRecord:
    record_id: Any
    anomaly_score: float
    is_anomaly: bool
    method: AnomalyMethod
    feature_explanations: Dict[str, Any] = field(default_factory=dict)
    remediation_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'anomaly_score': round(self.anomaly_score, 4),
            'is_anomaly': self.is_anomaly,
            'method': self.method.value,
            'feature_explanations': self.feature_explanations,
            'remediation_note': self.remediation_note
        }

@dataclass
class AnomalyDetectionSummary:
    method: AnomalyMethod
    target_table: str
    target_features: List[str]
    total_evaluated: int
    anomaly_count: int
    anomaly_rate_pct: float
    anomalous_records: List[AnomalyRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value,
            'table': self.target_table,
            'features': self.target_features,
            'total_evaluated': self.total_evaluated,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate_pct': self.anomaly_rate_pct,
            'anomalies': [a.to_dict() for a in self.anomalous_records[:50]],
            'metadata': self.metadata
        }


class IQROutlierDetector:
    """
    Tukey's Fences / Interquartile Range (IQR) Outlier Detector.
    Identifies values falling beyond Q1 - k*IQR or Q3 + k*IQR.
    Default k=1.5 (standard outliers), k=3.0 (extreme outliers).
    """

    def __init__(self, k: float = 1.5):
        if k <= 0:
            raise ValueError(f"Multiplier k must be positive, got {k}")
        self.k = k
        self.fitted_bounds_: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, features: List[str]) -> "IQROutlierDetector":
        """Computes and stores Q1, Q3, and lower/upper bounds on reference data."""
        self.fitted_bounds_ = {}
        for feat in features:
            if feat not in df.columns:
                continue
            series = pd.to_numeric(df[feat], errors='coerce').dropna()
            if len(series) == 0:
                continue
            q1 = float(np.percentile(series, 25))
            q3 = float(np.percentile(series, 75))
            iqr = q3 - q1
            self.fitted_bounds_[feat] = {
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': q1 - (self.k * iqr),
                'upper_bound': q3 + (self.k * iqr)
            }
        return self

    def score(self, df: pd.DataFrame, id_column: str, table_name: str = "dataset") -> AnomalyDetectionSummary:
        """Scores records against the fitted IQR thresholds."""
        if not self.fitted_bounds_:
            raise RuntimeError("Detector must be fitted with valid numeric features before scoring.")
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' not found in DataFrame.")

        features = list(self.fitted_bounds_.keys())
        anomalous_records = []
        total = len(df)

        if total == 0:
            return AnomalyDetectionSummary(
                method=AnomalyMethod.IQR,
                target_table=table_name,
                target_features=features,
                total_evaluated=0,
                anomaly_count=0,
                anomaly_rate_pct=0.0,
                metadata={'bounds': self.fitted_bounds_}
            )

        for idx, row in df.iterrows():
            rec_id = row[id_column]
            is_anomaly = False
            explanations = {}
            max_dev = 0.0

            for feat, bounds in self.fitted_bounds_.items():
                val = pd.to_numeric(row.get(feat), errors='coerce')
                if pd.isna(val):
                    continue
                lb = bounds['lower_bound']
                ub = bounds['upper_bound']
                iqr = bounds['iqr']

                if val < lb:
                    is_anomaly = True
                    dist = (lb - val) / (iqr if iqr > 0 else 1.0)
                    max_dev = max(max_dev, dist)
                    explanations[feat] = {
                        'value': val,
                        'reason': 'BELOW_LOWER_BOUND',
                        'bound': round(lb, 2),
                        'severity_ratio': round(dist, 2)
                    }
                elif val > ub:
                    is_anomaly = True
                    dist = (val - ub) / (iqr if iqr > 0 else 1.0)
                    max_dev = max(max_dev, dist)
                    explanations[feat] = {
                        'value': val,
                        'reason': 'ABOVE_UPPER_BOUND',
                        'bound': round(ub, 2),
                        'severity_ratio': round(dist, 2)
                    }

            if is_anomaly:
                anomalous_records.append(AnomalyRecord(
                    record_id=rec_id,
                    anomaly_score=max_dev,
                    is_anomaly=True,
                    method=AnomalyMethod.IQR,
                    feature_explanations=explanations,
                    remediation_note="Statistical anomaly: value lies outside Tukey fence. Review for data entry variance or edge-case utilization."
                ))

        rate = round((len(anomalous_records) / total) * 100, 2) if total > 0 else 0.0
        return AnomalyDetectionSummary(
            method=AnomalyMethod.IQR,
            target_table=table_name,
            target_features=features,
            total_evaluated=total,
            anomaly_count=len(anomalous_records),
            anomaly_rate_pct=rate,
            anomalous_records=anomalous_records,
            metadata={'k': self.k, 'bounds': self.fitted_bounds_}
        )


class ModifiedZScoreOutlierDetector:
    """
    Modified Z-score based on Median Absolute Deviation (MAD):
      M_i = 0.6745 * (x_i - median) / MAD
    Recommended by Iglewicz and Hoaglin (1993). More robust than standard Z-score
    because median and MAD are not distorted by extreme outliers.
    Default threshold = 3.5.
    """

    def __init__(self, threshold: float = 3.5):
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        self.threshold = threshold
        self.fitted_stats_: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, features: List[str]) -> "ModifiedZScoreOutlierDetector":
        """Calculates and stores median and MAD on reference data."""
        self.fitted_stats_ = {}
        for feat in features:
            if feat not in df.columns:
                continue
            series = pd.to_numeric(df[feat], errors='coerce').dropna()
            if len(series) == 0:
                continue
            median = float(series.median())
            abs_dev = (series - median).abs()
            mad = float(abs_dev.median())

            # If MAD == 0 (e.g. constant column or heavily tied values), fallback to mean absolute deviation or tiny epsilon
            if mad == 0.0:
                mean_dev = float(abs_dev.mean())
                mad = mean_dev if mean_dev > 0.0 else 1e-6

            self.fitted_stats_[feat] = {
                'median': median,
                'mad': mad
            }
        return self

    def score(self, df: pd.DataFrame, id_column: str, table_name: str = "dataset") -> AnomalyDetectionSummary:
        """Scores records with modified Z-score."""
        if not self.fitted_stats_:
            raise RuntimeError("Detector must be fitted before scoring.")
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' not found in DataFrame.")

        features = list(self.fitted_stats_.keys())
        anomalous_records = []
        total = len(df)

        if total == 0:
            return AnomalyDetectionSummary(
                method=AnomalyMethod.MODIFIED_Z_SCORE,
                target_table=table_name,
                target_features=features,
                total_evaluated=0,
                anomaly_count=0,
                anomaly_rate_pct=0.0,
                metadata={'stats': self.fitted_stats_}
            )

        for idx, row in df.iterrows():
            rec_id = row[id_column]
            is_anomaly = False
            explanations = {}
            max_score = 0.0

            for feat, stats in self.fitted_stats_.items():
                val = pd.to_numeric(row.get(feat), errors='coerce')
                if pd.isna(val):
                    continue
                med = stats['median']
                mad = stats['mad']

                mod_z = 0.6745 * abs(val - med) / mad
                if mod_z > max_score:
                    max_score = mod_z

                if mod_z > self.threshold:
                    is_anomaly = True
                    explanations[feat] = {
                        'value': val,
                        'median': round(med, 2),
                        'modified_z_score': round(mod_z, 2),
                        'threshold': self.threshold
                    }

            if is_anomaly:
                anomalous_records.append(AnomalyRecord(
                    record_id=rec_id,
                    anomaly_score=max_score,
                    is_anomaly=True,
                    method=AnomalyMethod.MODIFIED_Z_SCORE,
                    feature_explanations=explanations,
                    remediation_note="Statistical outlier: Modified Z-score exceeds MAD threshold. Flags abnormal variance for secondary clinical/financial audit."
                ))

        rate = round((len(anomalous_records) / total) * 100, 2) if total > 0 else 0.0
        return AnomalyDetectionSummary(
            method=AnomalyMethod.MODIFIED_Z_SCORE,
            target_table=table_name,
            target_features=features,
            total_evaluated=total,
            anomaly_count=len(anomalous_records),
            anomaly_rate_pct=rate,
            anomalous_records=anomalous_records,
            metadata={'threshold': self.threshold, 'stats': self.fitted_stats_}
        )


class IsolationForestOutlierDetector:
    """
    Multivariate Outlier Detection via Scikit-Learn IsolationForest.
    Isolates observations by randomly selecting a feature and split value.
    Configurable contamination rate and random state for deterministic execution.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42, n_estimators: int = 100):
        if not (0.0 < contamination <= 0.5):
            raise ValueError(f"Contamination must be in (0.0, 0.5], got {contamination}")
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_: Optional[IsolationForest] = None
        self.feature_names_: List[str] = []
        self.medians_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, features: List[str]) -> "IsolationForestOutlierDetector":
        """Preprocesses numeric data (median imputation for NaNs) and fits the IsolationForest."""
        self.feature_names_ = []
        valid_cols = []
        for feat in features:
            if feat in df.columns:
                series = pd.to_numeric(df[feat], errors='coerce')
                if series.notnull().sum() > 0:
                    valid_cols.append(feat)
                    self.medians_[feat] = float(series.median())

        if not valid_cols:
            raise ValueError("No valid numeric features available to fit Isolation Forest.")

        self.feature_names_ = valid_cols
        X = self._prepare_features(df)

        if len(X) < 10:
            raise ValueError(f"Sample size ({len(X)}) too small to reliably train Isolation Forest (minimum 10 required).")

        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators
        )
        self.model_.fit(X)
        return self

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Deterministic numeric coercion and median imputation."""
        cols = []
        for feat in self.feature_names_:
            series = pd.to_numeric(df[feat], errors='coerce')
            med = self.medians_.get(feat, 0.0)
            cols.append(series.fillna(med).values)
        return np.column_stack(cols)

    def score(self, df: pd.DataFrame, id_column: str, table_name: str = "dataset") -> AnomalyDetectionSummary:
        """Scores records using the fitted IsolationForest model."""
        if self.model_ is None:
            raise RuntimeError("IsolationForest model must be fitted before scoring.")
        if id_column not in df.columns:
            raise ValueError(f"id_column '{id_column}' not found in DataFrame.")

        total = len(df)
        if total == 0:
            return AnomalyDetectionSummary(
                method=AnomalyMethod.ISOLATION_FOREST,
                target_table=table_name,
                target_features=self.feature_names_,
                total_evaluated=0,
                anomaly_count=0,
                anomaly_rate_pct=0.0,
                metadata={'contamination': self.contamination}
            )

        X = self._prepare_features(df)
        preds = self.model_.predict(X) # -1 for anomaly, 1 for inlier
        scores = self.model_.decision_function(X) # Lower scores indicate higher anomaly tendency

        anomalous_records = []
        for i in range(total):
            rec_id = df.iloc[i][id_column]
            is_ano = bool(preds[i] == -1)
            raw_score = float(scores[i])
            # Normalized anomaly score where higher = more anomalous (0 = normal, >0 = anomalous)
            anomaly_intensity = float(max(0.0, -raw_score))

            if is_ano:
                explanations = {
                    feat: {
                        'value': float(X[i, f_idx]),
                        'feature_median': self.medians_[feat]
                    }
                    for f_idx, feat in enumerate(self.feature_names_)
                }
                anomalous_records.append(AnomalyRecord(
                    record_id=rec_id,
                    anomaly_score=anomaly_intensity,
                    is_anomaly=True,
                    method=AnomalyMethod.ISOLATION_FOREST,
                    feature_explanations=explanations,
                    remediation_note="Multivariate Isolation Forest anomaly detected. Represents high-dimensional deviation; flag for operational review, NOT fraud."
                ))

        rate = round((len(anomalous_records) / total) * 100, 2) if total > 0 else 0.0
        return AnomalyDetectionSummary(
            method=AnomalyMethod.ISOLATION_FOREST,
            target_table=table_name,
            target_features=self.feature_names_,
            total_evaluated=total,
            anomaly_count=len(anomalous_records),
            anomaly_rate_pct=rate,
            anomalous_records=anomalous_records,
            metadata={'contamination': self.contamination, 'n_estimators': self.n_estimators}
        )
