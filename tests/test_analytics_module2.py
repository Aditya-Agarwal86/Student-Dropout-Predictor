"""
Tests for Core Analytics Module 2:
- Tukey's Fences / IQR Outlier Detector
- Modified Z-Score (Median Absolute Deviation) Outlier Detector
- Scikit-Learn IsolationForest Outlier Detector
- Edge cases: constant columns, missing values, empty inputs, small samples, invalid parameters, reproducibility
- Ethical validation: no labels of confirmed fraud
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_analytics.datasets.synthetic_healthcare import generate_synthetic_healthcare_data
from core_analytics.anomaly.outlier_detector import (
    AnomalyMethod,
    AnomalyRecord,
    AnomalyDetectionSummary,
    IQROutlierDetector,
    ModifiedZScoreOutlierDetector,
    IsolationForestOutlierDetector
)

@pytest.fixture
def healthcare_data():
    """Generates deterministic synthetic data."""
    return generate_synthetic_healthcare_data(seed=42)

# ============================================================================
# 1. IQR / Tukey Fences Tests
# ============================================================================

def test_iqr_detector_ordinary_data_and_injected_outliers(healthcare_data):
    """Test IQR detector on claims billed_amount with known high/negative outliers."""
    _, _, claims_df = healthcare_data
    detector = IQROutlierDetector(k=1.5)
    detector.fit(claims_df, features=['billed_amount', 'paid_amount'])

    summary = detector.score(claims_df, id_column='claim_id', table_name='claims')
    assert summary.method == AnomalyMethod.IQR
    assert summary.total_evaluated == 1000
    assert summary.anomaly_count > 0
    assert summary.anomaly_rate_pct > 0.0

    # Ensure upper-bound extreme billed amount claim (e.g. CLM-100933 with 5523.15) is detected
    anom_ids = [rec.record_id for rec in summary.anomalous_records]
    max_claim_id = claims_df.loc[claims_df['billed_amount'].idxmax(), 'claim_id']
    assert max_claim_id in anom_ids

    # Check structure and remediation note (never labels as fraud)
    rec = summary.anomalous_records[0]
    assert isinstance(rec, AnomalyRecord)
    assert "fraud" not in rec.remediation_note.lower()
    assert rec.anomaly_score >= 0.0
    assert len(rec.feature_explanations) > 0

def test_iqr_detector_fit_and_reuse_on_new_data():
    """Verify parameters are fitted on training data and can score new unseen data."""
    train_df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'val': [10, 11, 10, 12, 10, 11, 12, 10, 11, 12]})
    test_df = pd.DataFrame({'id': [101, 102], 'val': [11, 999]}) # 999 is extreme outlier

    detector = IQROutlierDetector(k=1.5)
    detector.fit(train_df, features=['val'])
    
    summary = detector.score(test_df, id_column='id')
    assert summary.total_evaluated == 2
    assert summary.anomaly_count == 1
    assert summary.anomalous_records[0].record_id == 102

def test_iqr_invalid_parameters():
    """Ensure invalid multiplier k raises ValueError."""
    with pytest.raises(ValueError):
        IQROutlierDetector(k=-1.0)
    with pytest.raises(ValueError):
        IQROutlierDetector(k=0.0)

# ============================================================================
# 2. Modified Z-Score Tests
# ============================================================================

def test_modified_z_score_detector(healthcare_data):
    """Test Modified Z-score using Median Absolute Deviation."""
    _, _, claims_df = healthcare_data
    detector = ModifiedZScoreOutlierDetector(threshold=3.5)
    detector.fit(claims_df, features=['billed_amount'])

    summary = detector.score(claims_df, id_column='claim_id', table_name='claims')
    assert summary.method == AnomalyMethod.MODIFIED_Z_SCORE
    assert summary.total_evaluated == 1000
    assert summary.anomaly_count > 0

    # Check that remediation note does not contain fraud accusation
    assert "fraud" not in summary.anomalous_records[0].remediation_note.lower()

def test_modified_z_score_constant_columns():
    """Verify detector safely handles constant columns (MAD == 0) without ZeroDivisionError."""
    const_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'constant_feat': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    })
    detector = ModifiedZScoreOutlierDetector(threshold=3.5)
    detector.fit(const_df, features=['constant_feat'])
    
    # Score on identical data -> 0 anomalies
    summary = detector.score(const_df, id_column='id')
    assert summary.anomaly_count == 0

    # Score on perturbed data
    test_df = pd.DataFrame({'id': [7], 'constant_feat': [500.0]})
    summary_test = detector.score(test_df, id_column='id')
    assert summary_test.anomaly_count == 1

def test_modified_z_score_invalid_threshold():
    """Verify invalid thresholds raise ValueError."""
    with pytest.raises(ValueError):
        ModifiedZScoreOutlierDetector(threshold=-0.5)

# ============================================================================
# 3. Isolation Forest Tests
# ============================================================================

def test_isolation_forest_detector(healthcare_data):
    """Test multivariate Isolation Forest on claims data."""
    _, _, claims_df = healthcare_data
    detector = IsolationForestOutlierDetector(contamination=0.03, random_state=42)
    detector.fit(claims_df, features=['billed_amount', 'paid_amount'])

    summary = detector.score(claims_df, id_column='claim_id', table_name='claims')
    assert summary.method == AnomalyMethod.ISOLATION_FOREST
    assert summary.total_evaluated == 1000
    assert summary.anomaly_count > 0
    # Contamination roughly 3% -> ~30 anomalies
    assert 15 <= summary.anomaly_count <= 45

    # Check ethical wording
    for rec in summary.anomalous_records:
        assert "fraud" not in rec.remediation_note.lower() or "not fraud" in rec.remediation_note.lower()

def test_isolation_forest_deterministic_repeatability(healthcare_data):
    """Verify that multiple runs with the same seed yield identical anomaly sets."""
    _, _, claims_df = healthcare_data
    det1 = IsolationForestOutlierDetector(contamination=0.04, random_state=42)
    det1.fit(claims_df, features=['billed_amount', 'paid_amount'])
    s1 = det1.score(claims_df, id_column='claim_id')

    det2 = IsolationForestOutlierDetector(contamination=0.04, random_state=42)
    det2.fit(claims_df, features=['billed_amount', 'paid_amount'])
    s2 = det2.score(claims_df, id_column='claim_id')

    ids1 = [r.record_id for r in s1.anomalous_records]
    ids2 = [r.record_id for r in s2.anomalous_records]
    assert ids1 == ids2

def test_isolation_forest_missing_values_imputation():
    """Verify Isolation Forest safely handles missing values through median imputation."""
    df_with_nans = pd.DataFrame({
        'id': [f'REC-{i}' for i in range(20)],
        'feat1': [10.0, np.nan, 12.0, 11.0, 10.5, np.nan, 12.2, 11.8, 10.9, 11.1,
                  10.0, 12.0, 11.0, 10.5, 12.2, 11.8, 10.9, 11.1, 100.0, 10.0],
        'feat2': [5.0, 5.2, np.nan, 5.1, 5.0, 5.2, 5.1, 5.0, 5.3, 5.1,
                  5.0, 5.2, 5.1, 5.0, 5.2, 5.1, 5.0, 5.3, 50.0, 5.1]
    })
    detector = IsolationForestOutlierDetector(contamination=0.1, random_state=42)
    detector.fit(df_with_nans, features=['feat1', 'feat2'])
    summary = detector.score(df_with_nans, id_column='id')

    assert summary.total_evaluated == 20
    assert summary.anomaly_count > 0
    # REC-18 is the clear multivariate outlier (100.0, 50.0)
    anom_ids = [r.record_id for r in summary.anomalous_records]
    assert 'REC-18' in anom_ids

def test_isolation_forest_invalid_contamination():
    """Verify out-of-range contamination values raise ValueError."""
    with pytest.raises(ValueError):
        IsolationForestOutlierDetector(contamination=0.0)
    with pytest.raises(ValueError):
        IsolationForestOutlierDetector(contamination=0.7)

# ============================================================================
# 4. Common Edge Cases
# ============================================================================

def test_empty_dataset_scoring():
    """Verify detectors score empty datasets cleanly without crashing."""
    empty_df = pd.DataFrame(columns=['id', 'val'])
    train_df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'val': [10, 12, 11, 10, 12, 11, 10, 12, 11, 10, 12]})

    # IQR
    iqr = IQROutlierDetector().fit(train_df, ['val'])
    s_iqr = iqr.score(empty_df, id_column='id')
    assert s_iqr.total_evaluated == 0
    assert s_iqr.anomaly_count == 0

    # Modified Z-Score
    modz = ModifiedZScoreOutlierDetector().fit(train_df, ['val'])
    s_modz = modz.score(empty_df, id_column='id')
    assert s_modz.total_evaluated == 0
    assert s_modz.anomaly_count == 0

    # Isolation Forest
    iso = IsolationForestOutlierDetector(contamination=0.1, random_state=42).fit(train_df, ['val'])
    s_iso = iso.score(empty_df, id_column='id')
    assert s_iso.total_evaluated == 0
    assert s_iso.anomaly_count == 0

def test_small_dataset_fit_guard():
    """Isolation Forest should raise ValueError if sample size is too small for training."""
    tiny_df = pd.DataFrame({'id': [1, 2], 'val': [10.0, 20.0]})
    iso = IsolationForestOutlierDetector()
    with pytest.raises(ValueError, match="too small"):
        iso.fit(tiny_df, features=['val'])

def test_non_numeric_and_missing_feature_handling():
    """Verify non-numeric or missing columns in feature list do not crash detectors."""
    df = pd.DataFrame({
        'id': [f'ID-{i}' for i in range(15)],
        'text_col': ['alpha', 'beta', 'gamma'] * 5,
        'num_col': [10.0 + i for i in range(15)]
    })
    detector = IQROutlierDetector()
    detector.fit(df, features=['non_existent_column', 'text_col', 'num_col'])
    assert 'num_col' in detector.fitted_bounds_
    assert 'text_col' not in detector.fitted_bounds_

    summary = detector.score(df, id_column='id')
    assert summary.total_evaluated == 15
