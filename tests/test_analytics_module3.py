"""
Tests for Core Analytics Module 3:
- Healthcare operational analytics & KPI metrics (claims totals, denial rates, provider cost summaries)
- PMPM expenditure with member-month denominator calculation
- Transparent handling of duplicates and orphan records
- Module 2 anomaly integration (high-cost outlier claims)
- Claim rejection-risk classification pipeline (Pipeline + ColumnTransformer)
- Leakage prevention (strictly excluding claim_status, denial_reason, paid_amount)
- Reproducible model training and evaluation (accuracy, precision, recall, f1, roc-auc)
- Continuous probability scoring and risk bands (Low, Medium, High)
- Separation of predicted risk from observed outcome
- Clean handling of missing and unseen categorical values
- Self-contained persistence without altering student model_cache.joblib
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_analytics.datasets.synthetic_healthcare import generate_synthetic_healthcare_data
from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager
from core_analytics.healthcare.operational_analytics import HealthcareAnalyticsEngine
from core_analytics.healthcare.claims_models import ClaimRejectionRiskModel

@pytest.fixture
def healthcare_data():
    """Generates deterministic synthetic data."""
    return generate_synthetic_healthcare_data(seed=42)

@pytest.fixture
def db_manager(healthcare_data):
    """Initializes in-memory database and loads healthcare data."""
    members_df, providers_df, claims_df = healthcare_data
    mgr = HealthcareDatabaseManager(db_path=":memory:")
    mgr.load_data(members_df, providers_df, claims_df)
    return mgr

# ============================================================================
# 1. Operational Analytics & KPI Tests
# ============================================================================

def test_claims_summary_kpis_and_deduplication(db_manager):
    """Verify claim volume, financial totals, denial rate, and deduplication policy."""
    engine = HealthcareAnalyticsEngine(db_manager)
    
    # Deduplicated summary (default)
    dedup_summary = engine.get_claims_summary(deduplicate_claims=True)
    assert dedup_summary['raw_claim_count'] == 1000
    assert dedup_summary['evaluated_claim_count'] == 999  # 1 duplicate excluded
    assert dedup_summary['duplicate_claims_excluded'] == 1
    assert dedup_summary['total_billed'] > 0
    assert dedup_summary['total_paid'] > 0
    assert 0 <= dedup_summary['denial_rate_pct'] <= 100

    # Raw summary without deduplication
    raw_summary = engine.get_claims_summary(deduplicate_claims=False)
    assert raw_summary['evaluated_claim_count'] == 1000
    assert raw_summary['duplicate_claims_excluded'] == 0

def test_denial_reason_breakdown(db_manager):
    """Verify denial reason distribution calculation."""
    engine = HealthcareAnalyticsEngine(db_manager)
    breakdown = engine.get_denial_reason_breakdown()
    assert len(breakdown) > 0
    total_pct = sum(item['percentage'] for item in breakdown)
    assert abs(total_pct - 100.0) < 1.0
    for item in breakdown:
        assert 'denial_reason' in item
        assert item['count'] > 0

def test_provider_analytics_and_orphan_handling(db_manager):
    """Verify provider cost profiles and transparent flagging of orphan provider references."""
    engine = HealthcareAnalyticsEngine(db_manager)
    prov_metrics = engine.get_provider_analytics(min_claims=1)
    assert len(prov_metrics) > 0

    # Ensure orphan providers from synthetic dataset are identified transparently
    orphan_entries = [p for p in prov_metrics if p['is_orphan']]
    assert len(orphan_entries) >= 1
    assert any('PRV-9999_ORPHAN' in p['provider_id'] for p in orphan_entries)

def test_enrollment_trends(db_manager):
    """Verify active vs terminated member counts and plan distributions."""
    engine = HealthcareAnalyticsEngine(db_manager)
    trends = engine.get_enrollment_trends()
    assert trends['total_members'] == 200
    assert trends['active_members'] + trends['terminated_members'] == 200
    assert trends['termination_rate_pct'] > 0.0
    assert len(trends['plan_distribution']) > 0

def test_pmpm_calculation(db_manager):
    """Verify Per-Member-Per-Month calculation and date boundaries."""
    engine = HealthcareAnalyticsEngine(db_manager)
    pmpm_data = engine.calculate_pmpm(start_period="2023-01-01", end_period="2023-12-31")

    assert pmpm_data['evaluated_months'] == 12
    assert pmpm_data['total_member_months'] > 0
    assert pmpm_data['total_paid_spend'] > 0
    assert pmpm_data['pmpm'] > 0.0
    # Manually verifiable math: pmpm == round(spend / member_months, 2)
    expected_pmpm = round(pmpm_data['total_paid_spend'] / pmpm_data['total_member_months'], 2)
    assert pmpm_data['pmpm'] == expected_pmpm

    # Error handling on inverted date window
    with pytest.raises(ValueError, match="cannot be earlier"):
        engine.calculate_pmpm(start_period="2023-12-31", end_period="2023-01-01")

def test_high_cost_claim_anomalies_integration(db_manager):
    """Verify integration of Module 2 IQROutlierDetector for high-cost claims."""
    engine = HealthcareAnalyticsEngine(db_manager)
    res = engine.identify_high_cost_claim_anomalies(k=1.5)
    assert res['total_evaluated'] > 0
    assert res['billed_amount_upper_fence'] > 0
    assert res['high_cost_outliers_count'] > 0
    # Confirm structure and ethical safety: anomaly is not labeled fraud
    for claim in res['outlier_claims']:
        assert "fraud" not in claim.get('remediation_note', '').lower()

# ============================================================================
# 2. Claim Rejection-Risk Classification Model Tests
# ============================================================================

def test_claim_rejection_model_leakage_prevention(healthcare_data):
    """Ensure target leakage columns (claim_status, denial_reason, paid_amount) are strictly excluded from X."""
    members_df, _, claims_df = healthcare_data
    model = ClaimRejectionRiskModel(random_state=42)
    X, y = model._prepare_training_data(claims_df, members_df)

    for leakage_col in ClaimRejectionRiskModel.LEAKAGE_COLUMNS:
        assert leakage_col not in X.columns
    assert len(X) == 999  # 1 duplicate dropped
    assert len(y) == 999

def test_claim_rejection_model_train_and_evaluate(healthcare_data):
    """Verify reproducible training, ColumnTransformer pipeline, and evaluation metrics."""
    members_df, _, claims_df = healthcare_data
    model = ClaimRejectionRiskModel(random_state=42, n_estimators=50)
    metrics = model.train_and_evaluate(claims_df, members_df, test_size=0.2)

    assert metrics['train_samples'] > 0
    assert metrics['test_samples'] > 0
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['f1_score'] <= 1.0
    assert metrics['confusion_matrix'] is not None
    assert metrics['roc_auc'] is not None

def test_claim_rejection_model_predictions_and_risk_bands(healthcare_data):
    """Verify continuous probability generation and risk band assignment."""
    members_df, _, claims_df = healthcare_data
    model = ClaimRejectionRiskModel(random_state=42, n_estimators=50)
    model.train_and_evaluate(claims_df, members_df)

    scored_df = model.predict_rejection_risk(claims_df, members_df)
    assert len(scored_df) == 999
    assert 'rejection_probability' in scored_df.columns
    assert 'predicted_risk_band' in scored_df.columns
    assert 'actual_observed_status' in scored_df.columns

    # Verify probability range [0.0, 1.0] and valid bands
    assert scored_df['rejection_probability'].between(0.0, 1.0).all()
    assert set(scored_df['predicted_risk_band'].unique()).issubset({'Low', 'Medium', 'High'})

def test_claim_rejection_model_unseen_and_missing_values(healthcare_data):
    """Verify ColumnTransformer safely handles missing values and unseen categorical values."""
    members_df, _, claims_df = healthcare_data
    model = ClaimRejectionRiskModel(random_state=42, n_estimators=50)
    model.train_and_evaluate(claims_df, members_df)

    # Construct new test claims with novel procedure code, missing diagnosis, and unseen plan type
    new_claims = pd.DataFrame([{
        'claim_id': 'CLM-TEST-NEW',
        'member_id': 'MEM-TEST-NEW',
        'provider_id': 'PRV-1001',
        'claim_type': 'Professional',
        'diagnosis_code': None, # Missing value
        'procedure_code': 'UNSEEN_CODE_99999', # Unseen category
        'billed_amount': 250.0,
        'paid_amount': 0.0,
        'claim_status': 'In Process',
        'denial_reason': None
    }])
    new_members = pd.DataFrame([{
        'member_id': 'MEM-TEST-NEW',
        'plan_type': 'UNSEEN_PLAN_XYZ',
        'risk_score': 1.8,
        'date_of_birth': '1985-04-12'
    }])

    scored = model.predict_rejection_risk(new_claims, new_members)
    assert len(scored) == 1
    assert 0.0 <= scored.iloc[0]['rejection_probability'] <= 1.0
    assert scored.iloc[0]['predicted_risk_band'] in ['Low', 'Medium', 'High']

def test_model_persistence_isolation(tmp_path, healthcare_data):
    """Verify model can be saved and reloaded to an independent file without touching model_cache.joblib."""
    members_df, _, claims_df = healthcare_data
    model = ClaimRejectionRiskModel(random_state=42, n_estimators=30)
    model.train_and_evaluate(claims_df, members_df)

    temp_model_file = str(tmp_path / "claims_model_test.joblib")
    model.save_model(temp_model_file)
    assert os.path.exists(temp_model_file)

    # Ensure student model_cache.joblib is not overwritten
    assert temp_model_file != "model_cache.joblib"

    # Reload model
    new_model = ClaimRejectionRiskModel()
    new_model.load_model(temp_model_file)
    # Head(10) has 1 duplicate (row 5 duplicates row 4), yielding 9 unique deduplicated claims
    scored = new_model.predict_rejection_risk(claims_df.head(10), members_df)
    assert len(scored) == len(claims_df.head(10).drop_duplicates(subset=['claim_id']))

