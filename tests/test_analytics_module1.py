"""
Tests for Core Analytics Module 1:
- Synthetic healthcare dataset generation
- Schema initialization and idempotent SQL loading
- Parameterized SQL profiling & KPI calculations
- Reusable declarative Data Quality checks (nulls, ranges, duplicates, referential integrity, financial rules)
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_analytics.datasets.synthetic_healthcare import generate_synthetic_healthcare_data
from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager, SQLDataProfiler
from core_analytics.profiling.data_quality import DataQualityEngine, DQSeverity

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
# 1. Synthetic Data & Schema Tests
# ============================================================================

def test_synthetic_data_generation(healthcare_data):
    """Test deterministic generation of members, providers, and claims."""
    members_df, providers_df, claims_df = healthcare_data
    
    assert len(members_df) == 200
    assert len(providers_df) == 50
    assert len(claims_df) == 1000

    # Ensure required columns exist
    assert 'member_id' in members_df.columns
    assert 'provider_id' in providers_df.columns
    assert 'claim_id' in claims_df.columns
    assert 'billed_amount' in claims_df.columns

def test_database_idempotent_load(healthcare_data):
    """Test that loading data multiple times cleanly replaces data without duplicate growth."""
    members_df, providers_df, claims_df = healthcare_data
    mgr = HealthcareDatabaseManager(db_path=":memory:")
    
    mgr.load_data(members_df, providers_df, claims_df)
    with mgr.get_connection() as conn:
        c1 = conn.execute("SELECT COUNT(*) FROM claims;").fetchone()[0]
        assert c1 == 1000

    # Repeat load
    mgr.load_data(members_df, providers_df, claims_df)
    with mgr.get_connection() as conn:
        c2 = conn.execute("SELECT COUNT(*) FROM claims;").fetchone()[0]
        assert c2 == 1000  # Remains 1000, not 2000

# ============================================================================
# 2. SQL Profiling Tests
# ============================================================================

def test_sql_profiler_table_profile(db_manager):
    """Test table-level statistical profiling."""
    profiler = SQLDataProfiler(db_manager)
    profile = profiler.profile_table('claims')

    assert profile['total_rows'] == 1000
    assert 'billed_amount' in profile['columns']
    assert 'claim_status' in profile['columns']

    # Billed amount numeric stats
    col_stat = profile['columns']['billed_amount']
    assert col_stat['min'] is not None
    assert col_stat['max'] is not None
    assert col_stat['mean'] is not None

def test_sql_profiler_frequency_and_duplicates(db_manager):
    """Test categorical frequency and duplicate detection via SQL."""
    profiler = SQLDataProfiler(db_manager)
    freq = profiler.get_column_frequency('claims', 'claim_status')
    assert len(freq) > 0
    total_pct = sum(f['percentage'] for f in freq)
    assert abs(total_pct - 100.0) < 1.0

    # Duplicate check on claim_id (we intentionally injected a duplicate claim_id)
    dupe_report = profiler.get_duplicate_counts('claims', 'claim_id')
    assert dupe_report['total_duplicate_keys'] >= 1

def test_sql_claims_kpis(db_manager):
    """Test calculation of healthcare claims KPIs."""
    profiler = SQLDataProfiler(db_manager)
    kpis = profiler.calculate_claims_kpis()

    assert kpis['total_claims'] == 1000
    assert kpis['total_billed'] > 0
    assert kpis['total_paid'] > 0
    assert kpis['denied_claims'] > 0
    assert 0 <= kpis['denial_rate_pct'] <= 100
    assert len(kpis['top_denial_reasons']) > 0

# ============================================================================
# 3. Reusable Data Quality (DQ) Engine Tests
# ============================================================================

def test_dq_required_columns():
    """Test validation of required columns."""
    engine = DataQualityEngine()
    df = pd.DataFrame({'a': [1], 'b': [2]})
    
    # Valid
    r1 = engine.check_required_columns(df, 'test', ['a', 'b'])
    assert r1.passed is True
    
    # Missing
    r2 = engine.check_required_columns(df, 'test', ['a', 'b', 'c'])
    assert r2.passed is False
    assert 'c' in r2.affected_identifiers

def test_dq_missing_values(healthcare_data):
    """Test detection of nulls in mandatory fields."""
    _, _, claims_df = healthcare_data
    engine = DataQualityEngine()

    # In our synthetic data, Row 12 has missing member_id
    res = engine.check_no_nulls(claims_df, 'claims', 'member_id', id_column='claim_id')
    assert res.passed is False
    assert res.failure_count >= 1

def test_dq_duplicate_business_keys(healthcare_data):
    """Test duplicate detection on primary/business key."""
    _, _, claims_df = healthcare_data
    engine = DataQualityEngine()

    # In our synthetic data, Row 5 duplicates Row 4's claim_id
    res = engine.check_unique_key(claims_df, 'claims', key_column='claim_id')
    assert res.passed is False
    assert res.failure_count >= 2

def test_dq_numeric_ranges_and_financial_constraints(healthcare_data):
    """Test numeric bounds (e.g. non-negative) and financial logic (paid <= billed)."""
    _, _, claims_df = healthcare_data
    engine = DataQualityEngine()

    # Row 20 has negative billed amount (-150)
    range_res = engine.check_numeric_range(
        claims_df, 'claims', column_name='billed_amount', id_column='claim_id', min_val=0.0
    )
    assert range_res.passed is False
    assert range_res.failure_count >= 1

    # Row 30 has paid (350) > billed (100)
    fin_res = engine.check_financial_constraint(
        claims_df, 'claims', id_column='claim_id', billed_col='billed_amount', paid_col='paid_amount'
    )
    assert fin_res.passed is False
    assert fin_res.failure_count >= 1

def test_dq_date_chronology(healthcare_data):
    """Test chronological consistency: service_end_date >= service_start_date."""
    _, _, claims_df = healthcare_data
    engine = DataQualityEngine()

    # Row 40 has end date earlier than start date
    date_res = engine.check_date_chronology(
        claims_df, 'claims', id_column='claim_id', start_col='service_start_date', end_col='service_end_date'
    )
    assert date_res.passed is False
    assert date_res.failure_count >= 1

def test_dq_referential_integrity(healthcare_data):
    """Test referential integrity check detecting unmatched foreign keys."""
    _, providers_df, claims_df = healthcare_data
    engine = DataQualityEngine()

    # Rows 50 & 51 have provider_id PRV-9999_ORPHAN and PRV-9998_ORPHAN
    ref_res = engine.check_referential_integrity(
        child_df=claims_df,
        child_table='claims',
        child_fk_col='provider_id',
        parent_df=providers_df,
        parent_table='providers',
        parent_pk_col='provider_id',
        id_column='claim_id'
    )
    assert ref_res.passed is False
    assert ref_res.failure_count == 2
    assert 'CLM-100050' in ref_res.affected_identifiers
    assert 'CLM-100051' in ref_res.affected_identifiers


def test_dq_allowed_values(healthcare_data):
    """Test categorical domain validation."""
    members_df, _, _ = healthcare_data
    engine = DataQualityEngine()

    # Member 10 has 'INVALID_GENDER'
    cat_res = engine.check_allowed_values(
        members_df, 'members', column_name='gender', id_column='member_id', allowed_values=['M', 'F', 'U']
    )
    assert cat_res.passed is False
    assert cat_res.failure_count >= 1

def test_dq_outlier_warning(healthcare_data):
    """Test statistical outlier warning detection."""
    _, _, claims_df = healthcare_data
    engine = DataQualityEngine()

    warn_res = engine.check_outlier_warning(
        claims_df, 'claims', column_name='billed_amount', id_column='claim_id', threshold_std=3.0
    )
    assert warn_res.severity == DQSeverity.WARNING
    assert warn_res.total_evaluated == 1000

def test_dq_empty_dataset():
    """Test handling of empty datasets without raising unhandled exceptions."""
    empty_df = pd.DataFrame(columns=['claim_id', 'billed_amount'])
    engine = DataQualityEngine()

    r1 = engine.check_no_nulls(empty_df, 'empty', 'claim_id', id_column='claim_id')
    assert r1.passed is True
    assert r1.failure_count == 0

    r2 = engine.check_numeric_range(empty_df, 'empty', 'billed_amount', id_column='claim_id', min_val=0.0)
    assert r2.passed is True
    assert r2.failure_count == 0
