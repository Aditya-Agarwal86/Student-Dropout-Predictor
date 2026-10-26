"""
Tests for Core Analytics Module 5:
- Process mining event-log generation and validation
- Correct chronological event ordering and timestamp parsing
- Case cycle time and stage duration calculations
- Repeated activity and rework loop detection
- Override identification and categorical reason distribution
- Common process variants and transition matrices
- Dynamic bottleneck identification
- Handling edge cases: missing/invalid timestamps, duplicate events, single-event cases, empty datasets
- Structured analytical DataFrame outputs and CSV export
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_analytics.process_mining.synthetic_events import generate_synthetic_claims_event_log
from core_analytics.process_mining.process_analyzer import ProcessMiningEngine

@pytest.fixture
def raw_event_log():
    return generate_synthetic_claims_event_log(n_cases=100, seed=42)

# ============================================================================
# 1. Event Log Ingestion, Schema & Data Quality Tests
# ============================================================================

def test_synthetic_event_log_generation(raw_event_log):
    """Verify synthetic event log contains required columns and multiple cases."""
    assert len(raw_event_log) > 0
    for col in ProcessMiningEngine.REQUIRED_COLUMNS:
        assert col in raw_event_log.columns
    assert raw_event_log['case_id'].nunique() == 100

def test_event_log_cleaning_and_ordering(raw_event_log):
    """
    Verify clean ordering:
      - Chronological ordering of events within case_id
      - Audit counting of missing timestamps and duplicate events
    """
    engine = ProcessMiningEngine(raw_event_log)
    audit = engine.dq_audit
    assert audit['raw_events'] == len(raw_event_log)
    assert audit['missing_timestamps'] >= 1 # Injected in CASE-10002
    assert audit['duplicates_removed'] >= 1 # Injected in CASE-10003
    assert audit['clean_events'] > 0

    # Ensure clean_log is sorted strictly ascending per case_id
    for _, group in engine.clean_log.groupby('case_id'):
        ts_list = group['parsed_timestamp'].tolist()
        assert ts_list == sorted(ts_list)

def test_single_event_case_handling(raw_event_log):
    """Verify single-event cases (CASE-10000) have cycle_time = 0.0 without errors."""
    engine = ProcessMiningEngine(raw_event_log)
    case_df = engine.get_case_summaries()
    
    single_case = case_df[case_df['case_id'] == 'CASE-10000']
    assert len(single_case) == 1
    assert single_case.iloc[0]['event_count'] == 1
    assert single_case.iloc[0]['cycle_time_hours'] == 0.0
    assert bool(single_case.iloc[0]['has_rework']) is False


# ============================================================================
# 2. Cycle Time, Stage Durations, and Transition Tests
# ============================================================================

def test_case_summaries_and_cycle_time(raw_event_log):
    """Verify case-level cycle time calculations (hours and days)."""
    engine = ProcessMiningEngine(raw_event_log)
    case_df = engine.get_case_summaries()
    assert len(case_df) > 0
    assert 'cycle_time_hours' in case_df.columns
    assert 'cycle_time_days' in case_df.columns
    assert (case_df['cycle_time_hours'] >= 0.0).all()
    # Check days formula: days == round(hours / 24, 2)
    sample = case_df.iloc[1]
    assert sample['cycle_time_days'] == round(sample['cycle_time_hours'] / 24.0, 2)

def test_transition_matrix_durations(raw_event_log):
    """Verify direct-follows activity transitions and duration statistics."""
    engine = ProcessMiningEngine(raw_event_log)
    trans_df = engine.get_transition_matrix()
    assert not trans_df.empty
    assert 'from_activity' in trans_df.columns
    assert 'to_activity' in trans_df.columns
    assert 'avg_duration_hours' in trans_df.columns
    assert 'median_duration_hours' in trans_df.columns
    assert (trans_df['transition_count'] > 0).all()
    assert (trans_df['avg_duration_hours'] >= 0.0).all()

# ============================================================================
# 3. Process Variants, Rework Loops, and Overrides
# ============================================================================

def test_process_variants_discovery(raw_event_log):
    """Verify discovery of process variants and variant ranking."""
    engine = ProcessMiningEngine(raw_event_log)
    var_df = engine.get_process_variants()
    assert not var_df.empty
    assert 'variant_path' in var_df.columns
    assert 'case_count' in var_df.columns
    assert 'case_percentage' in var_df.columns
    assert abs(var_df['case_percentage'].sum() - 100.0) < 1.0

def test_rework_loop_detection(raw_event_log):
    """Verify identification of rework loops where activities are repeated."""
    engine = ProcessMiningEngine(raw_event_log)
    case_df = engine.get_case_summaries()
    rework_cases = case_df[case_df['has_rework'] == True]
    assert len(rework_cases) > 0
    assert (rework_cases['rework_events_count'] >= 1).all()

def test_override_analysis(raw_event_log):
    """Verify tracking of manual overrides and reason code breakdown."""
    engine = ProcessMiningEngine(raw_event_log)
    override_data = engine.get_override_analysis()
    assert override_data['total_override_events'] > 0
    assert override_data['affected_cases_count'] > 0
    assert len(override_data['reason_breakdown']) > 0
    for r in override_data['reason_breakdown']:
        assert 'reason' in r
        assert r['count'] > 0

# ============================================================================
# 4. Bottleneck Detection & CSV Export Tests
# ============================================================================

def test_bottleneck_identification(raw_event_log):
    """Verify identification of transition bottlenecks based on statistical thresholding."""
    engine = ProcessMiningEngine(raw_event_log)
    bottlenecks = engine.identify_bottlenecks(threshold_std=1.0)
    assert isinstance(bottlenecks, pd.DataFrame)
    if not bottlenecks.empty:
        assert 'bottleneck_severity_score' in bottlenecks.columns
        assert (bottlenecks['bottleneck_severity_score'] > 0).all()

def test_csv_export(tmp_path, raw_event_log):
    """Verify clean export of process mining analytical outputs to CSV."""
    engine = ProcessMiningEngine(raw_event_log)
    export_files = engine.export_all_metrics_to_csv(export_dir=str(tmp_path))
    for key, fpath in export_files.items():
        assert os.path.exists(fpath)
        assert os.path.getsize(fpath) > 0

# ============================================================================
# 5. Robustness & Edge Cases
# ============================================================================

def test_empty_dataset_handling():
    """Verify empty event log is handled gracefully without unhandled exceptions."""
    empty_df = pd.DataFrame(columns=ProcessMiningEngine.REQUIRED_COLUMNS)
    engine = ProcessMiningEngine(empty_df)
    assert engine.get_case_summaries().empty
    assert engine.get_process_variants().empty
    assert engine.get_transition_matrix().empty
    assert engine.identify_bottlenecks().empty
    assert engine.get_override_analysis()['total_overrides'] == 0

def test_missing_schema_columns_raises_value_error():
    """Verify ValueError is raised if mandatory schema columns are absent."""
    invalid_df = pd.DataFrame({'case_id': [1], 'activity': ['A']})
    with pytest.raises(ValueError, match="missing required schema columns"):
        ProcessMiningEngine(invalid_df)
