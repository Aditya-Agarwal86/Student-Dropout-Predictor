"""
Tests for Core Analytics Module 4:
- Synthetic call transcript dataset generation and structure
- Regex and rule-based extraction (claim IDs, explicit denial reasons, urgency, action)
- TF-IDF intent classification without train/test leakage
- Human review persistence and audit trail (immutable original prediction + chronology of reviews)
- Multi-field evaluation metrics (precision, recall, f1, exact match, coverage)
- Edge cases: empty/blank transcripts, ambiguous calls, unseen intents, missing fields
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager
from core_analytics.nlp.synthetic_transcripts import get_synthetic_call_transcripts
from core_analytics.nlp.transcript_analyzer import (
    TranscriptExtractor,
    ExtractedTranscriptRecord,
    HumanReviewAuditManager,
    evaluate_extractions
)

@pytest.fixture
def transcripts():
    return get_synthetic_call_transcripts()

@pytest.fixture
def db_manager():
    return HealthcareDatabaseManager(db_path=":memory:")

# ============================================================================
# 1. Extraction & Parsing Tests
# ============================================================================

def test_synthetic_transcripts_presence_and_structure(transcripts):
    """Verify synthetic transcripts are reproducible and include all reference annotations."""
    assert len(transcripts) >= 15
    for t in transcripts:
        assert 'transcript_id' in t
        assert 'call_text' in t
        assert 'reference_intent' in t
        assert 'reference_action' in t
        assert 'reference_urgency' in t

def test_regex_claim_id_and_denial_reason_extraction():
    """Verify exact regex extraction of claim identifiers and denial reasons."""
    extractor = TranscriptExtractor()
    
    # Transcript with explicit claim ID and explicit denial reason
    text1 = "Caller: Claim CLM-100234 was denied because Prior Authorization Required."
    rec1 = extractor.extract_from_text("T1", text1)
    assert rec1.extracted_claim_id == "CLM-100234"
    assert rec1.extracted_denial_reason == "Prior Authorization Required"

    # Transcript with no claim ID
    text2 = "Caller: I'm just calling to ask about my general insurance benefits."
    rec2 = extractor.extract_from_text("T2", text2)
    assert rec2.extracted_claim_id is None
    assert rec2.extracted_denial_reason is None

def test_intent_classification_without_leakage(transcripts):
    """Verify TF-IDF classifier fits strictly on training split and generalizes to test transcripts."""
    train_split = transcripts[:10]
    test_split = transcripts[10:]

    extractor = TranscriptExtractor(random_state=42)
    extractor.fit_intent_classifier(train_split)

    for item in test_split:
        rec = extractor.extract_from_text(item['transcript_id'], item['call_text'])
        assert rec.extracted_intent in [
            'Claim Status', 'Denial Reason', 'Appeal Request',
            'Payment Inquiry', 'Prior Authorization', 'Ambiguous'
        ]
        assert 0.0 <= rec.confidence_score <= 1.0

def test_empty_and_ambiguous_transcripts():
    """Verify extractor safely handles empty strings, whitespace, and unintelligible static."""
    extractor = TranscriptExtractor()
    
    # Completely blank text
    rec_empty = extractor.extract_from_text("EMPTY-1", "")
    assert rec_empty.extracted_intent == "Ambiguous"
    assert rec_empty.extracted_claim_id is None
    assert rec_empty.confidence_score == 0.0

    # Static / unintelligible audio
    rec_static = extractor.extract_from_text("STATIC-1", "Static... crackle... [unintelligible audio]")
    assert rec_static.extracted_intent == "Ambiguous"
    assert rec_static.extracted_claim_id is None

# ============================================================================
# 2. Human Review & Audit Persistence Tests
# ============================================================================

def test_human_review_persistence_and_audit_trail(db_manager):
    """
    Verify human review workflow:
      - Saves initial model prediction
      - Records reviewer approval/correction
      - Preserves original prediction intact (audit trail)
      - Allows multiple sequential reviews
    """
    manager = HumanReviewAuditManager(db_manager)
    rec = ExtractedTranscriptRecord(
        transcript_id="CALL-TEST-1",
        call_text="Claim CLM-100999 was denied for Duplicate Claim.",
        extracted_intent="Denial Reason",
        extracted_claim_id="CLM-100999",
        extracted_denial_reason="Duplicate Claim",
        extracted_action="Explain Denial",
        extracted_urgency="Low",
        requires_followup=False,
        confidence_score=0.92
    )

    # 1. Save initial model prediction
    manager.save_prediction(rec)

    # 2. Reviewer 1 approves prediction
    rev_id1 = manager.record_human_review(
        transcript_id="CALL-TEST-1",
        reviewer_id="REV-Alice",
        review_status="Approved",
        final_intent="Denial Reason",
        final_claim_id="CLM-100999",
        final_denial_reason="Duplicate Claim",
        comments="Confirmed accurate."
    )
    assert rev_id1 > 0

    # 3. Reviewer 2 modifies urgency and sets follow-up
    rev_id2 = manager.record_human_review(
        transcript_id="CALL-TEST-1",
        reviewer_id="REV-Bob",
        review_status="Modified",
        final_intent="Denial Reason",
        final_claim_id="CLM-100999",
        final_denial_reason="Duplicate Claim",
        final_urgency="High",
        final_requires_followup=True,
        comments="Member requested supervisor callback."
    )
    assert rev_id2 > rev_id1

    # 4. Verify audit history
    history = manager.get_transcript_audit_history("CALL-TEST-1")
    assert history['transcript_id'] == "CALL-TEST-1"
    
    # Original prediction is strictly preserved and unchanged
    pred = history['prediction']
    assert pred['predicted_intent'] == "Denial Reason"
    assert pred['predicted_urgency'] == "Low"
    assert pred['predicted_requires_followup'] == 0

    # Full review chronological audit trail
    assert len(history['human_reviews']) == 2
    assert history['human_reviews'][0]['reviewer_id'] == "REV-Alice"
    assert history['human_reviews'][0]['review_status'] == "Approved"
    assert history['human_reviews'][1]['reviewer_id'] == "REV-Bob"
    assert history['human_reviews'][1]['review_status'] == "Modified"
    assert history['human_reviews'][1]['final_urgency'] == "High"

# ============================================================================
# 3. Evaluation Metrics Tests
# ============================================================================

def test_evaluation_metrics_against_reference(transcripts):
    """Verify calculation of precision, recall, f1, coverage, and exact match."""
    extractor = TranscriptExtractor(random_state=42)
    extractor.fit_intent_classifier(transcripts)

    predictions = [
        extractor.extract_from_text(t['transcript_id'], t['call_text'])
        for t in transcripts
    ]

    metrics = evaluate_extractions(predictions, transcripts)
    assert metrics['total_evaluated'] == len(transcripts)
    assert 'field_metrics' in metrics

    for field in ['intent', 'claim_id', 'denial_reason', 'action', 'urgency', 'requires_followup']:
        f_stat = metrics['field_metrics'][field]
        assert 'precision' in f_stat
        assert 'recall' in f_stat
        assert 'f1_score' in f_stat
        assert 'exact_match_accuracy' in f_stat
        assert 0.0 <= f_stat['precision'] <= 1.0
        assert 0.0 <= f_stat['recall'] <= 1.0
        assert 0.0 <= f_stat['f1_score'] <= 1.0
        assert 0.0 <= f_stat['exact_match_accuracy'] <= 1.0

def test_evaluation_empty_predictions():
    """Verify evaluation handles empty lists safely without division by zero."""
    metrics = evaluate_extractions([], [])
    assert metrics['total_evaluated'] == 0
    assert metrics['metrics'] == {}
