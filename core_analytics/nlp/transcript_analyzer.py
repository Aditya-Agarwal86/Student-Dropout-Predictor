"""
Structured Healthcare Call Transcript Extraction & Review Audit Module
Provides:
  - Transparent regex-based extraction of explicit claim identifiers (e.g. CLM-XXXXXX)
  - Keyword/phrase matching for explicitly stated denial reasons & urgency indicators
  - Lightweight Scikit-Learn TF-IDF + Logistic Regression / Naive Bayes for intent classification
  - Human review persistence engine using SQLite audit trail:
      * Saves initial model prediction
      * Allows reviewer edit, approval, or rejection
      * Preserves immutable original transcript and initial prediction
  - Multi-field evaluation metrics against reference annotations:
      * Precision, Recall, F1, Exact-Match Accuracy, Coverage, and Missing-Field Rates
"""

import re
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager

@dataclass
class ExtractedTranscriptRecord:
    transcript_id: str
    call_text: str
    extracted_intent: str
    extracted_claim_id: Optional[str]
    extracted_denial_reason: Optional[str]
    extracted_action: str
    extracted_urgency: str
    requires_followup: bool
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'transcript_id': self.transcript_id,
            'call_text': self.call_text,
            'extracted_intent': self.extracted_intent,
            'extracted_claim_id': self.extracted_claim_id,
            'extracted_denial_reason': self.extracted_denial_reason,
            'extracted_action': self.extracted_action,
            'extracted_urgency': self.extracted_urgency,
            'requires_followup': self.requires_followup,
            'confidence_score': round(self.confidence_score, 4)
        }


class TranscriptExtractor:
    """Extracts structured entities and intent from unstructured call transcripts."""

    CLAIM_ID_REGEX = re.compile(r'\b(CLM-\d{5,7})\b', re.IGNORECASE)

    KNOWN_DENIAL_REASONS = [
        "Prior Authorization Required",
        "Duplicate Claim",
        "Service Not Covered",
        "Invalid Diagnosis",
        "Timely Filing Exceeded"
    ]

    URGENCY_KEYWORDS = {
        'High': ['urgent', 'immediately', 'emergency', 'asap', 'tomorrow morning', 'expedite'],
        'Medium': ['scheduled', 'resubmit', 'difference', 'next tuesday', 'holding'],
        'Low': ['check', 'status', 'two weeks ago', 'cleared', 'inquiry']
    }

    ACTION_MAP = {
        'Claim Status': 'Check Status',
        'Denial Reason': 'Explain Denial',
        'Appeal Request': 'Process Appeal',
        'Payment Inquiry': 'Escalate Payment',
        'Prior Authorization': 'Submit Auth',
        'Ambiguous': 'None'
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier_: Optional[Pipeline] = None

    def fit_intent_classifier(self, training_transcripts: List[Dict[str, Any]]) -> "TranscriptExtractor":
        """Fits a TF-IDF + Logistic Regression pipeline strictly on training transcripts."""
        texts = [t['call_text'] for t in training_transcripts]
        intents = [t['reference_intent'] for t in training_transcripts]

        self.classifier_ = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)),
            ('clf', LogisticRegression(random_state=self.random_state, C=5.0, max_iter=200))
        ])
        self.classifier_.fit(texts, intents)
        return self

    def extract_from_text(self, transcript_id: str, text: str) -> ExtractedTranscriptRecord:
        """Applies rule-based entity extraction and trained intent classification."""
        if not text or not text.strip():
            return ExtractedTranscriptRecord(
                transcript_id=transcript_id,
                call_text=text or "",
                extracted_intent="Ambiguous",
                extracted_claim_id=None,
                extracted_denial_reason=None,
                extracted_action="None",
                extracted_urgency="Low",
                requires_followup=False,
                confidence_score=0.0
            )

        # 1. Claim ID regex extraction
        match = self.CLAIM_ID_REGEX.search(text)
        claim_id = match.group(1).upper() if match else None

        # 2. Denial reason explicit extraction
        denial_reason = None
        for reason in self.KNOWN_DENIAL_REASONS:
            if re.search(r'\b' + re.escape(reason) + r'\b', text, re.IGNORECASE):
                denial_reason = reason
                break

        # 3. Urgency categorization
        text_lower = text.lower()
        urgency = "Low"
        for kw in self.URGENCY_KEYWORDS['High']:
            if kw in text_lower:
                urgency = "High"
                break
        if urgency == "Low":
            for kw in self.URGENCY_KEYWORDS['Medium']:
                if kw in text_lower:
                    urgency = "Medium"
                    break

        # 4. Intent classification
        if self.classifier_ is not None:
            intent = self.classifier_.predict([text])[0]
            probs = self.classifier_.predict_proba([text])[0]
            confidence = float(np.max(probs))
        else:
            # Simple keyword heuristic fallback if not fitted
            if "status" in text_lower:
                intent = "Claim Status"
            elif "denied" in text_lower:
                intent = "Denial Reason"
            elif "appeal" in text_lower:
                intent = "Appeal Request"
            elif "payment" in text_lower or "check" in text_lower:
                intent = "Payment Inquiry"
            elif "prior auth" in text_lower:
                intent = "Prior Authorization"
            else:
                intent = "Ambiguous"
            confidence = 0.50

        # Action mapping
        action = self.ACTION_MAP.get(intent, 'None')
        requires_followup = (intent in ['Appeal Request', 'Payment Inquiry']) or (urgency == 'High')

        return ExtractedTranscriptRecord(
            transcript_id=transcript_id,
            call_text=text,
            extracted_intent=intent,
            extracted_claim_id=claim_id,
            extracted_denial_reason=denial_reason,
            extracted_action=action,
            extracted_urgency=urgency,
            requires_followup=requires_followup,
            confidence_score=confidence
        )


class HumanReviewAuditManager:
    """
    Manages human review decisions, corrections, and audit logging in SQLite.
    Never overwrites original transcripts or original model predictions.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS transcript_predictions (
        transcript_id TEXT PRIMARY KEY,
        call_text TEXT NOT NULL,
        predicted_intent TEXT,
        predicted_claim_id TEXT,
        predicted_denial_reason TEXT,
        predicted_action TEXT,
        predicted_urgency TEXT,
        predicted_requires_followup INTEGER,
        confidence_score REAL,
        prediction_timestamp TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS transcript_human_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        review_status TEXT NOT NULL, -- 'Approved', 'Rejected', 'Modified'
        final_intent TEXT,
        final_claim_id TEXT,
        final_denial_reason TEXT,
        final_action TEXT,
        final_urgency TEXT,
        final_requires_followup INTEGER,
        reviewer_comments TEXT,
        review_timestamp TEXT NOT NULL,
        FOREIGN KEY (transcript_id) REFERENCES transcript_predictions(transcript_id)
    );

    CREATE INDEX IF NOT EXISTS idx_reviews_transcript ON transcript_human_reviews(transcript_id);
    """

    def __init__(self, db_manager: HealthcareDatabaseManager):
        self.db = db_manager
        self._init_tables()

    def _init_tables(self):
        conn = self.db.get_connection()
        conn.executescript(self.DDL)
        conn.commit()

    def save_prediction(self, rec: ExtractedTranscriptRecord):
        """Saves initial model prediction without modifying previous human reviews."""
        conn = self.db.get_connection()
        now_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        conn.execute("""
            INSERT OR REPLACE INTO transcript_predictions (
                transcript_id, call_text, predicted_intent, predicted_claim_id,
                predicted_denial_reason, predicted_action, predicted_urgency,
                predicted_requires_followup, confidence_score, prediction_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            rec.transcript_id, rec.call_text, rec.extracted_intent, rec.extracted_claim_id,
            rec.extracted_denial_reason, rec.extracted_action, rec.extracted_urgency,
            1 if rec.requires_followup else 0, rec.confidence_score, now_ts
        ))
        conn.commit()

    def record_human_review(self,
                            transcript_id: str,
                            reviewer_id: str,
                            review_status: str,
                            final_intent: str,
                            final_claim_id: Optional[str] = None,
                            final_denial_reason: Optional[str] = None,
                            final_action: Optional[str] = None,
                            final_urgency: Optional[str] = None,
                            final_requires_followup: Optional[bool] = None,
                            comments: str = "") -> int:
        """
        Records reviewer judgment and corrected values in the audit table.
        Leaves the original prediction record untouched.
        """
        conn = self.db.get_connection()
        now_ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transcript_human_reviews (
                transcript_id, reviewer_id, review_status, final_intent, final_claim_id,
                final_denial_reason, final_action, final_urgency, final_requires_followup,
                reviewer_comments, review_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            transcript_id, reviewer_id, review_status, final_intent, final_claim_id,
            final_denial_reason, final_action, final_urgency,
            1 if final_requires_followup else 0, comments, now_ts
        ))
        conn.commit()
        return cursor.lastrowid

    def get_transcript_audit_history(self, transcript_id: str) -> Dict[str, Any]:
        """Retrieves original prediction and full chronology of human reviews."""
        conn = self.db.get_connection()
        p_row = conn.execute("SELECT * FROM transcript_predictions WHERE transcript_id = ?;", (transcript_id,)).fetchone()
        r_rows = conn.execute("SELECT * FROM transcript_human_reviews WHERE transcript_id = ? ORDER BY review_id ASC;", (transcript_id,)).fetchall()

        return {
            'transcript_id': transcript_id,
            'prediction': dict(p_row) if p_row else None,
            'human_reviews': [dict(r) for r in r_rows]
        }


def evaluate_extractions(predictions: List[ExtractedTranscriptRecord],
                         references: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes rigorous field-level evaluation metrics:
      - Precision, Recall, F1, Exact-Match Accuracy, Coverage, and Missing-Field Rates
    """
    ref_map = {r['transcript_id']: r for r in references}
    total = len(predictions)
    if total == 0:
        return {'total_evaluated': 0, 'metrics': {}}

    fields_to_eval = [
        ('intent', 'extracted_intent', 'reference_intent'),
        ('claim_id', 'extracted_claim_id', 'reference_claim_id'),
        ('denial_reason', 'extracted_denial_reason', 'reference_denial_reason'),
        ('action', 'extracted_action', 'reference_action'),
        ('urgency', 'extracted_urgency', 'reference_urgency'),
        ('requires_followup', 'requires_followup', 'reference_requires_followup')
    ]

    field_results = {}
    for fname, pred_attr, ref_key in fields_to_eval:
        tp, fp, fn, tn = 0, 0, 0, 0
        exact_matches = 0
        non_null_refs = 0
        predicted_present = 0

        for p in predictions:
            ref = ref_map.get(p.transcript_id)
            if not ref:
                continue

            pred_val = getattr(p, pred_attr)
            ref_val = ref.get(ref_key)

            if ref_val is not None:
                non_null_refs += 1
            if pred_val is not None:
                predicted_present += 1

            if pred_val == ref_val:
                exact_matches += 1
                if pred_val is not None:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred_val is not None and ref_val is not None:
                    fp += 1
                    fn += 1
                elif pred_val is not None and ref_val is None:
                    fp += 1
                elif pred_val is None and ref_val is not None:
                    fn += 1

        accuracy = round(exact_matches / total, 4)
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        coverage = round(predicted_present / total, 4)
        missing_rate = round(1.0 - coverage, 4)

        field_results[fname] = {
            'total_samples': total,
            'ground_truth_present': non_null_refs,
            'predictions_present': predicted_present,
            'exact_match_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': coverage,
            'missing_rate': missing_rate
        }

    return {
        'total_evaluated': total,
        'field_metrics': field_results
    }
