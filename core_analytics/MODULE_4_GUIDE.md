# Core Analytics Module 4: NLP & Document AI for Call Transcript Extraction with Human-Review Evaluation

This guide documents the synthetic call transcript schema, entity extraction heuristics, TF-IDF intent classifier, human-review SQLite audit trail, multi-field evaluation metrics, limitations, and verification commands for **Module 4**.

---

## 1. Synthetic Call Transcript Dataset

Implemented in [`core_analytics/nlp/synthetic_transcripts.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/nlp/synthetic_transcripts.py).
- **Fictional & Reproducible**: 15 representative call transcripts generated deterministically with gold-standard annotations.
- **Transcript Categories Included**:
  - `Claim Status`: Routine status checks on pending/approved claims.
  - `Denial Reason`: Member questions regarding explanation of benefits denial rationale.
  - `Appeal Request`: Provider and member disputes regarding timeliness, coding, or duplicates.
  - `Payment Inquiry`: Inquiries regarding discrepancy between billed and paid checks.
  - `Prior Authorization`: Inquiries regarding pre-certification requirements for scheduled surgeries/procedures.
  - `Ambiguous`: Unintelligible audio, static, misdirected inquiries, or missing intent.
- **Ethical Integrity**: Does not use real patient identifiers, HIPAA-protected health information (PHI), proprietary company transcripts, or external cloud APIs.

---

## 2. Extraction & NLP Architecture

Implemented in [`core_analytics/nlp/transcript_analyzer.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/nlp/transcript_analyzer.py).

### A. Transparent Entity Extraction
- **Claim Identifiers**: Regex pattern `\b(CLM-\d{5,7})\b` extracts only explicitly mentioned claim IDs. Missing or unmentioned claim IDs are left as `None` (never invented).
- **Explicit Denial Reasons**: Keyword and phrase matching for verified standard rejection reasons:
  - *Prior Authorization Required*, *Duplicate Claim*, *Service Not Covered*, *Invalid Diagnosis*, *Timely Filing Exceeded*.
- **Urgency Classification**: Rule-based keyword matching:
  - *High*: `urgent`, `immediately`, `emergency`, `asap`, `tomorrow morning`, `expedite`.
  - *Medium*: `scheduled`, `resubmit`, `difference`, `next tuesday`, `holding`.
  - *Low*: `check`, `status`, `two weeks ago`, `cleared`, `inquiry`.
- **Action Mapping**: Deterministic intent-to-action routing (`Check Status`, `Explain Denial`, `Process Appeal`, `Submit Auth`, `Escalate Payment`, `None`).
- **Follow-up Requirement**: Flagged as `True` if intent represents an appeal, payment escalation, or high urgency.

### B. Lightweight Intent Classification
- **Pipeline**: Scikit-learn `TfidfVectorizer(ngram_range=(1,2), stop_words='english')` paired with `LogisticRegression(C=5.0)`.
- **Zero Train/Test Leakage**: The classifier fits strictly on designated training splits.
- **Confidence Scoring**: Extracts the maximum class posterior probability $\max P(c \mid \text{text})$.

---

## 3. Human Review & Audit Persistence Data Model

Implemented in `HumanReviewAuditManager` using the existing SQLite database.

```sql
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
```

### Reviewer Workflow Principles
1. **Immutable Ingestion**: Initial model predictions and original transcript texts are stored in `transcript_predictions` and are **never** overwritten by human modifications.
2. **Review Chronology**: Review decisions (`Approved`, `Rejected`, `Modified`) and corrected values are recorded in `transcript_human_reviews` with reviewer ID and timestamp.
3. **Audit History**: Complete chronology is queryable via `get_transcript_audit_history(transcript_id)`.

---

## 4. Multi-Field Evaluation Metrics

Implemented in `evaluate_extractions()`.
- **Field-Level Evaluation**: Evaluates `intent`, `claim_id`, `denial_reason`, `action`, `urgency`, and `requires_followup`.
- **Calculated Metrics**:
  - **Exact Match Accuracy**: Proportion of transcripts where prediction exactly equals reference value.
  - **Precision**: $\frac{TP}{TP + FP}$ for extracted non-null entities.
  - **Recall**: $\frac{TP}{TP + FN}$ capturing reference ground-truth entities.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **Coverage**: Proportion of transcripts where an entity was predicted.
  - **Missing-Field Rate**: $1.0 - \text{Coverage}$.
- **Zero-Division Handling**: Fields with zero occurrences default safely to $1.0$ (if $TP+FP=0$) and $F_1=0.0$.

---

## 5. Assumptions and Limitations

1. **Lightweight Rule & ML Approach**:
   - Uses regex and scikit-learn TF-IDF rather than heavyweight 10GB+ LLM models or paid cloud APIs (e.g. OpenAI/Anthropic/AWS Comprehend), ensuring complete offline testability and zero external cost.
2. **Sample Size Scope**:
   - The synthetic dataset ($N=15$) provides comprehensive coverage for unit testing and ergonomic auditing, but is not intended to represent full production variance across hundreds of distinct payer dialects.

---

## 6. Execution and Verification Commands

### Run Module 4 Tests:
```powershell
python -m pytest tests/test_analytics_module4.py -v
```

### Run Complete Pytest Suite (55 Tests across Modules 1–4 + Web App):
```powershell
python -m pytest
```

### Run Baseline Student Dropout ML Pipeline:
```powershell
python test_ml.py
```

### Run Environment Setup Verification:
```powershell
python setup.py
```
