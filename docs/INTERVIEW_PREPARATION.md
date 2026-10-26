# Technical Interview Preparation Guide: Analytics & AI Intern (HealthPlans.ai)

This document provides concise technical explanations, architectural rationales, and sample interview questions grounded in the actual codebase.

---

## 1. Architecture & Design Decisions

### Q: Why did you preserve the original Student Dropout System instead of starting from scratch?
**Answer**:
In enterprise engineering and healthcare technology, software systems evolve through incremental, backward-compatible extension rather than destructive refactoring. Preserving the working student prediction pipeline (`app.py`, SVM model, `model_cache.joblib`, existing routes/templates) demonstrates disciplined preservation of production behavior. The healthcare analytics suite was introduced as a cleanly isolated package (`core_analytics/`) coordinated by a singleton service layer (`HealthcareAnalyticsService`), leaving the original codebase functional and intact.

---

## 2. Data Quality & Profiling (Module 1)

### Q: How does your reusable Data Quality Engine work?
**Answer**:
The `DataQualityEngine` in `core_analytics/profiling/data_quality.py` provides declarative, parameterized validation rules operating on Pandas DataFrames and SQLite tables:
1. **Schema & Null Checks**: Ensures mandatory columns exist and primary identifiers (`claim_id`, `member_id`) are non-null.
2. **Referential Integrity**: Detects orphan foreign keys (e.g., claims referencing non-existent `provider_id`s) without halting processing.
3. **Domain Constraints**: Enforces non-negative billing amounts, valid status codes, and service chronology (`service_start_date <= service_end_date`).
4. **Financial Consistency**: Validates that `paid_amount <= billed_amount` and confirms `paid_amount == 0` for denied claims.

Every check returns a structured `DataQualityCheckResult` containing pass/fail flags, severity (`CRITICAL`, `WARNING`, `INFO`), and exact failing record IDs for audit logging.

---

## 3. Anomaly Detection vs. Fraud Detection (Module 2)

### Q: What is the critical difference between anomaly detection and fraud detection?
**Answer**:
Statistical anomaly detection identifies mathematical outliers—observations that deviate significantly from standard distributions. In healthcare analytics:
- **IQR Tukey Fences**: Identifies extreme high-cost claims ($Q_3 + 1.5 \times \text{IQR}$).
- **Modified Z-Score**: Uses Median Absolute Deviation (MAD) for robust evaluation resilient to heavy-tailed skewness.
- **Isolation Forest**: Multi-dimensional tree isolation measuring path length to isolate sparse records.

**Crucial Distinction**: An anomaly is **never proof of fraud**. High-cost outliers frequently represent legitimate emergency procedures, intensive care admissions, or specialized oncology therapies. Calling an outlier "fraud" introduces severe clinical and legal bias. Instead, anomalies are labeled as **candidates for investigation or clinical audit**.

---

## 4. Rejection-Risk Modeling & Preventing Target Leakage (Module 3)

### Q: How did you ensure your claim rejection-risk model avoided target leakage?
**Answer**:
Target leakage occurs when features unavailable at prediction time are included during model training:
1. **Excluded Post-Adjudication Features**: `claim_status` (target), `denial_reason`, `paid_amount`, and adjudication settlement timestamps were strictly dropped.
2. **Pre-Adjudication Feature Set**: Predictions rely strictly on data available at submission: `billed_amount`, member demographics (`member_age`, `plan_type`, `member_risk_score`), and billing codes (`claim_type`, `diagnosis_code`, `procedure_code`).
3. **Scikit-Learn Pipeline & ColumnTransformer**: Imputation (median for numeric, constant for categorical) and scaling/one-hot encoding are fitted strictly on the training partition (`X_train`) to prevent test set data contamination.
4. **Class Imbalance**: Addressed using `class_weight='balanced'` within `RandomForestClassifier` rather than arbitrary target resampling.

---

## 5. NLP Transcript Extraction & Human Review Governance (Module 4)

### Q: How do you evaluate NLP performance in high-stakes healthcare operations?
**Answer**:
1. **Hybrid Extraction**: Regular expressions extract deterministic alphanumeric patterns (`CLM-XXXXXX`), while keyword mapping and TF-IDF + Logistic Regression classify member call intents (e.g., "Denial Reason", "Claim Status").
2. **Evaluation Metrics**: Evaluated against reference annotations across 6 field-level dimensions: Precision, Recall, F1 Score, Exact Match Accuracy, Coverage, and Missing Rate.
3. **Human-in-the-Loop Audit Trail**: Automated predictions are stored in an append-only SQLite audit table (`transcript_predictions`). When human reviewers approve, modify, or reject an extraction, changes are logged in `transcript_human_reviews` with reviewer ID and timestamp, never overwriting the original raw prediction.

---

## 6. Process Mining & Workflow Delays (Module 5)

### Q: How did you detect bottlenecks and rework loops in claims adjudication?
**Answer**:
Using the synthetic claims event log (`core_analytics/process_mining/`):
1. **Case Grouping & Ordering**: Events are ordered chronologically by `timestamp` per `claim_id`.
2. **Process Variants**: Extracted unique end-to-end activity sequences to categorize standard straight-through paths vs. non-standard paths.
3. **Rework Loops**: Detected duplicate occurrences of the same activity within a single case (e.g., repeated `Review_Clinical_Documentation`).
4. **Bottleneck Identification**: Built a directed activity transition matrix measuring duration between sequential steps. Transitions whose mean duration exceeded the operational baseline by $> 1.0\sigma$ were flagged as statistical bottlenecks.

---

## 7. BI-Ready Exports & Star Schema Modeling (Module 6)

### Q: How did you design the CSV exports for downstream BI tools like Power BI or Tableau?
**Answer**:
1. **Star Schema Architecture**: Centered on `claims_summary.csv` as the core fact table, connected via 1-to-Many relationships to `provider_analytics.csv` and 1-to-1 extension dimensions to `rejection_risk_scores.csv` and `process_case_summaries.csv`.
2. **Consistent Table Grain**: Documented in `manifest.json`, ensuring every table has an explicit primary key and foreign key mapping.
3. **Deduplication Alignment**: Deduplication rules are unified across the Python analytics engine, the SQLite database, and the CSV export layer (`keep='first'`), ensuring financial KPIs reconcile exactly.
4. **Download Security**: Flask endpoints enforce strict filename whitelisting (`EXPORT_WHITELIST`) to prevent directory traversal attacks.

---

## 8. Limitations & Real-World Adaptation

### Q: What would be required to transition this prototype to real health plan data?
**Answer**:
1. **Data Standards & EDI Integration**: Ingest standard X12 EDI 837 (claims submission) and EDI 835 (electronic remittance/payment) feeds rather than flat CSVs.
2. **ICD-10 / CPT Ontologies**: Replace synthetic billing codes with standard medical coding grouping systems (e.g., Clinical Classifications Software Refined - CCSR) to handle high-cardinality diagnoses.
3. **HIPAA & PHI Security**: Implement Role-Based Access Control (RBAC), database encryption at rest (AES-256), and de-identification pipelines conforming to HIPAA Safe Harbor guidelines.
4. **Scalable Compute**: Migrate from SQLite and in-memory Pandas to a distributed lakehouse architecture (e.g., BigQuery, Databricks) orchestrating incremental batch and streaming pipelines.
