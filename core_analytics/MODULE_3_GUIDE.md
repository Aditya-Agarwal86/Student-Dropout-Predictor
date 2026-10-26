# Core Analytics Module 3: Healthcare Claims, Enrollment & Provider Analytics with Claim Rejection-Risk Classification

This guide describes the operational KPIs, mathematical formulas, leakage-free feature selection, model architecture, integration with anomaly detection, limitations, and verification commands for **Module 3**.

---

## 1. Operational Analytics & KPI Definitions

Implemented in [`core_analytics/healthcare/operational_analytics.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/healthcare/operational_analytics.py).

### A. Claims Volume & Financial KPIs
- **`evaluated_claim_count`**: Total claims after policy handling.
- **`total_billed`**: Sum of positive billed charges ($\sum \text{billed\_amount} > 0$).
- **`total_paid`**: Sum of positive adjudicated reimbursements ($\sum \text{paid\_amount} > 0$).
- **`denial_rate_pct`**:
  $$\text{Denial Rate} = \left( \frac{\text{Count of Denied Claims}}{\text{Evaluated Claims}} \right) \times 100$$
- **`avg_paid_per_paid_claim`**: Average reimbursement among paid claims:
  $$\frac{\text{Total Paid Spend}}{\text{Count of Paid Claims}}$$

### B. Transparent Data-Quality & Defect Policy
- **Duplicate Claims**: Claims with identical `claim_id` are identified. For financial KPI calculation, duplicate records are deduplicated (`keep='first'`) so charges and payments are not double-counted. The number of excluded duplicates is explicitly surfaced (`duplicate_claims_excluded`).
- **Orphan Provider References**: Claims referencing unknown provider keys (`PRV-9999_ORPHAN`) are preserved in provider analytics and labeled as `[Unmatched Provider / Orphan]` with an explicit `is_orphan: True` flag.

### C. Enrollment & Per-Member-Per-Month (PMPM) Expenditure
- **Member-Months Denominator**:
  For an evaluation period $[T_{\text{start}}, T_{\text{end}}]$, a member is credited with $1$ active member-month for each calendar month $M$ where:
  $$\text{enrollment\_start} \le \text{month\_end}(M) \quad \text{AND} \quad (\text{enrollment\_end} \text{ is NULL or } \ge \text{month\_start}(M))$$
- **PMPM Formula**:
  $$\text{PMPM} = \frac{\text{Total Paid Spend in Period}}{\text{Total Active Member-Months in Period}}$$

### D. Anomaly Detection Integration
- Integrates Module 2's `IQROutlierDetector` to identify unusually high-cost claims (`billed_amount` exceeding the upper Tukey fence $Q_3 + 1.5 \times IQR$).
- **Ethical & Operational Principle**: High-cost outliers represent statistical variations (e.g., intensive hospitalizations) and are **never** treated as fraud or automatic rejection reasons.

---

## 2. Claim Rejection-Risk Classification Model

Implemented in [`core_analytics/healthcare/claims_models.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/healthcare/claims_models.py).

### A. Target Definition
Binary adjudication outcome:
$$y = \begin{cases} 1 & \text{if } \text{claim\_status} = \text{'Denied'} \\ 0 & \text{if } \text{claim\_status} = \text{'Paid'} \text{ or other} \end{cases}$$

### B. Strict Target Leakage Prevention
To ensure the model is valid for pre-adjudication decision support, post-adjudication fields are **strictly excluded**:
- **Excluded Leakage Features**: `claim_status`, `denial_reason`, `paid_amount`, adjudication timestamps.
- **Pre-Adjudication Predictive Features**:
  - *Numeric*: `billed_amount`, `member_risk_score`, `member_age`.
  - *Categorical*: `claim_type`, `diagnosis_code`, `procedure_code`, `plan_type`.

### C. Pipeline Architecture & Imbalance Handling
- **Pre-Processing (`ColumnTransformer`)**:
  - Numeric Pipeline: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical Pipeline: `SimpleImputer(strategy='constant', fill_value='UNKNOWN')` followed by `OneHotEncoder(handle_unknown='ignore')`.
  - Fitted strictly on training splits; reusable for scoring new or unseen data without leakage.
- **Classifier**: `RandomForestClassifier(class_weight='balanced', max_depth=6, random_state=42)`.
  - Balanced class weights compensate for the natural denial rarity ($\sim 15\%$ denial prevalence in synthetic claims).

### D. Rejection-Risk Probabilities & Calibrated Risk Bands
The model generates continuous rejection probability $P(\text{Rejection})$ and maps it into calibrated risk tiers:
- **Low Risk**: $P(\text{Rejection}) < 0.25$
- **Medium Risk**: $0.25 \le P(\text{Rejection}) < 0.60$
- **High Risk**: $P(\text{Rejection}) \ge 0.60$
- **Separation of Concerns**: The prediction output maintains `rejection_probability` and `predicted_risk_band` distinct from `actual_observed_status`.

### E. Model Persistence Isolation
- Supported via `save_model()` and `load_model()` into standalone model files (`claims_rejection_model.joblib`).
- Guaranteed **zero collision** with the student dropout model cache (`model_cache.joblib`).

---

## 3. Assumptions and Limitations

1. **Synthetic Data Realism**:
   - The synthetic dataset is generated deterministically ($N=1,000$ claims). While suitable for testing, verifying pipeline ergonomics, and demonstrating feature preprocessing, production clinical classifiers should be trained on real claims clearinghouse datasets with hundreds of thousands of longitudinal claims.
2. **Provider Contracting Nuances**:
   - The model utilizes member risk scores, procedure codes, diagnosis codes, and billed charges. It does not model complex hospital fee schedule carve-outs or DRG grouper weights.

---

## 4. Verification and Testing Commands

### Run Module 3 Tests:
```powershell
python -m pytest tests/test_analytics_module3.py -v
```

### Run Full Test Suite (Modules 1, 2, 3 + Student Web App):
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
