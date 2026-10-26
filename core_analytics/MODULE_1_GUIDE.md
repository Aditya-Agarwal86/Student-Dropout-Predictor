# Core Analytics Module 1: Healthcare Data Foundation, SQL Profiling, and Data Quality

This document describes the synthetic datasets, database schemas, profiling routines, declarative data-quality rules, and execution commands introduced in **Module 1**.

---

## 1. Synthetic Healthcare Datasets

All records are **strictly fictional**, generated deterministically using NumPy's generator (`np.random.default_rng(seed=42)`). They do not represent real patients or any proprietary system.

### A. Members (Enrollment)
- **Table**: `members`
- **Volume**: 200 records
- **Primary Key**: `member_id`
- **Controlled Anomalies Included**:
  - `gender`: Contains domain-invalid string (`INVALID_GENDER` on row 10) to test categorical validation.
  - `risk_score`: Contains negative score (`-1.5` on row 25) to test numeric lower bound checks.

### B. Providers
- **Table**: `providers`
- **Volume**: 50 records
- **Primary Key**: `provider_id`
- **Controlled Anomalies Included**:
  - `npi`: Row 3 contains a null National Provider Identifier to test missing mandatory key checks.

### C. Claims
- **Table**: `claims`
- **Volume**: 1,000 records
- **Foreign Keys**: `member_id` $\rightarrow$ `members.member_id`, `provider_id` $\rightarrow$ `providers.provider_id`
- **Controlled Anomalies Included**:
  - Duplicate business key: Row 5 duplicates Row 4's `claim_id`.
  - Missing mandatory fields: Row 12 has `member_id = NULL`; Row 15 has `diagnosis_code = NULL`.
  - Negative charges: Row 20 has `billed_amount = -150.00`.
  - Financial discrepancy: Row 30 has `paid_amount = 350.00` exceeding `billed_amount = 100.00`.
  - Inverted chronology: Row 40 has `service_end_date` before `service_start_date`.
  - Orphan references: Rows 50 & 51 reference non-existent providers `PRV-9999_ORPHAN` and `PRV-9998_ORPHAN`.
  - Unexpected status: Row 60 contains `claim_status = 'UNKNOWN_STATUS'`.

---

## 2. Data Dictionary

### Table: `members`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `member_id` | TEXT | PRIMARY KEY | Unique member identifier (`MEM-10000+`) |
| `gender` | TEXT | | Gender code (`M`, `F`, `U`) |
| `date_of_birth` | TEXT | | Member birth date (`YYYY-MM-DD`) |
| `plan_type` | TEXT | | Health plan type (`HMO`, `PPO`, `EPO`, `POS`) |
| `enrollment_start` | TEXT | NOT NULL | Benefit coverage start date |
| `enrollment_end` | TEXT | NULLABLE | Benefit termination date (NULL if active) |
| `state` | TEXT | | Residence state code (e.g. `CA`, `TX`, `NY`) |
| `risk_score` | REAL | | Prospective clinical risk score |

### Table: `providers`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `provider_id` | TEXT | PRIMARY KEY | Unique provider identifier (`PRV-1000+`) |
| `npi` | TEXT | | 10-digit National Provider Identifier |
| `provider_name` | TEXT | NOT NULL | Rendering provider or clinic name |
| `specialty` | TEXT | | Clinical specialty |
| `state` | TEXT | | Practice location state |
| `status` | TEXT | | Network status (`Active`, `Inactive`) |

### Table: `claims`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `claim_id` | TEXT | | Adjudicated claim identifier (`CLM-100000+`) |
| `member_id` | TEXT | FK $\rightarrow$ `members` | Member who received services |
| `provider_id` | TEXT | FK $\rightarrow$ `providers` | Servicing provider |
| `claim_type` | TEXT | | Service category (`Professional`, `Institutional`, `Pharmacy`) |
| `service_start_date`| TEXT | | Date service commenced |
| `service_end_date` | TEXT | | Date service ended |
| `diagnosis_code` | TEXT | | ICD-10 diagnosis code |
| `procedure_code` | TEXT | | CPT/HCPCS procedure code |
| `billed_amount` | REAL | | Total submitted provider charges |
| `paid_amount` | REAL | | Total adjudicated plan reimbursement |
| `claim_status` | TEXT | | Adjudication status (`Paid`, `Denied`, `In Process`, `Pending`) |
| `denial_reason` | TEXT | NULLABLE | Categorical denial explanation if status is `Denied` |

---

## 3. SQL Profiling Capabilities

Implemented in [`core_analytics/profiling/sql_profiler.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/profiling/sql_profiler.py):
1. **`profile_table(table_name)`**:
   - Total row counts
   - Column-level null counts and null percentages
   - Cardinality (distinct counts)
   - Numerical minimum, maximum, and mean
2. **`get_column_frequency(table_name, column_name, top_n=10)`**:
   - Group-by frequency distributions and percentages for categorical columns
3. **`get_duplicate_counts(table_name, key_column)`**:
   - Identifies repeated primary/business keys and counts
4. **`calculate_claims_kpis()`**:
   - Total volume and spend (billed vs. paid)
   - Adjudication denial rate percentage
   - Top 5 denial reason frequencies
   - Top cost-concentrated providers

---

## 4. Reusable Data Quality (DQ) Engine

Implemented in [`core_analytics/profiling/data_quality.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/profiling/data_quality.py):
- **Structured Output**: Every check returns a `DQResult` containing:
  - `check_name`, `target_table`, `target_column`
  - `severity`: `ERROR` (blocks pipeline / critical failure) vs `WARNING` (advisory / informational)
  - `passed`: Boolean pass/fail indicator
  - `total_evaluated`, `failure_count`, `failure_rate_pct`
  - `affected_identifiers`: Specific primary/claim IDs exhibiting the violation
  - `suggested_remediation`: Prescriptive action to resolve the defect
- **Guiding Principle**: Never silently modifies, coerces, or deletes records. Reports all defective identifiers for governance audits.

---

## 5. Execution and Testing Commands

### Run Full Test Suite (Baseline Student App + New Analytics Module):
```powershell
python -m pytest
```

### Run Module 1 Analytics Tests Only:
```powershell
python -m pytest tests/test_analytics_module1.py -v
```

### Run Existing Student Dropout Model Baseline:
```powershell
python test_ml.py
```

### Run Setup Verification:
```powershell
python setup.py
```
