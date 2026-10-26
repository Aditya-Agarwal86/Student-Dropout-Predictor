# Core Analytics Module 5: Operational Process Mining for Healthcare Claims Workflows

This document outlines the event log schema, process discovery metrics, rework detection heuristics, transition duration calculations, bottleneck formulas, limitations, and verification commands for **Module 5**.

---

## 1. Event Log Schema

Implemented in [`core_analytics/process_mining/synthetic_events.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/process_mining/synthetic_events.py).

| Field | Type | Description |
| :--- | :--- | :--- |
| `case_id` | TEXT | Unique claim case instance identifier (`CASE-10000+`) |
| `activity` | TEXT | Name of process task or state transition |
| `event_timestamp`| TEXT | ISO-formatted execution timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `resource` | TEXT | Team, department, or bot performing the task |
| `outcome_status` | TEXT | Output state upon task completion (`Received`, `Validated`, `Approved`, `Denied`, `Settled`, etc.) |
| `is_override` | BOOLEAN | Indicates whether task was a managerial or clinical override |
| `override_reason` | TEXT/NULL | Categorical reason code explaining the override if applicable |

### Common Fictional Process Variants:
1. **Straight-Through Adjudication (~40%)**:
   `Claim Submitted` $\rightarrow$ `Automated Intake` $\rightarrow$ `Initial Review` $\rightarrow$ `Payment Processed`
2. **Denial & Appeal Rework Loop (~25%)**:
   `Claim Submitted` $\rightarrow$ `Automated Intake` $\rightarrow$ `Initial Review (Denied)` $\rightarrow$ `Appeal Received` $\rightarrow$ `Initial Review (Approved)` $\rightarrow$ `Payment Processed`
3. **Manual Clinical Override Path (~15%)**:
   `Claim Submitted` $\rightarrow$ `Automated Intake` $\rightarrow$ `Manual Clinical Review` $\rightarrow$ `Manager Override` $\rightarrow$ `Payment Processed`
4. **Information Pended Bottleneck (~15%)**:
   `Claim Submitted` $\rightarrow$ `Automated Intake` $\rightarrow$ `Information Requested` $\rightarrow$ `Information Received` $\rightarrow$ `Initial Review` $\rightarrow$ `Payment Processed`

---

## 2. Process Mining & Metric Definitions

Implemented in [`core_analytics/process_mining/process_analyzer.py`](file:///c:/Users/Aditya%20Agarwal/OneDrive/Desktop/ML-AI/core_analytics/process_mining/process_analyzer.py).

### A. Chronological Sequencing & Data Quality Handling
- **Parsing**: Timestamps are parsed to datetime objects. Missing/null timestamps are excluded from duration metrics and surfaced in `dq_audit['missing_timestamps']`.
- **Deduplication**: Identical event logs (same `case_id`, `activity`, and `timestamp`) are deduplicated to prevent artificial zero-duration transitions.
- **Ordering**: Events are strictly sorted chronologically by `(case_id, parsed_timestamp)` prior to path reconstruction.
- **Single-Event Cases**: Handled safely with cycle time $= 0.0$ hours without crashing.

### B. Case Cycle Time & Rework Metrics
- **Cycle Time**:
  $$\text{Cycle Time (Hours)} = \frac{T_{\text{end}} - T_{\text{start}}}{3600 \text{ seconds}}$$
  $$\text{Cycle Time (Days)} = \frac{\text{Cycle Time (Hours)}}{24}$$
- **Rework Loop Identification**:
  A case is flagged as having rework (`has_rework = True`) if any activity appears more than once in its lifecycle.
  $$\text{Rework Events Count} = \sum (\text{Frequency}(A) - 1) \quad \text{for } \text{Frequency}(A) > 1$$
  *Policy*: Rework indicates administrative reprocessing (e.g. repeated reviews after appeal). It is not automatically treated as human error.

### C. Direct-Follows Activity Transitions
For consecutive events $(A_i, t_i) \rightarrow (A_{i+1}, t_{i+1})$:
- $\text{Transition Duration} = t_{i+1} - t_i$
- Summarizes: `transition_count`, `avg_duration_hours`, `median_duration_hours`, and `max_duration_hours`.

### D. Dynamic Bottleneck Identification
Identifies transitions whose mean duration exceeds statistical thresholds:
$$\text{Cutoff} = \mu_{\text{transition}} + (\text{threshold\_std} \times \sigma_{\text{transition}})$$
$$\text{Bottleneck Severity Score} = \frac{\mu_{\text{transition}} - \mu_{\text{overall}}}{\sigma_{\text{overall}}}$$
Observed bottlenecks emerge from data (e.g., long provider pending delays) rather than hardcoded rules.

### E. Manual Override Analysis
- Measures total overrides, unique cases impacted, and categorical breakdown across reasons (*Medical Necessity Exception*, *Peer-to-Peer Concordance*, *Emergency Service Carveout*, *Prior Auth Extenuating Circumstance*).

---

## 3. Analytical Outputs & CSV Export

The engine provides direct DataFrame generation and file export:
```python
engine = ProcessMiningEngine(event_log)
files = engine.export_all_metrics_to_csv(export_dir="./exports")
```
- `process_case_summaries.csv`: Case-level duration, event counts, rework flags, and variant paths.
- `process_variants.csv`: Discovered workflow paths, case counts, percentages, and average durations.
- `process_transitions.csv`: Direct-follows transition counts and duration statistics.
- `process_bottlenecks.csv`: Statistical bottleneck candidates and severity scores.

---

## 4. Assumptions and Limitations

1. **Discrete Event Timestamps**:
   - Assumes event timestamps mark completion or logging of an activity.
2. **Correlation vs. Causation**:
   - Associated outcomes per variant represent observed distributions. Higher denial rates in appeal paths are naturally inherent to the appeal process and do not imply the workflow caused the initial denial.
3. **Lightweight Python & Pandas Architecture**:
   - Implements native direct-follows graph (DFG) logic without requiring external JVM tools, heavy C dependencies, or commercial software (e.g., Celonis).

---

## 5. Execution and Testing Commands

### Run Module 5 Tests:
```powershell
python -m pytest tests/test_analytics_module5.py -v
```

### Run Full Pytest Suite (67 Tests across Modules 1–5 + Web App):
```powershell
python -m pytest
```

### Run Baseline Student Dropout ML Pipeline:
```powershell
python test_ml.py
```

### Run Setup Verification:
```powershell
python setup.py
```
