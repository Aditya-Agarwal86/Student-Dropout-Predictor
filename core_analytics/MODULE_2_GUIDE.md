# Core Analytics Module 2: Reusable Anomaly & Outlier Detection Engine

This document outlines the algorithms, assumptions, mathematical thresholds, limitations, ethical guidelines, and testing commands for the outlier detection engine in `core_analytics/anomaly/`.

---

## 1. Implemented Algorithms & Mathematical Principles

### A. Tukey's Interquartile Range (IQR) Fences (`IQROutlierDetector`)
- **Formula**:
  - $IQR = Q_3 - Q_1$
  - $\text{Lower Bound} = Q_1 - (k \times IQR)$
  - $\text{Upper Bound} = Q_3 + (k \times IQR)$
- **Assumptions**:
  - Robust non-parametric metric; does not assume a normal distribution.
- **Configurable Multiplier ($k$)**:
  - $k = 1.5$: Standard Tukey fences (default, identifies mild outliers).
  - $k = 3.0$: Extreme outlier fences.
- **Scoring Output**:
  - `anomaly_score`: Distance beyond the violated fence normalized by $IQR$: $\frac{|x - \text{bound}|}{IQR}$.

### B. Modified Z-Score (`ModifiedZScoreOutlierDetector`)
- **Formula**:
  - Based on Iglewicz and Hoaglin (1993):
    $$M_i = \frac{0.6745 \times |x_i - \text{median}|}{\text{MAD}}$$
  - where $\text{MAD} = \text{median}(|x_i - \text{median}|)$.
- **Assumptions**:
  - Replaces traditional mean and standard deviation with median and median absolute deviation.
  - Mitigates the "masking effect" where extreme outliers artificially inflate the mean and standard deviation.
- **Configurable Threshold**:
  - Default: $M_i > 3.5$ flagged as statistical outlier.
- **Constant Column Fallback**:
  - If $\text{MAD} = 0$ (e.g. constant columns or heavily tied medians), falls back to mean absolute deviation or $\epsilon = 10^{-6}$ to prevent division by zero.

### C. Isolation Forest (`IsolationForestOutlierDetector`)
- **Algorithm**:
  - Scikit-learn's tree-based `IsolationForest`. Randomly partitions feature space; anomalies require fewer splits to isolate, resulting in shorter path lengths in isolation trees.
- **Assumptions**:
  - Multivariate dependencies; handles high-dimensional interactions between features.
- **Configurable Hyperparameters**:
  - `contamination`: Expected anomaly proportion ($0.0 < \text{contamination} \le 0.5$).
  - `random_state`: Fixed random seed for deterministic repeatability.
  - `n_estimators`: Number of isolation trees (default: 100).
- **Preprocessing**:
  - Explicit numeric coercion with median imputation fitted strictly on reference data and stored for scoring. Minimum sample size guard ($N \ge 10$).

---

## 2. Ethical Classification & Governance Standard

> [!IMPORTANT]
> **Anomalies are statistical deviations, NEVER confirmed fraud.**
> Flagging a claim, member, or provider as anomalous means the record exhibits high statistical variance from expected patterns. It warrants operational or clinical review, but must **never** be labeled as fraud without human investigation and legal adjudication.

All remediation notes and record explanations explicitly describe statistical variance without accusations of fraud or misconduct.

---

## 3. Assumptions and Limitations

1. **Univariate vs. Multivariate**:
   - `IQROutlierDetector` and `ModifiedZScoreOutlierDetector` evaluate features individually. They do not model correlations between features (e.g., high billed amount for a complex surgical procedure may be normal, but high for an office visit).
   - `IsolationForestOutlierDetector` evaluates multivariate interactions across all provided numeric features.
2. **Missing Values**:
   - IQR and Modified Z-Score evaluate available values without fabricating missing data.
   - Isolation Forest applies reference median imputation to ensure complete numeric arrays for tree traversal.
3. **Reference Data Splitting**:
   - Parameters ($\text{median}$, $\text{MAD}$, $Q_1$, $Q_3$, isolation trees) are fitted strictly on baseline/training data, and can be reused to score new incoming batches without data leakage.

---

## 4. Execution and Testing Commands

### Run Module 2 Tests:
```powershell
python -m pytest tests/test_analytics_module2.py -v
```

### Run Full pytest Suite (Baseline + Module 1 + Module 2):
```powershell
python -m pytest
```

### Run Baseline Model Verification:
```powershell
python test_ml.py
```

### Run Environment Setup Verification:
```powershell
python setup.py
```
