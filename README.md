# Student Dropout Prediction & Healthcare Analytics System

A dual-domain predictive analytics and business intelligence platform built on Flask, Scikit-Learn, and Python.

This project combines:
1. **Higher Education Retention**: An SVM classification system predicting student dropout risk from early-semester academic and socioeconomic indicators.
2. **Healthcare Operations & AI**: An operational analytics engine, claims rejection-risk model, process mining analyzer, NLP call audit pipeline, and BI-ready export framework designed for health plans.

---

## Architecture Overview

```mermaid
graph TD
    subgraph Web_Application ["Flask Web Application (app.py)"]
        IndexRoute["/ (Upload & Predict)"]
        DashRoute["/dashboard (Retention Dashboard)"]
        HealthRoute["/healthcare (Healthcare Dashboard)"]
        ExportRoute["/healthcare/export/* (BI Export API)"]
    end

    subgraph Student_ML ["Student Dropout Retention Engine"]
        SVMModel["SVM (RBF Kernel, Class-Balanced)"]
        Cache["model_cache.joblib"]
        StudentData["Data.csv (4,424 Records)"]
        SVMModel --> Cache
    end

    subgraph Healthcare_Core ["Healthcare Analytics & AI Core (core_analytics/)"]
        Service["HealthcareAnalyticsService (Singleton Coordinator)"]
        Mod1["Module 1: Synthetic Data, SQL Profiler & DQ Engine"]
        Mod2["Module 2: Anomaly Detection (IQR, Mod-Z, IsolationForest)"]
        Mod3["Module 3: Operational KPIs & Claims Risk Classification"]
        Mod4["Module 4: NLP Call Transcripts & Human Review Audit"]
        Mod5["Module 5: Claims Process Mining & Bottlenecks"]
        Mod6["Module 6: BI Exports & Manifest Generator"]

        Service --> Mod1
        Service --> Mod2
        Service --> Mod3
        Service --> Mod4
        Service --> Mod5
        Service --> Mod6
    end

    subgraph Downstream_BI ["Downstream BI Platforms"]
        PowerBI["Power BI Desktop"]
        Tableau["Tableau Desktop"]
    end

    IndexRoute --> Student_ML
    DashRoute --> Student_ML
    HealthRoute --> Service
    ExportRoute --> Mod6
    Mod6 -->|11 Clean CSVs + manifest.json| Downstream_BI
```

---

## Key Modules Summary

| Module | Core Responsibility | Key Technologies | Primary Outputs |
| :--- | :--- | :--- | :--- |
| **Student Retention** | Student dropout classification & diagnostic risk remarked reports | Scikit-Learn (SVM), Joblib, Pandas | 83.6% accuracy, 73.9% recall, Excel reports |
| **Module 1: Healthcare Data & DQ** | Deterministic synthetic healthcare dataset generator & SQL profiling | SQLite, Pandas, SQL profiling | `claims`, `members`, `providers` tables, 9 DQ checks |
| **Module 2: Anomaly Detection** | Statistical and machine learning anomaly detection on financial claims | IQR Tukey fences, Modified Z-score (MAD), Isolation Forest | Anomaly scores, record flags, feature-level attribution |
| **Module 3: Claims & Risk AI** | Operational KPIs, PMPM expenditure, and pre-adjudication rejection risk | Scikit-Learn Pipeline, ColumnTransformer, RandomForest | Billed/paid spend, denial rate %, PMPM, rejection risk tiers |
| **Module 4: NLP & Human Review** | Extraction of structured entities from call transcripts & audit trail | Regex, TF-IDF + LogisticRegression, SQLite Audit Table | Intent classification, claim ID extraction, reviewer audit log |
| **Module 5: Process Mining** | Adjudication workflow event mining, rework loops, bottlenecks | Markovian transitions, Pandas, NumPy | Case cycle times, variant taxonomy, delay bottlenecks |
| **Module 6: Dashboard & BI Exports** | Healthcare dashboard UI, REST summary API, BI CSV export bundle | Flask, Tailwind CSS, Chart.js, Zipfile | `/healthcare`, 11 CSV exports, `manifest.json`, ZIP bundle |
| **Module 7: QA & Validation** | Full-suite regression testing, data leakage audit, reconciliation | Pytest (78 tests), Git audit, validation guides | Complete test verification, interview preparation notes |

---

## Quick Start & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Aditya-Agarwal86/Student-Dropout-Predictor.git
cd Student-Dropout-Predictor

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python setup.py
```

### 3. Run Automated Tests
```bash
# Run all 78 unit and integration tests
python -m pytest

# Run ML pipeline benchmark
python test_ml.py
```

### 4. Start Flask Application
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## Walkthrough: Demonstrating the Application

1. **Student Dropout Prediction Portal**:
   - Navigate to `http://localhost:5000/`.
   - Download the pre-formatted dataset via **Download Sample CSV**.
   - Upload the file to inspect the interactive retention dashboard, risk bands, and per-student diagnostic remarks.
2. **Healthcare Analytics & AI Dashboard**:
   - Navigate to `http://localhost:5000/healthcare`.
   - Review operational claims KPIs, denial rate, PMPM spending, and data-quality health.
   - Inspect the Machine Learning rejection-risk distribution donut chart (Low / Medium / High).
   - Review process mining variants and statistical workflow bottleneck transitions.
3. **Power BI / Tableau BI Export Downloads**:
   - Click **"Export BI Bundle (.zip)"** or download individual CSV tables.
   - Refer to `core_analytics/BI_DASHBOARD_GUIDE.md` for star-schema relationship diagrams, DAX formulas, and Tableau LOD calculations.

---

## Responsible AI & Demonstration Limitations

1. **Synthetic Healthcare Data**: All member IDs, provider profiles, claims, event logs, and transcripts are generated with fixed random seeds for demonstration. No protected health information (PHI) is used.
2. **Prototype Scope**: Statistical outlier flags (IQR, Isolation Forest) indicate unusual records requiring review—they do not constitute proof of fraud or abuse.
3. **Target Leakage Prevention**: In the claims rejection model, post-adjudication features (`claim_status`, `denial_reason`, `paid_amount`) are excluded from pre-adjudication risk scoring.
