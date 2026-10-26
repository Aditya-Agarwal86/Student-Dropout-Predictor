# 🎓 Student Dropout Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-orange.svg)](https://scikit-learn.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38bdf8.svg)](https://tailwindcss.com/)
[![CI](https://github.com/actions/workflows/ci.yml/badge.svg)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, full-stack machine learning web application designed to predict student dropout risk. Powered by a Support Vector Machine (SVM) classifier trained on demographic, socioeconomic, and academic performance indicators, the system helps educational institutions identify at-risk students early and take targeted retention actions.

---

## ✨ Features

- **High-Accuracy ML Model**: Trained Support Vector Machine (SVM) with RBF kernel and class balancing, achieving >83% accuracy and balanced recall.
- **Instant Model Caching**: Uses `joblib` model caching to eliminate repetitive retraining overhead upon server cold starts.
- **Modern Bilingual UI**: Clean, mobile-friendly interface built with **Tailwind CSS**, Lucide icons, and bilingual support (**English & Hindi**).
- **Comprehensive Risk Classification**:
  - 🔴 **High Risk**: Dropout probability $\ge 70\%$
  - 🟡 **Medium Risk**: Dropout probability $40\% - 69\%$
  - 🟢 **Low Risk**: Dropout probability $< 40\%$
- **Deep-Dive Student Profiles**: Individual diagnostic profiles with granular risk factors, categorical course translations, and contextual metrics.
- **Dynamic Data Visualizations**: Real-time interactive breakdown with Chart.js and Plotly.
- **Color-Coded Excel Exports**: In-memory Excel generation with conditional risk color formatting (`openpyxl`) and optional SMTP automated email delivery.
- **Production-Ready**: Configured with `python-dotenv`, Gunicorn WSGI support, Docker containerization, and GitHub Actions CI.

---

## 📊 Dataset Schema

The system accepts CSV and Excel (`.xlsx`, `.xls`) files containing the following **23 required student indicators** (aligned with the Higher Education Student Performance dataset):

| Category | Columns | Description |
| :--- | :--- | :--- |
| **Demographics** | `Gender`, `Age at enrollment`, `Marital status`, `International`, `Displaced` | Age, gender, citizenship, displacement, and marital status |
| **Academic History** | `Previous qualification`, `Previous qualification (grade)`, `Course`, `Daytime/evening attendance`, `Educational special needs` | Secondary education grades, chosen program, attendance mode |
| **Socioeconomic** | `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` | Parental educational attainment and occupational tiers |
| **Financial** | `Tuition fees up to date`, `Debtor`, `Scholarship holder` | Fee status, debt status, and active scholarship holding |
| **First Semester Units**| `Curricular units 1st sem (credited)`, `(enrolled)`, `(evaluations)`, `(approved)`, `(grade)`, `(without evaluations)` | Credits enrolled, evaluations attempted, units passed, and average grades |

> [!NOTE]
> Inference datasets for prediction **must not** contain the `Target` column. A pre-formatted sample dataset (`sample_student_data.csv`) is provided and can be downloaded directly from the web interface.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.8+ (tested on Python 3.10 - 3.13)
- `pip` package manager

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction
```

### 3. Set Up Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration (Optional)
Copy the sample environment file to configure custom ports or email reporting:
```bash
cp .env.example .env
```
*(On Windows: `copy .env.example .env`)*

### 6. Run Setup Verification & Start Application
Verify system compatibility:
```bash
python setup.py
```

Launch the application:
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🐳 Running with Docker

Run the entire application in a container with zero local dependencies:

```bash
# Build Docker image
docker build -t student-dropout-prediction .

# Run container
docker run -p 5000:5000 --env-file .env.example student-dropout-prediction
```
Visit `http://localhost:5000` in your browser.

---

## 🧪 Testing

The repository includes a complete automated test suite verifying data pipelines, preprocessing, model inference, and web endpoints.

```bash
# Run pytest test suite
pytest tests/ -v

# Run standalone ML model benchmark
python test_ml.py
```

---

## 📁 Project Structure

```text
student-dropout-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI matrix workflow
├── static/
│   ├── css/                   # Custom styling assets
│   └── js/                    # Client-side scripts
├── templates/
│   ├── base.html              # Base layout with bilingual & dark mode toggles
│   ├── index.html             # Upload landing page & sample data download
│   ├── dashboard.html         # Interactive predictions dashboard
│   └── student_detail.html    # Detailed individual student risk diagnostic
├── tests/
│   └── test_app.py            # Automated Pytest suite
├── .dockerignore              # Docker build exclusions
├── .env.example               # Environment variables template
├── .gitignore                 # Comprehensive git ignore rules
├── app.py                     # Flask application & prediction API
├── chart_script_1.py          # Script for performance charts
├── CONTRIBUTING.md            # Open-source contribution guidelines
├── Data.csv                   # Base training dataset (UCI Student Performance)
├── Dockerfile                 # Production Docker configuration
├── LICENSE                    # MIT License
├── pytest.ini                 # Pytest configuration
├── README.md                  # Project documentation
├── requirements.txt           # Python package dependencies
├── run.py                     # Convenience launcher with auto-browser opening
├── sample_student_data.csv    # Ready-to-upload inference sample dataset
├── setup.py                   # Automated setup and environment verifier
└── test_ml.py                 # Standalone ML pipeline and metrics script
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Upload landing page |
| `POST` | `/upload` | Processes uploaded CSV or Excel dataset for risk inference |
| `GET` | `/dashboard` | Interactive results dashboard |
| `GET` | `/api/dashboard-data` | JSON REST API returning aggregated risk summary and student records |
| `GET` | `/student/<int:id>` | Granular diagnostic breakdown for a specific student record |
| `GET` | `/download_excel` | Generates color-coded Excel report with optional email delivery |
| `GET` | `/sample-data` | Downloads valid pre-formatted sample CSV for testing |

---

## 🤝 Contributing

Contributions, feature requests, and issue reports are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
