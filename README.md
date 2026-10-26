# Student Dropout Prediction System

Universities often discover a student has dropped out weeks after they stop attending lectures and submit no work. By that point, academic advisors have missed the window for intervention.

This project uses early-semester administrative, academic, and financial records to predict student dropout risk before students withdraw. Trained on 4,424 student records from the Higher Education dataset, the Support Vector Machine (SVM) pipeline reaches 83.6% test accuracy and 73.9% recall. Predictions are exposed through an interactive web interface with per-student risk breakdowns, diagnostic remarks, and color-coded Excel reports for retention teams.

## Model and Pipeline

The prediction pipeline uses an SVM classifier with a Radial Basis Function (RBF) kernel and balanced class weights to account for class imbalance between continuing students and dropouts.

- **Data source**: 4,424 records from the UCI Higher Education Student Performance dataset.
- **Preprocessing**: Categorical features are encoded using `LabelEncoder`. Numeric and categorical inputs are scaled with `StandardScaler`.
- **Model caching**: Trained weights, encoders, and scaler objects are serialized to `model_cache.joblib`. Warm starts load in under 50 milliseconds without refitting.
- **Evaluation** (885-sample test split, 20% holdout):
  - Accuracy: 83.6%
  - Precision: 74.7%
  - Recall: 73.9%
  - Confusion Matrix: 530 true negatives, 210 true positives, 71 false positives, 74 false negatives.

## Risk Scoring

The model outputs continuous dropout probabilities converted into three operational tiers:

- **High Risk (>= 70%)**: Flagged for immediate advisor contact.
- **Medium Risk (40% to 69%)**: Flagged for academic monitoring and tutoring support.
- **Low Risk (< 40%)**: On track for continuation.

Each prediction includes automated diagnostic remarks based on key risk indicators, including tuition fee arrears, debtor status, low first-semester units approved, previous qualification grade below threshold, and mature student enrollment age.

## Dataset Features

The application accepts bulk student files (.csv and .xlsx) containing 23 predictor columns. Datasets uploaded for prediction must omit the `Target` column.

| Category | Columns | Notes |
| :--- | :--- | :--- |
| Demographics | `Gender`, `Age at enrollment`, `Marital status`, `International`, `Displaced` | Encoded citizenship, marital, and relocation indicators |
| Academic History | `Course`, `Previous qualification`, `Previous qualification (grade)`, `Daytime/evening attendance`, `Educational special needs` | Secondary education grades and degree track |
| Socioeconomic | `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` | Parental education and employment tiers |
| Financial | `Tuition fees up to date`, `Debtor`, `Scholarship holder` | Fee status, debt status, and active financial aid |
| Semester 1 Performance | `Curricular units 1st sem (credited)`, `(enrolled)`, `(evaluations)`, `(approved)`, `(grade)`, `(without evaluations)` | Credits enrolled, completed, and grade point average |

A pre-formatted sample file matching this schema is available at `sample_student_data.csv` or through the **Download Sample** link in the web interface.

## Application Architecture

The system is organized into a Flask web application with a bilingual (English and Hindi) frontend built with Tailwind CSS.

- `app.py`: Core application routes, ML pipeline execution, Excel report generation, and API endpoints.
- `templates/`: Server-rendered Jinja templates including `index.html` (upload portal), `dashboard.html` (interactive risk charts and student tables), and `student_detail.html` (individual diagnostic cards).
- `setup.py`: Environment verification script that validates Python versions, dependencies, and file paths.
- `test_ml.py`: Standalone ML script that runs the data preprocessing and model evaluation without launching the web server.
- `tests/test_app.py`: Automated integration test suite covering route handling, file uploads, invalid file rejections, and Excel generation.

## Local Setup

### 1. Clone the repository and enter the directory
```bash
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Create and activate a virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify the setup
Run the environment verification script to check package installations and directory structures:
```bash
python setup.py
```

### 5. Start the server
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

## Running Tests

Run the full pytest suite:
```bash
pytest tests/ -v
```

Run the standalone ML pipeline benchmark:
```bash
python test_ml.py
```

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Upload portal for student data files |
| `POST` | `/upload` | Processes uploaded CSV or Excel file and generates risk scores |
| `GET` | `/dashboard` | Interactive risk distribution and sortable student tables |
| `GET` | `/api/dashboard-data` | JSON endpoint returning aggregated counts and student records |
| `GET` | `/student/<int:id>` | Detailed demographic, academic, and financial breakdown for a student |
| `GET` | `/download_excel` | Exports color-coded Excel report with risk highlighting |
| `GET` | `/sample-data` | Downloads valid sample CSV for testing |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
