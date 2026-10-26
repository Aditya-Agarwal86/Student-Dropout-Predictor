import os
import io
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from openpyxl.styles import PatternFill
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import joblib

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'student-dropout-prediction-svm-2025')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_RECIPIENT'] = os.environ.get('MAIL_RECIPIENT', '')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model, scaler, label_encoders, model_columns, processed_data = None, None, {}, None, None
MODEL_CACHE_FILE = 'model_cache.joblib'

REQUIRED_COLUMNS_PREDICTION = [
    "Marital status", "Course", "Daytime/evening attendance", "Previous qualification", 
    "Previous qualification (grade)", "Mother's qualification", "Father's qualification", 
    "Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", 
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", 
    "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", 
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)", 
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"
]
REQUIRED_COLUMNS_TRAINING = REQUIRED_COLUMNS_PREDICTION + ["Target"]

COURSE_MAP = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design", 
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design", 
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture", 
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management", 
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)"
}
MARITAL_STATUS_MAP = {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto Union', 6: 'Legally Separated'}
QUALIFICATION_MAP = {
    1: "Secondary Education", 2: "Higher Education - Bachelor's", 3: "Higher Education - Degree", 
    4: "Higher Education - Master's", 5: "Higher Education - Doctorate", 6: "Frequency of Higher Education", 
    9: "12th Year of Schooling - Not Completed", 10: "11th Year of Schooling - Not Completed", 
    11: "7th Year (Old)", 12: "Other - 11th Year of Schooling", 14: "10th Year of Schooling", 
    19: "Basic Education 3rd Cycle", 38: "Basic Education 2nd Cycle", 39: "Technological specialization course", 
    40: "Higher education - degree (1st cycle)", 42: "Professional higher technical course", 
    43: "Higher education - master (2nd cycle)", 34: "Unknown"
}
OCCUPATION_MAP = {
    0: "Student", 1: "Director/Executive Manager", 2: "Specialist", 3: "Technician/Intermediate Professional", 
    4: "Administrative Staff", 5: "Personal Services/Sales", 6: "Skilled Agriculture/Fishery Worker", 
    7: "Skilled Industry/Construction Worker", 8: "Machine Operator/Assembler", 9: "Unskilled Worker", 
    10: "Armed Forces", 90: "Other", 99: "N/A", 122: "Health professionals", 123: "Teachers", 
    125: "ICT Specialists", 131: "Intermediate science/engineering technicians", 
    132: "Intermediate health technicians", 134: "Intermediate technicians (legal, social, sports)", 
    141: "Office workers", 143: "Data/accounting operators", 144: "Other administrative staff", 
    151: "Personal service workers", 152: "Sellers", 153: "Personal care workers", 
    171: "Skilled construction workers", 173: "Skilled printing/instrument workers", 
    175: "Skilled food processing/woodworking workers", 191: "Cleaning workers", 
    192: "Unskilled agriculture workers", 193: "Unskilled industry/construction workers", 
    194: "Meal preparation assistants"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv', 'xlsx']

def preprocess_data(df, is_training=True):
    global label_encoders, scaler, model_columns
    processed_df = df.copy()
    if is_training:
        processed_df['Target'] = processed_df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
        label_encoders = {}
        for col in sorted(processed_df.select_dtypes(include=['object']).columns):
            if col != 'Target':
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                label_encoders[col] = le
        X = processed_df.drop('Target', axis=1)
        y = processed_df['Target']
        model_columns = X.columns.tolist()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    else:
        if not model_columns:
            raise ValueError("Model not trained")
        for col in model_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0
        processed_df = processed_df[model_columns]
        for col, le in label_encoders.items():
            if col in processed_df.columns:
                known_labels = list(le.classes_)
                processed_df[col] = processed_df[col].astype(str).apply(
                    lambda x: x if x in known_labels else '<unknown>'
                )
                if '<unknown>' not in le.classes_:
                    le.classes_ = list(le.classes_) + ['<unknown>']
                processed_df[col] = le.transform(processed_df[col])
        return scaler.transform(processed_df)

def init_model(force_retrain=False):
    """Initializes the SVM model from cache or trains it from the base dataset."""
    global model, scaler, label_encoders, model_columns

    if not force_retrain and os.path.exists(MODEL_CACHE_FILE):
        try:
            cached_data = joblib.load(MODEL_CACHE_FILE)
            model = cached_data['model']
            scaler = cached_data['scaler']
            label_encoders = cached_data['label_encoders']
            model_columns = cached_data['model_columns']
            logger.info("Loaded pre-trained SVM model from cache.")
            return True
        except Exception as e:
            logger.warning("Failed to load model cache: %s. Retraining...", e)

    try:
        data_path = os.environ.get('TRAINING_DATA_PATH', 'Data.csv')
        if not os.path.exists(data_path):
            logger.error("Training dataset '%s' not found.", data_path)
            return False

        df_train = pd.read_csv(data_path)
        if not all(col in df_train.columns for col in REQUIRED_COLUMNS_TRAINING):
            raise ValueError("Training data is missing required columns.")

        X_train_scaled, y_train = preprocess_data(df_train, is_training=True)
        trained_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        trained_model.fit(X_train_scaled, y_train)

        model = trained_model

        try:
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'model_columns': model_columns
            }, MODEL_CACHE_FILE)
            logger.info("Model saved to cache (%s).", MODEL_CACHE_FILE)
        except Exception as cache_err:
            logger.warning("Could not persist model cache: %s", cache_err)

        logger.info("SVM model trained successfully.")
        return True
    except Exception as e:
        logger.error("Error during model training: %s", e)
        return False

# Ensure model is initialized when module is loaded (supports WSGI/Vercel/CLI)
init_model()

def generate_remarks(row):
    remarks = []
    if row.get('Tuition fees up to date') == 0:
        remarks.append("Tuition fees not up to date")
    if row.get('Previous qualification (grade)', 0) < 110:
        remarks.append("Low previous grade")
    if row.get('Age at enrollment', 0) > 25:
        remarks.append("Mature student")
    if row.get('Debtor') == 1:
        remarks.append("Is a debtor")
    if row.get('Curricular units 1st sem (approved)', 0) < 3:
        remarks.append("Low approved units in 1st sem")
    return ", ".join(remarks[:2]) if remarks else "Multiple contributing factors."

def send_report_email(excel_output, filename):
    """Constructs and sends an email with the Excel report as an attachment if configured."""
    username = app.config.get('MAIL_USERNAME')
    password = app.config.get('MAIL_PASSWORD')
    recipient = app.config.get('MAIL_RECIPIENT')

    if not username or not recipient or username == 'your_email@gmail.com':
        logger.info("Email notification skipped: SMTP credentials not configured.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = recipient
        msg['Subject'] = "Student Dropout Prediction Report"

        part = MIMEBase('application', "octet-stream")
        part.set_payload(excel_output.getvalue())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={filename}')
        msg.attach(part)

        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        if app.config.get('MAIL_USE_TLS'):
            server.starttls()
        if password and password != 'your_app_password':
            server.login(username, password)
        server.send_message(msg)
        server.quit()
        logger.info("Email report sent to %s", recipient)
        return True
    except Exception as e:
        logger.error("Failed to send email report: %s", e)
        return False

def create_excel_report(data_df):
    """Creates a color-coded Excel file in memory from a DataFrame."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data_df.to_excel(writer, index=False, sheet_name='Predictions')
        
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        high_risk_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        medium_risk_fill = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
        low_risk_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        
        risk_level_col_idx = data_df.columns.get_loc('Risk_Level') + 1

        for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row):
            risk_level_cell = worksheet.cell(row=row[0].row, column=risk_level_col_idx)
            fill_to_apply = None
            if risk_level_cell.value == 'High':
                fill_to_apply = high_risk_fill
            elif risk_level_cell.value == 'Medium':
                fill_to_apply = medium_risk_fill
            elif risk_level_cell.value == 'Low':
                fill_to_apply = low_risk_fill
            
            if fill_to_apply:
                for cell in row:
                    cell.fill = fill_to_apply
    output.seek(0)
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global processed_data, model
    if not model:
        init_model()
    if not model:
        flash('Model is currently unavailable. Please check server logs.', 'danger')
        return redirect(url_for('index'))

    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file selected.', 'warning')
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            df_raw = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            if 'Target' in df_raw.columns:
                flash('Uploaded file should not contain a "Target" column.', 'warning')
                return redirect(url_for('index'))
            if not all(col in df_raw.columns for col in REQUIRED_COLUMNS_PREDICTION):
                missing = [c for c in REQUIRED_COLUMNS_PREDICTION if c not in df_raw.columns]
                flash(f'File missing required columns: {", ".join(missing)}', 'danger')
                return redirect(url_for('index'))
            
            X_predict_scaled = preprocess_data(df_raw.copy(), is_training=False)
            probabilities = model.predict_proba(X_predict_scaled)[:, 1]
            results_df = df_raw.copy()
            results_df['Dropout_Probability'] = probabilities
            results_df['Risk_Level'] = pd.cut(
                probabilities, bins=[-0.1, 0.4, 0.7, 1.1], labels=['Low', 'Medium', 'High'], right=False
            ).astype(str)
            results_df['Remarks'] = results_df.apply(generate_remarks, axis=1)
            processed_data = results_df
            return redirect(url_for('dashboard'))
        except Exception as e:
            logger.exception("Upload processing error")
            flash(f'An error occurred while processing the file: {e}', 'danger')
            return redirect(url_for('index'))

    flash('Invalid file type. Allowed formats: CSV, XLSX, XLS', 'danger')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if processed_data is None:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

def format_student_details(student_data_raw):
    def get_info(label_en, label_hi, key, data_map=None, default='N/A'):
        val = student_data_raw.get(key)
        return {
            "label_en": label_en, 
            "label_hi": label_hi, 
            "value": data_map.get(val, default) if data_map and val is not None else (val if val is not None else default)
        }
    
    details = {
        'summary': student_data_raw.to_dict(),
        'academic': [
            get_info('Course', 'कोर्स', 'Course', COURSE_MAP),
            get_info('Attendance Type', 'उपस्थिति प्रकार', 'Daytime/evening attendance', {1: 'Daytime', 0: 'Evening'}),
            get_info('Previous Qualification', 'पिछली योग्यता', 'Previous qualification', QUALIFICATION_MAP),
            get_info('Previous Grade', 'पिछला ग्रेड', 'Previous qualification (grade)'),
            get_info('1st Sem Approved Units', 'प्रथम सेम स्वीकृत इकाइयां', 'Curricular units 1st sem (approved)'),
            get_info('1st Sem Grade', 'प्रथम सेम ग्रेड', 'Curricular units 1st sem (grade)'),
            get_info('Educational Special Needs', 'शैक्षिक विशेष आवश्यकताएं', 'Educational special needs', {1: 'Yes', 0: 'No'}),
        ],
        'demographics': [
            get_info('Age at Enrollment', 'नामांकन के समय आयु', 'Age at enrollment'),
            get_info('Gender', 'लिंग', 'Gender', {1: 'Male', 0: 'Female'}),
            get_info('Marital Status', 'वैवाहिक स्थिति', 'Marital status', MARITAL_STATUS_MAP),
            get_info('International Student', 'अंतर्राष्ट्रीय छात्र', 'International', {1: 'Yes', 0: 'No'}),
            get_info('Displaced', 'विस्थापित', 'Displaced', {1: 'Yes', 0: 'No'}),
        ],
        'financial': [
            get_info('Tuition Fees Up to Date', 'ट्यूशन फीस अद्यतित', 'Tuition fees up to date', {1: 'Yes', 0: 'No'}),
            get_info('Debtor Status', 'देनदार स्थिति', 'Debtor', {1: 'Yes', 0: 'No'}),
            get_info('Scholarship Holder', 'छात्रवृत्ति धारक', 'Scholarship holder', {1: 'Yes', 0: 'No'}),
        ],
        'family': [
            get_info("Mother's Qualification", "माँ की योग्यता", "Mother's qualification", QUALIFICATION_MAP),
            get_info("Father's Qualification", "पिता की योग्यता", "Father's qualification", QUALIFICATION_MAP),
            get_info("Mother's Occupation", "माँ का पेशा", "Mother's occupation", OCCUPATION_MAP),
            get_info("Father's Occupation", "पिता का पेशा", "Father's occupation", OCCUPATION_MAP),
        ]
    }
    details['summary']['Course'] = COURSE_MAP.get(details['summary'].get('Course'), 'N/A')
    return details

@app.route('/student/<int:student_id>')
def student_detail(student_id):
    if processed_data is None:
        return redirect(url_for('index'))
    try:
        student_data_raw = processed_data.iloc[student_id]
        student_full_details = format_student_details(student_data_raw)
        return render_template('student_detail.html', student=student_full_details, student_id=student_id)
    except IndexError:
        flash('Student ID not found.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/api/dashboard-data')
def api_dashboard_data():
    if processed_data is None:
        return jsonify({"error": "No data available."}), 404
    df = processed_data.copy()
    df['Course'] = df['Course'].map(COURSE_MAP)
    summary = {
        'total_students': len(df),
        'high_risk': int((df['Risk_Level'] == 'High').sum()),
        'medium_risk': int((df['Risk_Level'] == 'Medium').sum()),
        'low_risk': int((df['Risk_Level'] == 'Low').sum())
    }
    students = df.reset_index().rename(columns={'index': 'id'}).to_dict('records')
    return jsonify({"summary": summary, "students": students})

@app.route('/download_excel')
def download_excel():
    """Generates and sends a color-coded Excel report."""
    global processed_data
    if processed_data is None:
        flash('No data to process.', 'warning')
        return redirect(url_for('index'))

    try:
        excel_output = create_excel_report(processed_data)
        
        # Optional email notification
        if send_report_email(excel_output, 'prediction_results.xlsx'):
            flash(f"Report emailed to {app.config['MAIL_RECIPIENT']}.", 'success')
        
        excel_output.seek(0)
        return send_file(
            excel_output,
            as_attachment=True,
            download_name='prediction_results.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.exception("Error generating excel report")
        flash(f'An error occurred while processing the report: {e}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/sample-data')
def sample_data():
    """Provides a sample CSV file for users to download."""
    sample_file = 'sample_student_data.csv'
    if os.path.exists(sample_file):
        return send_file(sample_file, as_attachment=True, download_name='sample_student_data.csv')
    flash("Sample data file not found on the server.", "danger")
    return redirect(url_for('index'))


# =====================================================================
# Healthcare Analytics & BI-Ready Exports (Portfolio Extension)
# =====================================================================
from core_analytics.service import HealthcareAnalyticsService
from core_analytics.exports.bi_export_generator import (
    EXPORT_WHITELIST,
    generate_bi_export_tables,
    generate_manifest,
    export_all_to_zip_stream
)

@app.route('/healthcare')
def healthcare_dashboard():
    """Renders the comprehensive healthcare analytics & BI export dashboard."""
    try:
        service = HealthcareAnalyticsService.get_instance()
        data = service.get_dashboard_data()
        return render_template('healthcare_dashboard.html', **data)
    except Exception as e:
        logger.exception("Error loading healthcare dashboard")
        flash(f"Error loading healthcare analytics dashboard: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/healthcare/api/summary')
def healthcare_api_summary():
    """JSON API endpoint returning high-level healthcare operational summaries."""
    service = HealthcareAnalyticsService.get_instance()
    data = service.get_dashboard_data()
    return jsonify({
        "status": "success",
        "overview": data['overview'],
        "claims_kpis": data['claims_kpis'],
        "pmpm_metrics": data['pmpm_metrics'],
        "risk_distribution": data['risk_distribution'],
        "anomaly_summary": {
            "iqr_high_cost_count": data['anomaly_summary']['iqr_high_cost_count'],
            "iqr_upper_fence": data['anomaly_summary']['iqr_upper_fence'],
            "isolation_forest_anomalies": data['anomaly_summary']['isolation_forest_anomalies']
        },
        "process_summary": {
            "total_cases": data['process_summary']['total_cases'],
            "avg_cycle_time_days": data['process_summary']['avg_cycle_time_days'],
            "rework_case_rate_pct": data['process_summary']['rework_case_rate_pct'],
            "override_count": data['process_summary']['override_count']
        }
    })

@app.route('/healthcare/export/<filename>')
def healthcare_download_export(filename):
    """Safely serves a single BI export CSV or manifest.json with strict filename whitelisting."""
    if filename not in EXPORT_WHITELIST:
        return jsonify({"error": "File not found or export access denied."}), 404

    service = HealthcareAnalyticsService.get_instance()
    tables = generate_bi_export_tables(service)

    if filename == 'manifest.json':
        manifest = generate_manifest(tables)
        manifest_str = json.dumps(manifest, indent=2) if 'json' in globals() else __import__('json').dumps(manifest, indent=2)
        buf = io.BytesIO(manifest_str.encode('utf-8'))
        return send_file(
            buf,
            as_attachment=True,
            download_name='manifest.json',
            mimetype='application/json'
        )

    if filename in tables:
        df = tables[filename]
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(
            buf,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )

    return jsonify({"error": "Export generation failed for the requested table."}), 500

@app.route('/healthcare/export-all')
def healthcare_download_bundle():
    """Packages all 11 BI export CSVs and the manifest.json into a single downloadable ZIP archive."""
    try:
        service = HealthcareAnalyticsService.get_instance()
        zip_stream = export_all_to_zip_stream(service)
        return send_file(
            zip_stream,
            as_attachment=True,
            download_name='healthplans_ai_bi_export_bundle.zip',
            mimetype='application/zip'
        )
    except Exception as e:
        logger.exception("Error generating BI export bundle")
        flash(f"Error packaging BI export bundle: {e}", "danger")
        return redirect(url_for('healthcare_dashboard'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

