import os
import sys
import io
import pytest
import pandas as pd

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app as flask_app

@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    flask_app.app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.app.test_client() as client:
        with flask_app.app.app_context():
            # Ensure model is initialized
            if not flask_app.model:
                flask_app.init_model()
            yield client

def test_model_loaded(client):
    """Verify that the SVM model is initialized and can make predictions."""
    assert flask_app.model is not None
    assert flask_app.scaler is not None
    assert len(flask_app.REQUIRED_COLUMNS_PREDICTION) == 23

def test_index_route(client):
    """Test landing page renders successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Predict student" in response.data
    assert b"Upload Student Data" in response.data

def test_sample_data_download(client):
    """Test sample data route provides downloadable CSV."""
    response = client.get('/sample-data')
    assert response.status_code == 200
    assert len(response.data) > 0

def test_upload_missing_file(client):
    """Test uploading with no file selected."""
    response = client.post('/upload', data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"No file selected" in response.data

def test_upload_invalid_extension(client):
    """Test uploading an unsupported file format."""
    data = {'file': (io.BytesIO(b"fake data"), 'test.txt')}
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid file type" in response.data

def test_upload_file_with_target_column(client):
    """Test uploading a dataset that contains Target column (should be rejected for inference)."""
    df = pd.read_csv('Data.csv').head(5)
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    data = {'file': (csv_bytes, 'data_with_target.csv')}
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"Target" in response.data

def test_upload_valid_sample_and_dashboard_flow(client):
    """Test end-to-end flow: upload valid CSV, view dashboard, query API, view detail, export excel."""
    with open('sample_student_data.csv', 'rb') as f:
        file_bytes = io.BytesIO(f.read())

    data = {'file': (file_bytes, 'sample_student_data.csv')}
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard')

    # View Dashboard
    dash_response = client.get('/dashboard')
    assert dash_response.status_code == 200
    assert b"Overall Risk Distribution" in dash_response.data

    # Query API
    api_response = client.get('/api/dashboard-data')
    assert api_response.status_code == 200
    json_data = api_response.get_json()
    assert 'summary' in json_data
    assert 'students' in json_data
    assert json_data['summary']['total_students'] > 0
    assert 'high_risk' in json_data['summary']

    # View Student Detail
    student_response = client.get('/student/0')
    assert student_response.status_code == 200
    assert b"Student #" in student_response.data
    assert b"Key Risk Factors" in student_response.data

    # Download Excel Report
    excel_response = client.get('/download_excel')
    assert excel_response.status_code == 200
    assert 'spreadsheetml' in excel_response.content_type

def test_upload_excel_file(client):
    """Test uploading an Excel (.xlsx) file and generating predictions."""
    df = pd.read_csv('sample_student_data.csv')
    excel_bytes = io.BytesIO()
    df.to_excel(excel_bytes, index=False)
    excel_bytes.seek(0)

    data = {'file': (excel_bytes, 'sample.xlsx')}
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=False)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard')

def test_upload_missing_required_columns(client):
    """Test upload fails gracefully when required columns are absent."""
    df = pd.DataFrame({'Age at enrollment': [20, 22]})
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    data = {'file': (csv_bytes, 'missing_cols.csv')}
    response = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 200
    assert b"File missing required columns" in response.data

def test_student_detail_invalid_id(client):
    """Test navigating to an out-of-range student ID redirects to dashboard with error."""
    # Ensure there's data in the dashboard first
    with open('sample_student_data.csv', 'rb') as f:
        client.post('/upload', data={'file': (f, 'sample.csv')}, content_type='multipart/form-data')
    
    response = client.get('/student/999999', follow_redirects=True)
    assert response.status_code == 200
    assert b"Student ID not found" in response.data

