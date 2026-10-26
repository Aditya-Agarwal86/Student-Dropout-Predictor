"""
Unit and Integration Tests for Module 6: Healthcare Analytics Dashboard and BI-Ready Exports
Validates:
  1. Healthcare service initialization and metric consistency.
  2. BI export tables generator (all 11 tables present, correct schemas, non-empty).
  3. Metadata manifest.json correctness, grain descriptions, and primary keys.
  4. Zip archive streaming without disk corruption.
  5. Flask routes:
      - GET /healthcare (200 OK, HTML contains synthetic warning & key KPIs)
      - GET /healthcare/api/summary (200 OK, valid JSON metrics)
      - GET /healthcare/export/<filename> (200 OK for whitelisted CSVs, 404 for unknown/traversal)
      - GET /healthcare/export-all (200 OK, valid application/zip)
  6. Non-regression of existing student application routes.
"""

import io
import json
import zipfile
import pytest
import pandas as pd
from app import app
from core_analytics.service import HealthcareAnalyticsService
from core_analytics.exports.bi_export_generator import (
    EXPORT_WHITELIST,
    generate_bi_export_tables,
    generate_manifest,
    export_all_to_zip_stream,
    export_all_to_directory
)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(scope="module")
def analytics_service():
    service = HealthcareAnalyticsService.get_instance(seed=42)
    service.initialize()
    return service


def test_service_initialization_and_dashboard_data(analytics_service):
    """Verifies that the healthcare analytics service populates all required dashboard sections."""
    data = analytics_service.get_dashboard_data()
    assert isinstance(data, dict)
    expected_sections = [
        'overview', 'dq_summary', 'claims_kpis', 'denial_reasons',
        'provider_analytics', 'pmpm_metrics', 'risk_distribution',
        'model_metrics', 'anomaly_summary', 'process_summary', 'nlp_eval_metrics'
    ]
    for section in expected_sections:
        assert section in data, f"Missing section '{section}' in dashboard data"

    # Validate high-level numbers
    assert data['overview']['total_claims_raw'] > 0
    assert data['overview']['total_members'] > 0
    assert data['overview']['total_providers'] > 0
    assert data['claims_kpis']['evaluated_claim_count'] > 0
    assert data['risk_distribution']['total_scored'] > 0


def test_bi_export_tables_generation(analytics_service):
    """Verifies that all 11 required BI CSV tables are produced with appropriate non-empty schemas."""
    tables = generate_bi_export_tables(analytics_service)
    assert isinstance(tables, dict)

    expected_csvs = [
        'claims_summary.csv',
        'provider_analytics.csv',
        'denial_breakdown.csv',
        'data_quality_results.csv',
        'anomaly_flags.csv',
        'rejection_risk_scores.csv',
        'transcript_extraction_results.csv',
        'process_case_summaries.csv',
        'process_variants.csv',
        'process_transitions.csv',
        'process_bottlenecks.csv'
    ]

    for filename in expected_csvs:
        assert filename in tables, f"Expected table '{filename}' not found in generated tables"
        df = tables[filename]
        assert isinstance(df, pd.DataFrame), f"Table '{filename}' is not a pandas DataFrame"
        assert not df.empty, f"Table '{filename}' should not be empty"

    # Specific schema assertions
    claims_df = tables['claims_summary.csv']
    assert 'claim_id' in claims_df.columns
    assert 'billed_amount' in claims_df.columns
    assert 'paid_amount' in claims_df.columns
    assert 'claim_status' in claims_df.columns

    scores_df = tables['rejection_risk_scores.csv']
    assert 'rejection_probability' in scores_df.columns
    assert 'predicted_risk_band' in scores_df.columns
    assert 'actual_observed_status' in scores_df.columns

    anom_df = tables['anomaly_flags.csv']
    assert 'claim_id' in anom_df.columns
    assert 'is_anomaly' in anom_df.columns

    cases_df = tables['process_case_summaries.csv']
    assert 'cycle_time_days' in cases_df.columns
    assert 'has_rework' in cases_df.columns


def test_bi_export_manifest_completeness(analytics_service):
    """Verifies that manifest.json correctly references each table and grain."""
    tables = generate_bi_export_tables(analytics_service)
    manifest = generate_manifest(tables)

    assert 'version' in manifest
    assert 'exported_tables' in manifest
    assert 'DEMONSTRATION' in manifest['data_notice']

    for filename in tables:
        assert filename in manifest['exported_tables'], f"Manifest missing entry for '{filename}'"
        entry = manifest['exported_tables'][filename]
        assert 'rows' in entry and entry['rows'] > 0
        assert 'columns' in entry and len(entry['columns']) > 0
        assert 'grain' in entry
        assert 'primary_key' in entry


def test_export_all_to_zip_stream(analytics_service):
    """Verifies that export_all_to_zip_stream produces a valid, readable zip buffer with all files."""
    zip_buffer = export_all_to_zip_stream(analytics_service)
    assert isinstance(zip_buffer, io.BytesIO)

    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        namelist = zf.namelist()
        assert 'manifest.json' in namelist
        assert 'claims_summary.csv' in namelist
        assert 'rejection_risk_scores.csv' in namelist
        assert 'process_case_summaries.csv' in namelist

        # Check manifest content from inside zip
        manifest_data = json.loads(zf.read('manifest.json').decode('utf-8'))
        assert manifest_data['version'] == '1.0.0'


def test_healthcare_dashboard_route(client):
    """Verifies that GET /healthcare renders 200 OK with expected synthetic warning and sections."""
    response = client.get('/healthcare')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Healthcare Operations" in html
    assert "Synthetic Healthcare Analytics Environment" in html
    assert "Total Billed" in html
    assert "Claims Denial Rate" in html
    assert "riskDistributionChart" in html
    assert "Export BI Bundle (.zip)" in html


def test_healthcare_api_summary_route(client):
    """Verifies that GET /healthcare/api/summary returns expected JSON schema."""
    response = client.get('/healthcare/api/summary')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'overview' in data
    assert 'claims_kpis' in data
    assert 'risk_distribution' in data
    assert 'process_summary' in data


def test_healthcare_download_export_csv_route(client):
    """Verifies that GET /healthcare/export/<filename> returns valid CSV attachments for whitelisted files."""
    for filename in ['claims_summary.csv', 'provider_analytics.csv', 'rejection_risk_scores.csv']:
        response = client.get(f'/healthcare/export/{filename}')
        assert response.status_code == 200
        assert response.mimetype == 'text/csv'
        assert f'filename={filename}' in response.headers.get('Content-Disposition', '')
        # Check CSV header
        content = response.data.decode('utf-8')
        assert len(content.splitlines()) > 1


def test_healthcare_download_manifest_route(client):
    """Verifies that GET /healthcare/export/manifest.json serves valid JSON."""
    response = client.get('/healthcare/export/manifest.json')
    assert response.status_code == 200
    assert response.mimetype == 'application/json'
    manifest = json.loads(response.data.decode('utf-8'))
    assert 'exported_tables' in manifest


def test_healthcare_download_export_security(client):
    """Verifies that unauthorized or directory traversal file requests are strictly rejected with 404."""
    traversal_attacks = [
        '../app.py',
        '..%2Fapp.py',
        'passwords.txt',
        'model_cache.joblib',
        'unknown_table.csv'
    ]
    for bad_file in traversal_attacks:
        response = client.get(f'/healthcare/export/{bad_file}')
        assert response.status_code in [404, 400], f"Expected 404/400 for '{bad_file}', got {response.status_code}"


def test_healthcare_download_bundle_route(client):
    """Verifies that GET /healthcare/export-all downloads the full zip archive."""
    response = client.get('/healthcare/export-all')
    assert response.status_code == 200
    assert response.mimetype == 'application/zip'
    assert 'healthplans_ai_bi_export_bundle.zip' in response.headers.get('Content-Disposition', '')

    zip_bytes = io.BytesIO(response.data)
    with zipfile.ZipFile(zip_bytes, 'r') as zf:
        assert 'manifest.json' in zf.namelist()
        assert 'claims_summary.csv' in zf.namelist()
        assert len(zf.namelist()) == 12


def test_preservation_of_student_app_routes(client):
    """Verifies that the original student dropout routes remain intact and functional."""
    r_index = client.get('/')
    assert r_index.status_code == 200

    r_sample = client.get('/sample-data')
    assert r_sample.status_code in [200, 302]


def test_financial_kpi_and_pmpm_reconciliation(analytics_service):
    """
    Verifies the mathematical and filtering distinction between:
      1. All-time deduplicated paid claims ($195,779.34 across FY 2023 and FY 2024)
      2. Strictly FY 2023 paid spend ($146,766.82) used for PMPM ($123.65 / 1,187 member-months)
    """
    dash_data = analytics_service.get_dashboard_data()
    claims_kpi = dash_data['claims_kpis']
    pmpm_kpi = dash_data['pmpm_metrics']

    # 1. Dashboard claims KPI has all-time deduplicated totals
    assert claims_kpi['evaluated_claim_count'] == 999
    assert claims_kpi['total_paid'] == 195779.34
    assert claims_kpi['denial_rate_pct'] == 14.11

    # 2. PMPM is strictly bounded to the FY 2023 window
    assert pmpm_kpi['period'] == "2023-01-01 to 2023-12-31"
    assert pmpm_kpi['evaluated_months'] == 12
    assert pmpm_kpi['total_paid_spend'] == 146766.82
    assert pmpm_kpi['total_member_months'] == 1187
    assert pmpm_kpi['pmpm'] == 123.65

    # 3. Mathematically confirm: total all-time paid > FY 2023 paid spend
    assert claims_kpi['total_paid'] > pmpm_kpi['total_paid_spend']

    # 4. Confirm exact formula: pmpm == round(spend / member_months, 2)
    assert pmpm_kpi['pmpm'] == round(pmpm_kpi['total_paid_spend'] / pmpm_kpi['total_member_months'], 2)

