"""
BI-Ready Export Generator
Produces documented, standardized CSV tables and manifest for Power BI / Tableau consumption.
All data is derived from the synthetic healthcare analytics service layer.
"""

import os
import json
import io
import zipfile
from typing import Dict, Any, List
import pandas as pd

from core_analytics.service import HealthcareAnalyticsService

EXPORT_WHITELIST = {
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
    'process_bottlenecks.csv',
    'manifest.json'
}

def generate_bi_export_tables(service: HealthcareAnalyticsService) -> Dict[str, pd.DataFrame]:
    """Generates all 11 required BI dataframes adhering to documented grain and schemas."""
    tables: Dict[str, pd.DataFrame] = {}

    # Ensure service is initialized
    if not service._initialized:
        service.initialize()

    # 1. claims_summary.csv (Deduplicated Claim Level)
    claims_df = service.claims_df.copy()
    # Deduplicate keeping first occurrence, matching HealthcareAnalyticsEngine policy
    dup_mask = claims_df.duplicated(subset=['claim_id'], keep='first') & claims_df['claim_id'].notnull()
    dedup_claims = claims_df[~dup_mask].copy()
    tables['claims_summary.csv'] = dedup_claims[[
        'claim_id', 'member_id', 'provider_id', 'claim_type', 'service_start_date',
        'service_end_date', 'billed_amount', 'paid_amount', 'claim_status', 'denial_reason'
    ]].copy()

    # 2. provider_analytics.csv (Provider Level)
    prov_metrics = service.analytics_engine.get_provider_analytics(min_claims=1)
    tables['provider_analytics.csv'] = pd.DataFrame(prov_metrics)

    # 3. denial_breakdown.csv (Denial Reason Aggregate Level)
    denials = service.analytics_engine.get_denial_reason_breakdown()
    tables['denial_breakdown.csv'] = pd.DataFrame(denials)

    # 4. data_quality_results.csv (Rule Level)
    dashboard_data = service.get_dashboard_data()
    dq_results = dashboard_data['dq_summary']['results']
    tables['data_quality_results.csv'] = pd.DataFrame(dq_results)

    # 5. anomaly_flags.csv (Claim Level Outlier Flags)
    # Using IQR detector from Module 2 on positive billed_amount
    from core_analytics.anomaly.outlier_detector import IQROutlierDetector
    detector = IQROutlierDetector(k=1.5)
    valid_mask = pd.to_numeric(dedup_claims['billed_amount'], errors='coerce') > 0
    detector.fit(dedup_claims[valid_mask], features=['billed_amount'])
    det_res = detector.score(dedup_claims, id_column='claim_id', table_name='claims')
    anom_rows = []
    for rec in det_res.anomalous_records:
        anom_rows.append({
            'claim_id': rec.record_id,
            'anomaly_score': rec.anomaly_score,
            'is_anomaly': rec.is_anomaly,
            'method': rec.method.value,
            'description': 'Statistical High-Cost Outlier (Requires Review - Not Proof of Fraud)'
        })
    tables['anomaly_flags.csv'] = pd.DataFrame(anom_rows) if anom_rows else pd.DataFrame(
        columns=['claim_id', 'anomaly_score', 'is_anomaly', 'method', 'description']
    )

    # 6. rejection_risk_scores.csv (Model Predictions on Active/Scored Claims)
    if service.rejection_model and service.rejection_model.pipeline is not None:
        scores_df = service.rejection_model.predict_rejection_risk(claims_df, service.members_df)
        tables['rejection_risk_scores.csv'] = scores_df[[
            'claim_id', 'rejection_probability', 'predicted_risk_band', 'actual_observed_status'
        ]].copy()
    else:
        tables['rejection_risk_scores.csv'] = pd.DataFrame(columns=[
            'claim_id', 'rejection_probability', 'predicted_risk_band', 'actual_observed_status'
        ])

    # 7. transcript_extraction_results.csv (Transcript Level NLP & Audit)
    if service.review_manager and service.db_manager:
        with service.db_manager.get_connection() as conn:
            query = """
                SELECT 
                    p.transcript_id,
                    p.predicted_intent,
                    p.predicted_claim_id,
                    p.predicted_denial_reason,
                    p.predicted_action,
                    p.predicted_urgency,
                    p.confidence_score,
                    r.reviewer_id,
                    r.review_status,
                    r.final_intent,
                    r.final_claim_id,
                    r.final_denial_reason,
                    r.reviewer_comments,
                    r.review_timestamp
                FROM transcript_predictions p
                LEFT JOIN transcript_human_reviews r ON p.transcript_id = r.transcript_id
            """
            tables['transcript_extraction_results.csv'] = pd.read_sql_query(query, conn)
    else:
        tables['transcript_extraction_results.csv'] = pd.DataFrame()

    # 8. process_case_summaries.csv (Case Level Process Performance)
    tables['process_case_summaries.csv'] = service.process_engine.get_case_summaries()

    # 9. process_variants.csv (Variant Level Path Patterns)
    tables['process_variants.csv'] = service.process_engine.get_process_variants()

    # 10. process_transitions.csv (Directed Activity-to-Activity Edge Level)
    tables['process_transitions.csv'] = service.process_engine.get_transition_matrix()

    # 11. process_bottlenecks.csv (Bottleneck Transition Candidates)
    tables['process_bottlenecks.csv'] = service.process_engine.identify_bottlenecks()

    return tables


def generate_manifest(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generates the metadata manifest documenting grain, keys, and purpose for each export."""
    manifest = {
        "dataset_name": "HealthPlans.ai Synthetic Operational Analytics Export Bundle",
        "data_notice": "DEMONSTRATION ONLY. All records are synthetically generated and contain controlled quality defects for analytics evaluation.",
        "version": "1.0.0",
        "exported_tables": {}
    }

    metadata_map = {
        'claims_summary.csv': {
            'grain': 'One row per unique claim (Deduplicated)',
            'primary_key': 'claim_id',
            'foreign_keys': {'member_id': 'members', 'provider_id': 'providers'},
            'description': 'Core claims operational fact table containing billed, paid, and settlement status.'
        },
        'provider_analytics.csv': {
            'grain': 'One row per provider',
            'primary_key': 'provider_id',
            'foreign_keys': {},
            'description': 'Operational provider scorecard tracking claim volumes, denial rates, and revenue.'
        },
        'denial_breakdown.csv': {
            'grain': 'One row per denial reason code',
            'primary_key': 'denial_reason',
            'foreign_keys': {},
            'description': 'Aggregated denial frequencies and financial impact for root-cause analysis.'
        },
        'data_quality_results.csv': {
            'grain': 'One row per data quality validation rule',
            'primary_key': 'rule_name',
            'foreign_keys': {},
            'description': 'Automated quality test results, severity levels, and affected record counts.'
        },
        'anomaly_flags.csv': {
            'grain': 'One row per claim evaluated for statistical anomalies',
            'primary_key': 'claim_id',
            'foreign_keys': {'claim_id': 'claims_summary.csv'},
            'description': 'Statistical anomaly flags (Tukey/IQR) flagged for review (not proof of fraud).'
        },
        'rejection_risk_scores.csv': {
            'grain': 'One row per evaluated claim',
            'primary_key': 'claim_id',
            'foreign_keys': {'claim_id': 'claims_summary.csv'},
            'description': 'Supervised ML model predicted rejection probability and risk tier (Low/Medium/High).'
        },
        'transcript_extraction_results.csv': {
            'grain': 'One row per member call transcript extraction and human review record',
            'primary_key': 'transcript_id',
            'foreign_keys': {'extracted_claim_id': 'claims_summary.csv'},
            'description': 'NLP intent classification, entity extraction, and compliance audit trail.'
        },
        'process_case_summaries.csv': {
            'grain': 'One row per claim adjudication workflow case',
            'primary_key': 'claim_id',
            'foreign_keys': {'claim_id': 'claims_summary.csv'},
            'description': 'Process mining case cycle times, event counts, rework flags, and override flags.'
        },
        'process_variants.csv': {
            'grain': 'One row per unique process execution pathway',
            'primary_key': 'variant_id',
            'foreign_keys': {},
            'description': 'Workflow path taxonomy with case counts, share percentages, and cycle times.'
        },
        'process_transitions.csv': {
            'grain': 'One row per directed activity-to-activity transition',
            'primary_key': ['source_activity', 'target_activity'],
            'foreign_keys': {},
            'description': 'Markovian transition network with frequencies and median/mean transition durations.'
        },
        'process_bottlenecks.csv': {
            'grain': 'One row per detected workflow bottleneck transition',
            'primary_key': ['source_activity', 'target_activity'],
            'foreign_keys': {},
            'description': 'Statistical stage delays exceeding operational baseline thresholds.'
        }
    }

    for filename, df in tables.items():
        meta = metadata_map.get(filename, {'grain': 'Custom', 'primary_key': 'N/A', 'description': 'Export table'})
        manifest['exported_tables'][filename] = {
            'rows': len(df),
            'columns': list(df.columns),
            'grain': meta['grain'],
            'primary_key': meta['primary_key'],
            'description': meta['description']
        }

    return manifest


def export_all_to_directory(output_dir: str, service: HealthcareAnalyticsService = None) -> List[str]:
    """Writes all 11 CSV files and the manifest.json into output_dir. Returns list of written files."""
    if service is None:
        service = HealthcareAnalyticsService.get_instance()

    os.makedirs(output_dir, exist_ok=True)
    tables = generate_bi_export_tables(service)
    manifest = generate_manifest(tables)

    written_paths = []
    for filename, df in tables.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        written_paths.append(filepath)

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    written_paths.append(manifest_path)

    return written_paths


def export_all_to_zip_stream(service: HealthcareAnalyticsService = None) -> io.BytesIO:
    """Generates an in-memory zip archive containing all 11 CSVs and the manifest.json."""
    if service is None:
        service = HealthcareAnalyticsService.get_instance()

    tables = generate_bi_export_tables(service)
    manifest = generate_manifest(tables)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, df in tables.items():
            csv_str = df.to_csv(index=False)
            zip_file.writestr(filename, csv_str)
        zip_file.writestr('manifest.json', json.dumps(manifest, indent=2))

    zip_buffer.seek(0)
    return zip_buffer


if __name__ == '__main__':
    # CLI entry point to generate exports locally
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    written = export_all_to_directory(out_dir)
    print(f"Generated {len(written)} BI export files in {out_dir}:")
    for w in written:
        print(f" - {os.path.basename(w)}")
