"""
Core Analytics Service Layer
Coordinates Modules 1–5 to provide cached, deterministic data and analytical
metrics for the Healthcare Dashboard and BI Exports without repeated expensive computations.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
from core_analytics.datasets.synthetic_healthcare import generate_synthetic_healthcare_data
from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager, SQLDataProfiler
from core_analytics.profiling.data_quality import DataQualityEngine
from core_analytics.healthcare.operational_analytics import HealthcareAnalyticsEngine
from core_analytics.healthcare.claims_models import ClaimRejectionRiskModel
from core_analytics.anomaly.outlier_detector import IQROutlierDetector, ModifiedZScoreOutlierDetector, IsolationForestOutlierDetector
from core_analytics.nlp.synthetic_transcripts import get_synthetic_call_transcripts
from core_analytics.nlp.transcript_analyzer import TranscriptExtractor, HumanReviewAuditManager, evaluate_extractions
from core_analytics.process_mining.synthetic_events import generate_synthetic_claims_event_log
from core_analytics.process_mining.process_analyzer import ProcessMiningEngine

class HealthcareAnalyticsService:
    """Singleton-style coordinator for healthcare analytics."""

    _instance: Optional["HealthcareAnalyticsService"] = None

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._initialized = False

        # Raw synthetic dataframes
        self.members_df: Optional[pd.DataFrame] = None
        self.providers_df: Optional[pd.DataFrame] = None
        self.claims_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.transcripts: Optional[list] = None

        # Core engines
        self.db_manager: Optional[HealthcareDatabaseManager] = None
        self.sql_profiler: Optional[SQLDataProfiler] = None
        self.analytics_engine: Optional[HealthcareAnalyticsEngine] = None
        self.rejection_model: Optional[ClaimRejectionRiskModel] = None
        self.review_manager: Optional[HumanReviewAuditManager] = None
        self.process_engine: Optional[ProcessMiningEngine] = None

        # Cached computed outputs
        self._cached_overview: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls, seed: int = 42) -> "HealthcareAnalyticsService":
        if cls._instance is None:
            cls._instance = cls(seed=seed)
            cls._instance.initialize()
        return cls._instance

    def initialize(self, force: bool = False):
        """Initializes all data, models, and analytical engines deterministically."""
        if self._initialized and not force:
            return

        # 1. Generate Synthetic Data
        self.members_df, self.providers_df, self.claims_df = generate_synthetic_healthcare_data(seed=self.seed)
        self.events_df = generate_synthetic_claims_event_log(n_cases=100, seed=self.seed)
        self.transcripts = get_synthetic_call_transcripts()

        # 2. SQLite Database & SQL Profiler
        self.db_manager = HealthcareDatabaseManager(db_path=":memory:")
        self.db_manager.load_data(self.members_df, self.providers_df, self.claims_df)
        self.sql_profiler = SQLDataProfiler(self.db_manager)

        # 3. Healthcare Operational Analytics
        self.analytics_engine = HealthcareAnalyticsEngine(self.db_manager)

        # 4. Train Claim Rejection Risk Model (Pipeline + ColumnTransformer)
        self.rejection_model = ClaimRejectionRiskModel(random_state=self.seed, n_estimators=40)
        self.rejection_model.train_and_evaluate(self.claims_df, self.members_df)

        # 5. NLP and Human Review Manager
        self.review_manager = HumanReviewAuditManager(self.db_manager)
        extractor = TranscriptExtractor(random_state=self.seed)
        extractor.fit_intent_classifier(self.transcripts[:10])
        # Save initial predictions and mock 2 reviews for audit trail demonstration
        for t in self.transcripts:
            rec = extractor.extract_from_text(t['transcript_id'], t['call_text'])
            self.review_manager.save_prediction(rec)
        self.review_manager.record_human_review(
            transcript_id="CALL-1002",
            reviewer_id="REV-Lead",
            review_status="Approved",
            final_intent="Denial Reason",
            final_claim_id="CLM-100582",
            final_denial_reason="Prior Authorization Required",
            comments="Verified against call recording."
        )

        # 6. Process Mining Engine
        self.process_engine = ProcessMiningEngine(self.events_df)

        self._initialized = True

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Gathers and caches all dashboard metrics across Modules 1–5."""
        if not self._initialized:
            self.initialize()

        if self._cached_overview is not None:
            return self._cached_overview

        # 1. Data Overview
        overview = {
            'total_claims_raw': len(self.claims_df),
            'total_members': len(self.members_df),
            'total_providers': len(self.providers_df),
            'total_events': len(self.events_df),
            'total_transcripts': len(self.transcripts)
        }

        # 2. Data Quality Suite (Module 1)
        dq = DataQualityEngine()
        dq.check_required_columns(self.claims_df, 'claims', ['claim_id', 'billed_amount', 'paid_amount', 'claim_status'])
        dq.check_no_nulls(self.claims_df, 'claims', 'member_id', id_column='claim_id')
        dq.check_unique_key(self.claims_df, 'claims', key_column='claim_id')
        dq.check_numeric_range(self.claims_df, 'claims', 'billed_amount', id_column='claim_id', min_val=0.0)
        dq.check_financial_constraint(self.claims_df, 'claims', id_column='claim_id')
        dq.check_date_chronology(self.claims_df, 'claims', id_column='claim_id', start_col='service_start_date', end_col='service_end_date')
        dq.check_referential_integrity(self.claims_df, 'claims', 'provider_id', self.providers_df, 'providers', 'provider_id', 'claim_id')
        dq.check_allowed_values(self.claims_df, 'claims', 'claim_status', id_column='claim_id', allowed_values=['Paid', 'Denied', 'In Process', 'Pending'])
        dq.check_outlier_warning(self.claims_df, 'claims', 'billed_amount', id_column='claim_id', threshold_std=3.0)
        dq_summary = dq.summarize_suite()

        # 3. Claims KPIs & Denial Breakdown (Module 3)
        claims_kpis = self.analytics_engine.get_claims_summary(deduplicate_claims=True)
        denial_reasons = self.analytics_engine.get_denial_reason_breakdown()
        provider_analytics = self.analytics_engine.get_provider_analytics(min_claims=1)
        pmpm_metrics = self.analytics_engine.calculate_pmpm("2023-01-01", "2023-12-31")

        # 4. Rejection Risk Distribution (Module 3)
        rejection_preds = self.rejection_model.predict_rejection_risk(self.claims_df, self.members_df)
        risk_counts = rejection_preds['predicted_risk_band'].value_counts().to_dict()
        risk_distribution = {
            'low': int(risk_counts.get('Low', 0)),
            'medium': int(risk_counts.get('Medium', 0)),
            'high': int(risk_counts.get('High', 0)),
            'total_scored': len(rejection_preds)
        }

        # 5. Anomaly Detection (Module 2)
        anom_res = self.analytics_engine.identify_high_cost_claim_anomalies(k=1.5)
        mod_z = ModifiedZScoreOutlierDetector(threshold=3.5).fit(self.claims_df, ['billed_amount'])
        z_res = mod_z.score(self.claims_df, id_column='claim_id')
        iso_f = IsolationForestOutlierDetector(contamination=0.03, random_state=42).fit(self.claims_df, ['billed_amount', 'paid_amount'])
        iso_res = iso_f.score(self.claims_df, id_column='claim_id')

        anomaly_summary = {
            'iqr_high_cost_count': anom_res['high_cost_outliers_count'],
            'iqr_upper_fence': anom_res['billed_amount_upper_fence'],
            'modified_z_outliers': z_res.anomaly_count,
            'isolation_forest_anomalies': iso_res.anomaly_count,
            'iqr_sample_records': anom_res['outlier_claims'][:5]
        }

        # 6. Process Mining (Module 5)
        case_summaries = self.process_engine.get_case_summaries()
        variants_df = self.process_engine.get_process_variants()
        bottlenecks_df = self.process_engine.identify_bottlenecks(threshold_std=1.0)
        overrides = self.process_engine.get_override_analysis()

        process_summary = {
            'total_cases': len(case_summaries),
            'avg_cycle_time_days': round(float(case_summaries['cycle_time_days'].mean()), 2) if not case_summaries.empty else 0.0,
            'rework_case_rate_pct': round(float(case_summaries['has_rework'].mean()) * 100, 2) if not case_summaries.empty else 0.0,
            'override_count': overrides['total_override_events'],
            'top_variants': variants_df.head(5).to_dict(orient='records') if not variants_df.empty else [],
            'bottlenecks': bottlenecks_df.to_dict(orient='records') if not bottlenecks_df.empty else []
        }

        # 7. Transcript Human Review Status (Module 4)
        extractor = TranscriptExtractor(random_state=42).fit_intent_classifier(self.transcripts[:10])
        preds = [extractor.extract_from_text(t['transcript_id'], t['call_text']) for t in self.transcripts]
        eval_metrics = evaluate_extractions(preds, self.transcripts)

        self._cached_overview = {
            'overview': overview,
            'dq_summary': dq_summary,
            'claims_kpis': claims_kpis,
            'denial_reasons': denial_reasons,
            'provider_analytics': provider_analytics[:10], # Top 10 for dashboard preview
            'pmpm_metrics': pmpm_metrics,
            'risk_distribution': risk_distribution,
            'model_metrics': self.rejection_model.evaluation_metrics,
            'anomaly_summary': anomaly_summary,
            'process_summary': process_summary,
            'nlp_eval_metrics': eval_metrics['field_metrics']
        }
        return self._cached_overview
