"""
Healthcare Operational Analytics & KPI Module
Provides enterprise analytical aggregation and calculations:
  - Claim volume, billed/paid amounts, denial counts & denial rates
  - Categorical denial-reason distributions
  - Provider-level claim volume, reimbursement, and denial profiles
  - Enrollment trends by plan and calendar period
  - Per-Member-Per-Month (PMPM) expenditure with robust member-month denominator calculation
  - Integrated high-cost outlier detection using Module 2's IQROutlierDetector
  - Transparent data-quality defect handling:
      * Duplicate claims policy: Deduplicated on (claim_id) using first occurrence for financial KPIs,
        while maintaining audit logs of duplicate volume.
      * Invalid/orphan records policy: Preserved with explicit indicators; not silently discarded.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import sqlite3
from core_analytics.profiling.sql_profiler import HealthcareDatabaseManager
from core_analytics.anomaly.outlier_detector import IQROutlierDetector

class HealthcareAnalyticsEngine:
    """Computes operational KPIs and healthcare financial analytics."""

    def __init__(self, db_manager: HealthcareDatabaseManager):
        self.db = db_manager

    def get_claims_summary(self, deduplicate_claims: bool = True) -> Dict[str, Any]:
        """
        Calculates aggregate claim metrics.
        Policy: If deduplicate_claims is True, deduplicates by claim_id (keeping first)
        to prevent artificial inflation of financial amounts from duplicate claim submissions.
        """
        with self.db.get_connection() as conn:
            # Query claims
            df_claims = pd.read_sql_query("SELECT * FROM claims", conn)

        raw_count = len(df_claims)
        if raw_count == 0:
            return {
                'raw_claim_count': 0,
                'evaluated_claim_count': 0,
                'duplicate_claims_excluded': 0,
                'total_billed': 0.0,
                'total_paid': 0.0,
                'denied_count': 0,
                'denial_rate_pct': 0.0,
                'avg_paid_per_claim': 0.0
            }

        dup_mask = df_claims.duplicated(subset=['claim_id'], keep='first') & df_claims['claim_id'].notnull()
        dups_count = int(dup_mask.sum())

        eval_df = df_claims[~dup_mask].copy() if deduplicate_claims else df_claims.copy()
        evaluated_count = len(eval_df)

        # Financial totals (coerce non-numeric / negative values safely)
        valid_billed = pd.to_numeric(eval_df['billed_amount'], errors='coerce').fillna(0.0)
        valid_paid = pd.to_numeric(eval_df['paid_amount'], errors='coerce').fillna(0.0)

        total_billed = float(np.round(valid_billed[valid_billed > 0].sum(), 2))
        total_paid = float(np.round(valid_paid[valid_paid > 0].sum(), 2))

        denied_mask = eval_df['claim_status'].str.strip() == 'Denied'
        denied_count = int(denied_mask.sum())
        denial_rate = round((denied_count / evaluated_count) * 100, 2) if evaluated_count > 0 else 0.0

        paid_claims_count = int((eval_df['claim_status'].str.strip() == 'Paid').sum())
        avg_paid = round(total_paid / paid_claims_count, 2) if paid_claims_count > 0 else 0.0

        return {
            'raw_claim_count': raw_count,
            'evaluated_claim_count': evaluated_count,
            'duplicate_claims_excluded': dups_count if deduplicate_claims else 0,
            'total_billed': total_billed,
            'total_paid': total_paid,
            'denied_count': denied_count,
            'paid_count': paid_claims_count,
            'denial_rate_pct': denial_rate,
            'avg_paid_per_paid_claim': avg_paid
        }

    def get_denial_reason_breakdown(self) -> List[Dict[str, Any]]:
        """Returns frequency and proportion of denial reasons across all denied claims."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COALESCE(denial_reason, 'Unspecified / Missing') AS reason,
                    COUNT(*) AS count
                FROM claims
                WHERE claim_status = 'Denied'
                GROUP BY reason
                ORDER BY count DESC;
            """)
            rows = cursor.fetchall()

        total_denials = sum(r['count'] for r in rows)
        breakdown = []
        for r in rows:
            cnt = r['count']
            pct = round((cnt / total_denials) * 100, 2) if total_denials > 0 else 0.0
            breakdown.append({'denial_reason': r['reason'], 'count': cnt, 'percentage': pct})
        return breakdown

    def get_provider_analytics(self, min_claims: int = 1) -> List[Dict[str, Any]]:
        """
        Aggregates volume, billed/paid totals, and denial rate per rendering provider.
        Includes provider network status and handles orphan providers transparently.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT 
                    c.provider_id,
                    COALESCE(p.provider_name, '[Unmatched Provider / Orphan]') AS provider_name,
                    COALESCE(p.specialty, 'Unknown') AS specialty,
                    COALESCE(p.status, 'Unknown') AS network_status,
                    COUNT(c.claim_id) AS total_claims,
                    ROUND(SUM(CASE WHEN c.billed_amount > 0 THEN c.billed_amount ELSE 0 END), 2) AS total_billed,
                    ROUND(SUM(CASE WHEN c.paid_amount > 0 THEN c.paid_amount ELSE 0 END), 2) AS total_paid,
                    SUM(CASE WHEN c.claim_status = 'Denied' THEN 1 ELSE 0 END) AS denied_claims
                FROM claims c
                LEFT JOIN providers p ON c.provider_id = p.provider_id
                GROUP BY c.provider_id, p.provider_name, p.specialty, p.status
                HAVING COUNT(c.claim_id) >= ?
                ORDER BY total_paid DESC;
            """
            cursor.execute(query, (min_claims,))
            results = []
            for row in cursor.fetchall():
                tot = row['total_claims']
                denied = row['denied_claims']
                denial_rate = round((denied / tot) * 100, 2) if tot > 0 else 0.0
                results.append({
                    'provider_id': row['provider_id'],
                    'provider_name': row['provider_name'],
                    'specialty': row['specialty'],
                    'network_status': row['network_status'],
                    'total_claims': tot,
                    'total_billed': row['total_billed'] or 0.0,
                    'total_paid': row['total_paid'] or 0.0,
                    'denied_claims': denied,
                    'denial_rate_pct': denial_rate,
                    'is_orphan': 'Orphan' in row['provider_name']
                })
            return results

    def get_enrollment_trends(self) -> Dict[str, Any]:
        """Analyzes active enrollment counts, terminated members, and distribution across plan types."""
        with self.db.get_connection() as conn:
            df_members = pd.read_sql_query("SELECT * FROM members", conn)

        if len(df_members) == 0:
            return {'total_members': 0, 'active_members': 0, 'terminated_members': 0, 'plan_distribution': []}

        total_members = len(df_members)
        terminated_mask = df_members['enrollment_end'].notnull() & (df_members['enrollment_end'].astype(str).str.strip() != '')
        terminated_count = int(terminated_mask.sum())
        active_count = total_members - terminated_count

        plan_counts = df_members['plan_type'].value_counts()
        plan_dist = [
            {'plan_type': plan, 'count': int(cnt), 'percentage': round((cnt / total_members) * 100, 2)}
            for plan, cnt in plan_counts.items()
        ]

        return {
            'total_members': total_members,
            'active_members': active_count,
            'terminated_members': terminated_count,
            'termination_rate_pct': round((terminated_count / total_members) * 100, 2) if total_members > 0 else 0.0,
            'plan_distribution': plan_dist
        }

    def calculate_pmpm(self, start_period: str = "2023-01-01", end_period: str = "2023-12-31") -> Dict[str, Any]:
        """
        Calculates Per-Member-Per-Month (PMPM) expenditure:
          PMPM = Total Paid Claims in Period / Total Member-Months in Period
        Formula & Denominator Rules:
          - A member is credited with 1 member-month for each calendar month in [start_period, end_period]
            during which they were actively enrolled (start <= month_end and (end is null or end >= month_start)).
        """
        dt_start = pd.to_datetime(start_period)
        dt_end = pd.to_datetime(end_period)

        if dt_end < dt_start:
            raise ValueError(f"end_period ({end_period}) cannot be earlier than start_period ({start_period})")

        with self.db.get_connection() as conn:
            df_members = pd.read_sql_query("SELECT * FROM members", conn)
            # Query claims paid in period
            query_claims = """
                SELECT claim_id, member_id, billed_amount, paid_amount, service_start_date, claim_status
                FROM claims
                WHERE service_start_date >= ? AND service_start_date <= ?
                  AND claim_status = 'Paid' AND paid_amount > 0;
            """
            df_claims = pd.read_sql_query(query_claims, conn, params=(start_period, end_period))

        if len(df_members) == 0:
            return {'period': f"{start_period} to {end_period}", 'total_paid_spend': 0.0, 'member_months': 0, 'pmpm': 0.0}

        # Calculate monthly intervals
        month_starts = pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')
        total_member_months = 0

        m_starts = pd.to_datetime(df_members['enrollment_start'], errors='coerce')
        m_ends = pd.to_datetime(df_members['enrollment_end'], errors='coerce')

        for m_st in month_starts:
            m_en = m_st + pd.offsets.MonthEnd(1)
            # Active in month if enrollment_start <= m_en and (enrollment_end is NaT or enrollment_end >= m_st)
            active_mask = (m_starts <= m_en) & (m_ends.isna() | (m_ends >= m_st))
            total_member_months += int(active_mask.sum())

        # Deduplicate claims if duplicated
        deduped_claims = df_claims.drop_duplicates(subset=['claim_id'])
        total_paid_spend = float(np.round(deduped_claims['paid_amount'].sum(), 2))
        pmpm = round(total_paid_spend / total_member_months, 2) if total_member_months > 0 else 0.0

        return {
            'period': f"{start_period} to {end_period}",
            'evaluated_months': len(month_starts),
            'total_paid_spend': total_paid_spend,
            'total_member_months': total_member_months,
            'pmpm': pmpm
        }

    def identify_high_cost_claim_anomalies(self, k: float = 1.5) -> Dict[str, Any]:
        """
        Integrates Module 2's IQROutlierDetector to identify high-cost outlier claims.
        Assumptions: Evaluates positive billed amounts. Anomalies are statistical outliers,
        NEVER treated as fraud or automatic denials.
        """
        with self.db.get_connection() as conn:
            df_claims = pd.read_sql_query("SELECT * FROM claims", conn)

        valid_claims = df_claims[pd.to_numeric(df_claims['billed_amount'], errors='coerce') > 0].copy()
        if len(valid_claims) == 0:
            return {'total_claims': 0, 'high_cost_outliers_count': 0, 'outlier_claims': []}

        detector = IQROutlierDetector(k=k)
        detector.fit(valid_claims, features=['billed_amount'])
        summary = detector.score(valid_claims, id_column='claim_id', table_name='claims')

        # Filter upper-bound outliers
        ub = detector.fitted_bounds_['billed_amount']['upper_bound']
        high_cost_anomalies = [
            rec.to_dict() for rec in summary.anomalous_records
            if rec.feature_explanations.get('billed_amount', {}).get('reason') == 'ABOVE_UPPER_BOUND'
        ]

        return {
            'total_evaluated': len(valid_claims),
            'billed_amount_upper_fence': round(ub, 2),
            'high_cost_outliers_count': len(high_cost_anomalies),
            'outlier_claims': high_cost_anomalies
        }
