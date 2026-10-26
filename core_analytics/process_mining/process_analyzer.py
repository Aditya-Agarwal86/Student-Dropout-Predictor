"""
Operational Process Mining Engine for Healthcare Claims Workflows
Provides:
  1. Event log schema validation and chronological sequence ordering.
  2. Case-level cycle time & duration metrics (in hours and days).
  3. Repeated activities and rework loop detection (e.g. repeated reviews/resubmissions).
  4. Activity transition frequencies and common process variants.
  5. Manual override tracking and categorical reason distribution.
  6. Transition duration summaries and bottleneck candidate identification.
  7. Outcome distribution summaries per process variant.
  8. Structured DataFrame generation and CSV export capabilities.

Transparent Defect Policy:
  - Missing/null timestamps: Filtered from duration calculations and surfaced in data quality audit counts.
  - Duplicate events (identical case_id, activity, timestamp): Deduplicated to prevent zero-duration transition distortion.
  - Single-event cases: Handled safely with cycle time = 0.0 hours.
  - Out-of-order events: Automatically ordered chronologically by (case_id, parsed_timestamp).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class ProcessMiningEngine:
    """Performs process discovery, variant mining, rework detection, and bottleneck analysis."""

    REQUIRED_COLUMNS = ['case_id', 'activity', 'event_timestamp', 'resource', 'outcome_status']

    def __init__(self, event_log: pd.DataFrame):
        self.raw_log = event_log.copy()
        self.clean_log, self.dq_audit = self._clean_and_order_log(self.raw_log)

    def _clean_and_order_log(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates schema, filters invalid rows safely, removes duplicate events,
        and ensures strict chronological order within each case_id.
        """
        if df.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS), {
                'raw_events': 0, 'missing_timestamps': 0, 'duplicates_removed': 0, 'clean_events': 0
            }

        # Validate required columns
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Event log is missing required schema columns: {missing_cols}")

        raw_count = len(df)
        clean = df.copy()

        # Parse timestamps safely
        clean['parsed_timestamp'] = pd.to_datetime(clean['event_timestamp'], errors='coerce')
        missing_ts_count = int(clean['parsed_timestamp'].isna().sum())

        # Exclude unparsable/null timestamps for chronological sequencing
        clean = clean[clean['parsed_timestamp'].notna()].copy()

        # Deduplicate identical event logs (case_id, activity, parsed_timestamp)
        dup_mask = clean.duplicated(subset=['case_id', 'activity', 'parsed_timestamp'], keep='first')
        dup_count = int(dup_mask.sum())
        clean = clean[~dup_mask].copy()

        # Sort chronologically by case_id and timestamp
        clean = clean.sort_values(by=['case_id', 'parsed_timestamp']).reset_index(drop=True)

        audit = {
            'raw_events': raw_count,
            'missing_timestamps': missing_ts_count,
            'duplicates_removed': dup_count,
            'clean_events': len(clean)
        }
        return clean, audit

    def get_case_summaries(self) -> pd.DataFrame:
        """
        Calculates case-level metrics:
          - start_time, end_time
          - event_count
          - cycle_time_hours, cycle_time_days
          - distinct_activities_count
          - has_rework (boolean)
          - rework_activity_count
          - has_override (boolean)
          - final_outcome_status
          - variant_path (e.g. 'Claim Submitted -> Initial Review -> Approved')
        """
        if self.clean_log.empty:
            return pd.DataFrame()

        cases = []
        for case_id, group in self.clean_log.groupby('case_id', sort=False):
            events = group['activity'].tolist()
            t_start = group['parsed_timestamp'].min()
            t_end = group['parsed_timestamp'].max()
            duration_hrs = round((t_end - t_start).total_seconds() / 3600.0, 2)
            duration_days = round(duration_hrs / 24.0, 2)

            # Rework: repeated activities
            activity_counts = pd.Series(events).value_counts()
            rework_activities = activity_counts[activity_counts > 1].to_dict()
            has_rework = len(rework_activities) > 0
            rework_count = int(sum(cnt - 1 for cnt in rework_activities.values()))

            # Override
            has_override = bool(group['is_override'].any()) if 'is_override' in group.columns else False
            final_status = group['outcome_status'].iloc[-1] if 'outcome_status' in group.columns else 'Unknown'
            variant_path = " -> ".join(events)

            cases.append({
                'case_id': case_id,
                'start_time': t_start,
                'end_time': t_end,
                'event_count': len(events),
                'cycle_time_hours': duration_hrs,
                'cycle_time_days': duration_days,
                'distinct_activities': len(set(events)),
                'has_rework': has_rework,
                'rework_events_count': rework_count,
                'has_override': has_override,
                'final_outcome': final_status,
                'variant_path': variant_path
            })

        return pd.DataFrame(cases)

    def get_process_variants(self) -> pd.DataFrame:
        """
        Discovers and ranks all process variants (unique activity sequences),
        including frequency, percentage of total cases, average cycle time,
        and primary outcome distribution.
        """
        case_df = self.get_case_summaries()
        if case_df.empty:
            return pd.DataFrame()

        total_cases = len(case_df)
        variants = []

        for path, group in case_df.groupby('variant_path'):
            count = len(group)
            pct = round((count / total_cases) * 100, 2)
            avg_hrs = round(float(group['cycle_time_hours'].mean()), 2)
            avg_days = round(float(group['cycle_time_days'].mean()), 2)
            rework_pct = round(float(group['has_rework'].mean()) * 100, 2)
            override_pct = round(float(group['has_override'].mean()) * 100, 2)

            # Outcome distribution
            outcomes = group['final_outcome'].value_counts().to_dict()

            variants.append({
                'variant_path': path,
                'case_count': count,
                'case_percentage': pct,
                'avg_cycle_time_hours': avg_hrs,
                'avg_cycle_time_days': avg_days,
                'rework_case_pct': rework_pct,
                'override_case_pct': override_pct,
                'outcome_distribution': str(outcomes)
            })

        variant_df = pd.DataFrame(variants).sort_values(by='case_count', ascending=False).reset_index(drop=True)
        return variant_df

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Calculates direct-follows activity transitions:
          - from_activity -> to_activity
          - transition_count
          - avg_duration_hours (time elapsed between activity transitions)
          - median_duration_hours
          - max_duration_hours
        """
        if self.clean_log.empty:
            return pd.DataFrame()

        transitions = []
        for case_id, group in self.clean_log.groupby('case_id', sort=False):
            if len(group) < 2:
                continue
            rows = group[['activity', 'parsed_timestamp']].values
            for i in range(len(rows) - 1):
                from_act, from_ts = rows[i]
                to_act, to_ts = rows[i + 1]
                dur_hrs = (to_ts - from_ts).total_seconds() / 3600.0
                transitions.append({
                    'from_activity': from_act,
                    'to_activity': to_act,
                    'duration_hours': max(0.0, dur_hrs)
                })

        if not transitions:
            return pd.DataFrame()

        trans_df = pd.DataFrame(transitions)
        summary = trans_df.groupby(['from_activity', 'to_activity']).agg(
            transition_count=('duration_hours', 'count'),
            avg_duration_hours=('duration_hours', 'mean'),
            median_duration_hours=('duration_hours', 'median'),
            max_duration_hours=('duration_hours', 'max')
        ).reset_index()

        summary['avg_duration_hours'] = summary['avg_duration_hours'].round(2)
        summary['median_duration_hours'] = summary['median_duration_hours'].round(2)
        summary['max_duration_hours'] = summary['max_duration_hours'].round(2)
        return summary.sort_values(by='transition_count', ascending=False).reset_index(drop=True)

    def identify_bottlenecks(self, threshold_std: float = 1.5) -> pd.DataFrame:
        """
        Identifies potential process bottlenecks dynamically:
        Transitions whose mean duration exceeds (overall transition mean + threshold_std * std).
        """
        trans_df = self.get_transition_matrix()
        if trans_df.empty or len(trans_df) < 2:
            return pd.DataFrame()

        mean_dur = trans_df['avg_duration_hours'].mean()
        std_dur = trans_df['avg_duration_hours'].std()
        cutoff = mean_dur + (threshold_std * std_dur)

        bottlenecks = trans_df[trans_df['avg_duration_hours'] > cutoff].copy()
        bottlenecks['bottleneck_severity_score'] = (
            (bottlenecks['avg_duration_hours'] - mean_dur) / (std_dur if std_dur > 0 else 1.0)
        ).round(2)

        return bottlenecks.sort_values(by='avg_duration_hours', ascending=False).reset_index(drop=True)

    def get_override_analysis(self) -> Dict[str, Any]:
        """Summarizes manual override frequencies and reason code breakdown."""
        if 'is_override' not in self.clean_log.columns:
            return {'total_overrides': 0, 'reason_breakdown': []}

        overrides = self.clean_log[self.clean_log['is_override'] == True]
        total = len(overrides)

        reasons = overrides['override_reason'].fillna('Unspecified Reason').value_counts().to_dict()
        reasons_list = [
            {'reason': r, 'count': cnt, 'percentage': round((cnt / total) * 100, 2) if total > 0 else 0.0}
            for r, cnt in reasons.items()
        ]

        return {
            'total_override_events': total,
            'affected_cases_count': int(overrides['case_id'].nunique()),
            'reason_breakdown': reasons_list
        }

    def export_all_metrics_to_csv(self, export_dir: str = ".") -> Dict[str, str]:
        """Exports all analytical tables to CSV for downstream Tableau/Power BI or reporting."""
        import os
        os.makedirs(export_dir, exist_ok=True)

        case_p = os.path.join(export_dir, "process_case_summaries.csv")
        var_p = os.path.join(export_dir, "process_variants.csv")
        trans_p = os.path.join(export_dir, "process_transitions.csv")
        bottle_p = os.path.join(export_dir, "process_bottlenecks.csv")

        self.get_case_summaries().to_csv(case_p, index=False)
        self.get_process_variants().to_csv(var_p, index=False)
        self.get_transition_matrix().to_csv(trans_p, index=False)
        self.identify_bottlenecks().to_csv(bottle_p, index=False)

        return {
            'case_summaries': case_p,
            'process_variants': var_p,
            'transitions': trans_p,
            'bottlenecks': bottle_p
        }
