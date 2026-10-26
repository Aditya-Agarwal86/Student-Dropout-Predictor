"""
Declarative, Reusable Data Quality (DQ) Engine
Provides structured audit checks across:
  - Required columns and schema data types
  - Null / Missing values
  - Duplicate business keys
  - Numerical constraints & financial bounds (e.g. billed > 0, paid <= billed)
  - Date validity & chronological coherence (e.g. end_date >= start_date)
  - Allowed categorical values
  - Referential integrity across related tables
  - Configurable outlier warnings (e.g. high-cost claims)

Outputs structured DQResult objects with severity (ERROR vs WARNING), affected row IDs,
failure counts, and remediation guidance. Never silently alters or drops records.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

class DQSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"

@dataclass
class DQResult:
    check_name: str
    target_table: str
    target_column: Optional[str]
    severity: DQSeverity
    passed: bool
    total_evaluated: int
    failure_count: int
    failure_rate_pct: float
    affected_identifiers: List[Any] = field(default_factory=list)
    suggested_remediation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'check_name': self.check_name,
            'table': self.target_table,
            'column': self.target_column,
            'severity': self.severity.value,
            'passed': self.passed,
            'total_evaluated': self.total_evaluated,
            'failure_count': self.failure_count,
            'failure_rate_pct': self.failure_rate_pct,
            'affected_identifiers': self.affected_identifiers[:50], # Cap list for readability
            'suggested_remediation': self.suggested_remediation
        }


class DataQualityEngine:
    """Executes declarative data quality validation suites on pandas DataFrames."""

    def __init__(self):
        self.results: List[DQResult] = []

    def check_required_columns(self, df: pd.DataFrame, table_name: str, required_cols: List[str]) -> DQResult:
        """Verifies presence of required columns in the dataset."""
        missing = [col for col in required_cols if col not in df.columns]
        passed = len(missing) == 0
        res = DQResult(
            check_name="check_required_columns",
            target_table=table_name,
            target_column=None,
            severity=DQSeverity.ERROR,
            passed=passed,
            total_evaluated=len(required_cols),
            failure_count=len(missing),
            failure_rate_pct=round((len(missing) / len(required_cols)) * 100, 2) if required_cols else 0.0,
            affected_identifiers=missing,
            suggested_remediation="Add missing schema columns to dataset before ingestion." if not passed else "Schema valid."
        )
        self.results.append(res)
        return res

    def check_no_nulls(self, df: pd.DataFrame, table_name: str, column_name: str, id_column: str, severity: DQSeverity = DQSeverity.ERROR) -> DQResult:
        """Checks that mandatory columns do not contain null or empty values."""
        if column_name not in df.columns:
            return self._col_not_found(table_name, column_name)

        null_mask = df[column_name].isnull() | (df[column_name].astype(str).str.strip().isin(['', 'nan', 'None']))
        fails = df[null_mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_no_nulls",
            target_table=table_name,
            target_column=column_name,
            severity=severity,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Populate missing values for mandatory field '{column_name}' or request source correction."
        )
        self.results.append(res)
        return res

    def check_unique_key(self, df: pd.DataFrame, table_name: str, key_column: str) -> DQResult:
        """Verifies primary or business key uniqueness without duplicates."""
        if key_column not in df.columns:
            return self._col_not_found(table_name, key_column)

        dup_mask = df.duplicated(subset=[key_column], keep=False) & df[key_column].notnull()
        fails = df[dup_mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_keys = fails[key_column].unique().tolist()

        res = DQResult(
            check_name="check_unique_key",
            target_table=table_name,
            target_column=key_column,
            severity=DQSeverity.ERROR,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_keys,
            suggested_remediation=f"Deduplicate records by business key '{key_column}' using primary version or timestamp."
        )
        self.results.append(res)
        return res

    def check_allowed_values(self, df: pd.DataFrame, table_name: str, column_name: str, id_column: str, allowed_values: List[Any], severity: DQSeverity = DQSeverity.ERROR) -> DQResult:
        """Ensures categorical columns only contain domain-valid values."""
        if column_name not in df.columns:
            return self._col_not_found(table_name, column_name)

        invalid_mask = df[column_name].notnull() & (~df[column_name].isin(allowed_values))
        fails = df[invalid_mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_allowed_values",
            target_table=table_name,
            target_column=column_name,
            severity=severity,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Map unexpected values in '{column_name}' to standard vocabulary ({allowed_values})."
        )
        self.results.append(res)
        return res

    def check_numeric_range(self, df: pd.DataFrame, table_name: str, column_name: str, id_column: str, min_val: Optional[float] = None, max_val: Optional[float] = None, severity: DQSeverity = DQSeverity.ERROR) -> DQResult:
        """Verifies numeric fields adhere to mathematical or business bounds."""
        if column_name not in df.columns:
            return self._col_not_found(table_name, column_name)

        series = pd.to_numeric(df[column_name], errors='coerce')
        mask = pd.Series(False, index=df.index)
        if min_val is not None:
            mask = mask | (series < min_val)
        if max_val is not None:
            mask = mask | (series > max_val)

        fails = df[mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        bound_desc = f"between {min_val} and {max_val}" if min_val is not None and max_val is not None else (f">= {min_val}" if min_val is not None else f"<= {max_val}")
        res = DQResult(
            check_name="check_numeric_range",
            target_table=table_name,
            target_column=column_name,
            severity=severity,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Review out-of-bound entries in '{column_name}'; expected values {bound_desc}."
        )
        self.results.append(res)
        return res

    def check_financial_constraint(self, df: pd.DataFrame, table_name: str, id_column: str, billed_col: str = 'billed_amount', paid_col: str = 'paid_amount') -> DQResult:
        """Ensures paid amount does not exceed billed amount for any claim."""
        if billed_col not in df.columns or paid_col not in df.columns:
            return self._col_not_found(table_name, f"{billed_col}/{paid_col}")

        b = pd.to_numeric(df[billed_col], errors='coerce')
        p = pd.to_numeric(df[paid_col], errors='coerce')
        
        # Violations where paid > billed
        mask = (p > b) & b.notnull() & p.notnull()
        fails = df[mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_financial_constraint",
            target_table=table_name,
            target_column=f"{paid_col} <= {billed_col}",
            severity=DQSeverity.ERROR,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation="Adjust claims adjudication: paid amount cannot exceed billed charges."
        )
        self.results.append(res)
        return res

    def check_date_chronology(self, df: pd.DataFrame, table_name: str, id_column: str, start_col: str, end_col: str) -> DQResult:
        """Checks that start date precedes or equals end date."""
        if start_col not in df.columns or end_col not in df.columns:
            return self._col_not_found(table_name, f"{start_col}/{end_col}")

        dt_start = pd.to_datetime(df[start_col], errors='coerce')
        dt_end = pd.to_datetime(df[end_col], errors='coerce')

        mask = (dt_end < dt_start) & dt_start.notnull() & dt_end.notnull()
        fails = df[mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_date_chronology",
            target_table=table_name,
            target_column=f"{start_col} <= {end_col}",
            severity=DQSeverity.ERROR,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Correct inverted date sequence where '{end_col}' is earlier than '{start_col}'."
        )
        self.results.append(res)
        return res

    def check_referential_integrity(self, child_df: pd.DataFrame, child_table: str, child_fk_col: str, parent_df: pd.DataFrame, parent_table: str, parent_pk_col: str, id_column: str) -> DQResult:
        """Verifies that all child foreign keys resolve to a valid parent key."""
        if child_fk_col not in child_df.columns or parent_pk_col not in parent_df.columns:
            return self._col_not_found(child_table, f"{child_fk_col} -> {parent_table}.{parent_pk_col}")

        parent_keys = set(parent_df[parent_pk_col].dropna().unique())
        orphan_mask = child_df[child_fk_col].notnull() & (~child_df[child_fk_col].isin(parent_keys))
        fails = child_df[orphan_mask]
        fail_cnt = len(fails)
        total = len(child_df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in child_df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_referential_integrity",
            target_table=child_table,
            target_column=child_fk_col,
            severity=DQSeverity.ERROR,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Resolve orphaned {child_table}.{child_fk_col} keys against {parent_table}.{parent_pk_col} master table."
        )
        self.results.append(res)
        return res

    def check_outlier_warning(self, df: pd.DataFrame, table_name: str, column_name: str, id_column: str, threshold_std: float = 3.0) -> DQResult:
        """Identifies statistical numerical outliers as configurable WARNINGs."""
        if column_name not in df.columns:
            return self._col_not_found(table_name, column_name)

        series = pd.to_numeric(df[column_name], errors='coerce').dropna()
        if len(series) < 5:
            return DQResult(
                check_name="check_outlier_warning",
                target_table=table_name,
                target_column=column_name,
                severity=DQSeverity.WARNING,
                passed=True,
                total_evaluated=len(series),
                failure_count=0,
                failure_rate_pct=0.0,
                suggested_remediation="Insufficient sample size for outlier calculation."
            )

        mean = series.mean()
        std = series.std()
        upper_limit = mean + (threshold_std * std)
        lower_limit = mean - (threshold_std * std)

        outlier_mask = (df[column_name] > upper_limit) | (df[column_name] < lower_limit)
        fails = df[outlier_mask]
        fail_cnt = len(fails)
        total = len(df)
        pct = round((fail_cnt / total) * 100, 2) if total > 0 else 0.0
        affected_ids = fails[id_column].tolist() if id_column in df.columns else fails.index.tolist()

        res = DQResult(
            check_name="check_outlier_warning",
            target_table=table_name,
            target_column=column_name,
            severity=DQSeverity.WARNING,
            passed=(fail_cnt == 0),
            total_evaluated=total,
            failure_count=fail_cnt,
            failure_rate_pct=pct,
            affected_identifiers=affected_ids,
            suggested_remediation=f"Review {fail_cnt} statistical outliers (> {threshold_std} std dev from mean: {round(mean, 2)})."
        )
        self.results.append(res)
        return res

    def _col_not_found(self, table: str, col: str) -> DQResult:
        return DQResult(
            check_name="column_lookup",
            target_table=table,
            target_column=col,
            severity=DQSeverity.ERROR,
            passed=False,
            total_evaluated=0,
            failure_count=1,
            failure_rate_pct=100.0,
            suggested_remediation=f"Target column '{col}' does not exist in DataFrame."
        )

    def summarize_suite(self) -> Dict[str, Any]:
        """Provides high-level audit summary across all evaluated checks."""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_errors = sum(1 for r in self.results if not r.passed and r.severity == DQSeverity.ERROR)
        failed_warnings = sum(1 for r in self.results if not r.passed and r.severity == DQSeverity.WARNING)

        return {
            'total_checks_run': total_checks,
            'passed_checks': passed_checks,
            'failed_errors': failed_errors,
            'failed_warnings': failed_warnings,
            'overall_status': 'FAILED' if failed_errors > 0 else ('WARNING' if failed_warnings > 0 else 'PASSED'),
            'results': [r.to_dict() for r in self.results]
        }
