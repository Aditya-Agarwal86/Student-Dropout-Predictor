"""
SQL Storage & Profiling Module
Provides:
  - SQLite database schema initialization with Primary and Foreign Keys
  - Idempotent, parameterized data loading
  - Detailed profiling metrics (null counts/rates, row counts, distinct values, numerical stats, frequency distributions)
  - Parameterized analytical KPI queries (claims denial rates, cost per member, provider volume)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class HealthcareDatabaseManager:
    """Manages SQLite schema and data loading for synthetic healthcare data."""

    SCHEMA_DDL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS members (
        member_id TEXT PRIMARY KEY,
        gender TEXT,
        date_of_birth TEXT,
        plan_type TEXT,
        enrollment_start TEXT NOT NULL,
        enrollment_end TEXT,
        state TEXT,
        risk_score REAL
    );

    CREATE TABLE IF NOT EXISTS providers (
        provider_id TEXT PRIMARY KEY,
        npi TEXT,
        provider_name TEXT NOT NULL,
        specialty TEXT,
        state TEXT,
        status TEXT
    );

    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT,
        member_id TEXT,
        provider_id TEXT,
        claim_type TEXT,
        service_start_date TEXT,
        service_end_date TEXT,
        diagnosis_code TEXT,
        procedure_code TEXT,
        billed_amount REAL,
        paid_amount REAL,
        claim_status TEXT,
        denial_reason TEXT,
        FOREIGN KEY (member_id) REFERENCES members(member_id),
        FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
    );

    CREATE INDEX IF NOT EXISTS idx_claims_member ON claims(member_id);
    CREATE INDEX IF NOT EXISTS idx_claims_provider ON claims(provider_id);
    CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(claim_status);
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes tables and indexes."""
        conn = self.get_connection()
        conn.executescript(self.SCHEMA_DDL)
        conn.commit()


    def load_data(self, members_df: pd.DataFrame, providers_df: pd.DataFrame, claims_df: pd.DataFrame, clean_existing: bool = True):
        """
        Idempotent load of members, providers, and claims.
        Temporarily turns OFF PRAGMA foreign_keys during staging load so that
        raw data with controlled anomalies (e.g. orphan foreign keys) can be ingested
        for SQL profiling and DQ auditing without throwing raw fatal DB driver crashes.
        """
        conn = self.get_connection()
        try:
            conn.execute("PRAGMA foreign_keys = OFF;")
            if clean_existing:
                conn.execute("DELETE FROM claims;")
                conn.execute("DELETE FROM providers;")
                conn.execute("DELETE FROM members;")

            # Load members
            m_data = members_df.where(pd.notnull(members_df), None).to_dict(orient='records')
            conn.executemany("""
                INSERT OR REPLACE INTO members (
                    member_id, gender, date_of_birth, plan_type, enrollment_start, enrollment_end, state, risk_score
                ) VALUES (
                    :member_id, :gender, :date_of_birth, :plan_type, :enrollment_start, :enrollment_end, :state, :risk_score
                );
            """, m_data)

            # Load providers
            p_data = providers_df.where(pd.notnull(providers_df), None).to_dict(orient='records')
            conn.executemany("""
                INSERT OR REPLACE INTO providers (
                    provider_id, npi, provider_name, specialty, state, status
                ) VALUES (
                    :provider_id, :npi, :provider_name, :specialty, :state, :status
                );
            """, p_data)

            # Load claims
            c_data = claims_df.where(pd.notnull(claims_df), None).to_dict(orient='records')
            conn.executemany("""
                INSERT INTO claims (
                    claim_id, member_id, provider_id, claim_type, service_start_date, service_end_date,
                    diagnosis_code, procedure_code, billed_amount, paid_amount, claim_status, denial_reason
                ) VALUES (
                    :claim_id, :member_id, :provider_id, :claim_type, :service_start_date, :service_end_date,
                    :diagnosis_code, :procedure_code, :billed_amount, :paid_amount, :claim_status, :denial_reason
                );
            """, c_data)
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys = ON;")



class SQLDataProfiler:
    """SQL-based statistical profiling and health metric generator."""

    def __init__(self, db_manager: HealthcareDatabaseManager):
        self.db = db_manager

    def profile_table(self, table_name: str) -> Dict[str, Any]:
        """
        Profiles a given table with row count, column-level null rates,
        cardinality (distinct counts), and basic numerical/distribution stats.
        """
        with self.db.get_connection() as conn:
            # Table row count
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = cursor.fetchone()[0]

            # Get table pragma info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()

            columns_profile = {}
            for col in columns_info:
                col_name = col['name']
                col_type = col['type']

                # Null count & distinct count
                query = f"""
                SELECT 
                    COUNT(*) - COUNT({col_name}) AS null_count,
                    COUNT(DISTINCT {col_name}) AS distinct_count
                FROM {table_name}
                """
                cursor.execute(query)
                res = cursor.fetchone()
                null_count = res['null_count']
                distinct_count = res['distinct_count']
                null_rate = (null_count / total_rows) if total_rows > 0 else 0.0

                col_metrics = {
                    'type': col_type,
                    'null_count': null_count,
                    'null_rate': round(null_rate, 4),
                    'distinct_count': distinct_count
                }

                # Numeric metrics if REAL or INTEGER
                if col_type in ('REAL', 'INTEGER', 'NUMERIC'):
                    num_query = f"""
                    SELECT 
                        MIN({col_name}) as min_val,
                        MAX({col_name}) as max_val,
                        AVG({col_name}) as avg_val
                    FROM {table_name}
                    WHERE {col_name} IS NOT NULL
                    """
                    cursor.execute(num_query)
                    num_res = cursor.fetchone()
                    col_metrics.update({
                        'min': num_res['min_val'],
                        'max': num_res['max_val'],
                        'mean': round(num_res['avg_val'], 2) if num_res['avg_val'] is not None else None
                    })

                columns_profile[col_name] = col_metrics

            return {
                'table_name': table_name,
                'total_rows': total_rows,
                'columns': columns_profile
            }

    def get_column_frequency(self, table_name: str, column_name: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Calculates value frequencies and percentages for categorical columns."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total = cursor.fetchone()[0]

            query = f"""
            SELECT 
                COALESCE({column_name}, '[NULL]') as val,
                COUNT(*) as count
            FROM {table_name}
            GROUP BY {column_name}
            ORDER BY count DESC
            LIMIT ?
            """
            cursor.execute(query, (top_n,))
            rows = cursor.fetchall()

            result = []
            for r in rows:
                cnt = r['count']
                pct = round((cnt / total) * 100, 2) if total > 0 else 0.0
                result.append({'value': r['val'], 'count': cnt, 'percentage': pct})
            return result

    def get_duplicate_counts(self, table_name: str, key_column: str) -> Dict[str, Any]:
        """Identifies duplicate business key occurrences."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = f"""
            SELECT {key_column}, COUNT(*) as cnt
            FROM {table_name}
            WHERE {key_column} IS NOT NULL
            GROUP BY {key_column}
            HAVING COUNT(*) > 1
            """
            cursor.execute(query)
            dupes = cursor.fetchall()
            duplicate_keys = [{'key': d[0], 'count': d['cnt']} for d in dupes]
            return {
                'table_name': table_name,
                'key_column': key_column,
                'total_duplicate_keys': len(duplicate_keys),
                'details': duplicate_keys
            }

    def calculate_claims_kpis(self) -> Dict[str, Any]:
        """
        Calculates core healthcare claims KPIs using SQL:
          - Total claims volume
          - Total billed and paid spend
          - Overall claims denial rate
          - Average paid ratio on non-denied claims
          - Top denial reasons
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_claims,
                SUM(billed_amount) as total_billed,
                SUM(paid_amount) as total_paid,
                SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) as denied_claims,
                SUM(CASE WHEN claim_status = 'Paid' THEN 1 ELSE 0 END) as paid_claims
            FROM claims;
            """)
            summary = cursor.fetchone()

            total = summary['total_claims'] or 0
            denied = summary['denied_claims'] or 0
            denial_rate = round((denied / total) * 100, 2) if total > 0 else 0.0

            # Top denial reasons
            cursor.execute("""
            SELECT denial_reason, COUNT(*) as cnt
            FROM claims
            WHERE denial_reason IS NOT NULL
            GROUP BY denial_reason
            ORDER BY cnt DESC
            LIMIT 5;
            """)
            denial_breakdown = [{'reason': r['denial_reason'], 'count': r['cnt']} for r in cursor.fetchall()]

            # Provider cost concentration
            cursor.execute("""
            SELECT 
                p.provider_name,
                p.specialty,
                COUNT(c.claim_id) as claim_count,
                ROUND(SUM(c.paid_amount), 2) as total_paid
            FROM claims c
            JOIN providers p ON c.provider_id = p.provider_id
            GROUP BY p.provider_id
            ORDER BY total_paid DESC
            LIMIT 5;
            """)
            top_providers = [dict(r) for r in cursor.fetchall()]

            return {
                'total_claims': total,
                'total_billed': round(summary['total_billed'] or 0.0, 2),
                'total_paid': round(summary['total_paid'] or 0.0, 2),
                'denied_claims': denied,
                'paid_claims': summary['paid_claims'] or 0,
                'denial_rate_pct': denial_rate,
                'top_denial_reasons': denial_breakdown,
                'top_cost_providers': top_providers
            }
