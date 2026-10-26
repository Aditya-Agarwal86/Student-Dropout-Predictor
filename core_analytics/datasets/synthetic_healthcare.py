"""
Deterministic Synthetic Healthcare Dataset Generator
Generates fictional datasets for Members/Enrollment, Providers, and Claims
with controlled, realistic data-quality anomalies for profiling & testing.
All records are strictly fictional and generated with fixed random seeds.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_healthcare_data(seed=42):
    """
    Generates three DataFrames:
      - members_df (Enrollment data)
      - providers_df (Healthcare providers)
      - claims_df (Healthcare claims referencing members & providers)
    Includes deliberate, controlled data quality issues:
      - Duplicate claim IDs
      - Missing required values (e.g. missing diagnosis or member id)
      - Negative/invalid billed amounts
      - Unmatched provider IDs (referential integrity failure)
      - Inconsistent dates (e.g., service end date before service start date)
    """
    rng = np.random.default_rng(seed)

    # 1. Providers (50 fictional providers)
    n_providers = 50
    specialties = ['Family Medicine', 'Internal Medicine', 'Cardiology', 'Orthopedics', 'Pediatrics', 'Oncology', 'Emergency Medicine']
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'OH', 'PA']
    
    provider_ids = [f"PRV-{1000 + i}" for i in range(n_providers)]
    provider_names = [f"Dr. Provider {chr(65 + (i % 26))}{i // 26 + 1}" for i in range(n_providers)]
    prov_specialty = rng.choice(specialties, size=n_providers)
    prov_state = rng.choice(states, size=n_providers)
    prov_npi = [f"1{rng.integers(100000000, 999999999)}" for _ in range(n_providers)]

    providers_df = pd.DataFrame({
        'provider_id': provider_ids,
        'npi': prov_npi,
        'provider_name': provider_names,
        'specialty': prov_specialty,
        'state': prov_state,
        'status': rng.choice(['Active', 'Active', 'Active', 'Inactive'], size=n_providers)
    })

    # Add 1 duplicate provider row with different specialty to test provider DQ check
    # But keep provider_id strictly unique in primary master list, we can test duplicate NPI or flag
    # Let's keep provider table clean except for 1 missing NPI
    providers_df.loc[3, 'npi'] = None

    # 2. Members / Enrollment (200 fictional members)
    n_members = 200
    member_ids = [f"MEM-{10000 + i}" for i in range(n_members)]
    genders = rng.choice(['M', 'F', 'U'], size=n_members, p=[0.48, 0.48, 0.04])
    plans = rng.choice(['HMO', 'PPO', 'EPO', 'POS'], size=n_members)
    base_date = datetime(2023, 1, 1)

    dob_list = []
    enroll_start_list = []
    enroll_end_list = []

    for _ in range(n_members):
        age_days = rng.integers(365 * 18, 365 * 80)
        dob = base_date - timedelta(days=int(age_days))
        dob_list.append(dob.strftime('%Y-%m-%d'))
        
        start_offset = rng.integers(0, 365)
        start_dt = base_date + timedelta(days=int(start_offset))
        enroll_start_list.append(start_dt.strftime('%Y-%m-%d'))
        
        # 15% have enrollment end date (terminated)
        if rng.random() < 0.15:
            end_dt = start_dt + timedelta(days=int(rng.integers(60, 365)))
            enroll_end_list.append(end_dt.strftime('%Y-%m-%d'))
        else:
            enroll_end_list.append(None)

    members_df = pd.DataFrame({
        'member_id': member_ids,
        'gender': genders,
        'date_of_birth': dob_list,
        'plan_type': plans,
        'enrollment_start': enroll_start_list,
        'enrollment_end': enroll_end_list,
        'state': rng.choice(states, size=n_members),
        'risk_score': np.round(rng.uniform(0.5, 3.8, size=n_members), 2)
    })

    # Controlled DQ anomalies in members:
    # 1 null member_id (which should be flagged by DQ check)
    # We will keep master member_id intact but add 2 edge cases:
    members_df.loc[10, 'gender'] = 'INVALID_GENDER' # Invalid categorical value
    members_df.loc[25, 'risk_score'] = -1.5 # Negative risk score anomaly

    # 3. Claims (1,000 fictional claims)
    n_claims = 1000
    claim_ids = [f"CLM-{100000 + i}" for i in range(n_claims)]
    
    # Assign members and providers
    assigned_members = rng.choice(member_ids, size=n_claims)
    assigned_providers = rng.choice(provider_ids, size=n_claims)

    claim_types = rng.choice(['Professional', 'Institutional', 'Pharmacy'], size=n_claims, p=[0.6, 0.3, 0.1])
    diagnosis_codes = ['I10', 'E11.9', 'J45.909', 'M54.5', 'Z00.00', 'F41.1', 'K21.9', 'R05']
    procedure_codes = ['99213', '99214', '99283', '80053', '93000', '71045', '36415']
    claim_statuses = rng.choice(['Paid', 'Denied', 'In Process', 'Pending'], size=n_claims, p=[0.75, 0.15, 0.05, 0.05])
    denial_reasons = ['Prior Authorization Required', 'Duplicate Claim', 'Service Not Covered', 'Invalid Diagnosis', 'Timely Filing Exceeded']

    service_starts = []
    service_ends = []
    billed_amounts = []
    paid_amounts = []
    assigned_diagnoses = []
    assigned_procedures = []
    reasons = []

    for i in range(n_claims):
        claim_date = base_date + timedelta(days=int(rng.integers(0, 500)))
        service_starts.append(claim_date.strftime('%Y-%m-%d'))
        
        # In general, end date is 0-3 days after start
        end_date = claim_date + timedelta(days=int(rng.integers(0, 4)))
        service_ends.append(end_date.strftime('%Y-%m-%d'))

        # Billed & Paid
        billed = float(np.round(rng.lognormal(mean=5.5, sigma=1.0), 2))
        status = claim_statuses[i]
        if status == 'Paid':
            # Paid amount typically 40-90% of billed
            paid = float(np.round(billed * rng.uniform(0.4, 0.9), 2))
            reasons.append(None)
        elif status == 'Denied':
            paid = 0.0
            reasons.append(rng.choice(denial_reasons))
        else:
            paid = 0.0
            reasons.append(None)

        billed_amounts.append(billed)
        paid_amounts.append(paid)
        assigned_diagnoses.append(rng.choice(diagnosis_codes))
        assigned_procedures.append(rng.choice(procedure_codes))

    claims_df = pd.DataFrame({
        'claim_id': claim_ids,
        'member_id': assigned_members,
        'provider_id': assigned_providers,
        'claim_type': claim_types,
        'service_start_date': service_starts,
        'service_end_date': service_ends,
        'diagnosis_code': assigned_diagnoses,
        'procedure_code': assigned_procedures,
        'billed_amount': billed_amounts,
        'paid_amount': paid_amounts,
        'claim_status': claim_statuses,
        'denial_reason': reasons
    })

    # Controlled Data-Quality Injections (Deterministic and documented):
    # 1. Duplicate claim_id (Row 5 duplicates Row 4's claim_id)
    claims_df.loc[5, 'claim_id'] = claims_df.loc[4, 'claim_id']

    # 2. Missing required field (Row 12 has missing member_id, Row 15 has missing diagnosis_code)
    claims_df.loc[12, 'member_id'] = None
    claims_df.loc[15, 'diagnosis_code'] = None

    # 3. Invalid financial constraint (Row 20 has negative billed amount)
    claims_df.loc[20, 'billed_amount'] = -150.00

    # 4. Paid > Billed anomaly (Row 30 has paid_amount > billed_amount)
    claims_df.loc[30, 'billed_amount'] = 100.00
    claims_df.loc[30, 'paid_amount'] = 350.00

    # 5. Invalid date chronology (Row 40 has service_end_date before service_start_date)
    claims_df.loc[40, 'service_start_date'] = '2023-08-15'
    claims_df.loc[40, 'service_end_date'] = '2023-08-10'

    # 6. Unmatched provider ID (Referential integrity failure: Provider does not exist in providers_df)
    claims_df.loc[50, 'provider_id'] = 'PRV-9999_ORPHAN'
    claims_df.loc[51, 'provider_id'] = 'PRV-9998_ORPHAN'

    # 7. Disallowed categorical value
    claims_df.loc[60, 'claim_status'] = 'UNKNOWN_STATUS'

    return members_df, providers_df, claims_df
