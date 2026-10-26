"""
Synthetic Healthcare Claims Workflow Event Log Generator
Generates reproducible, strictly fictional operational event logs representing
the lifecycle of adjudicated healthcare claims for process mining.
Includes realistic workflow variants, rework loops, manual overrides,
bottlenecks, missing timestamps, and out-of-order logs.
All data is strictly fictional and non-proprietary.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

def generate_synthetic_claims_event_log(n_cases: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Generates an event-log DataFrame with columns:
      - case_id: Claim case tracking identifier (e.g. CASE-10001)
      - activity: Process activity name
      - event_timestamp: ISO-formatted timestamp (YYYY-MM-DD HH:MM:SS)
      - resource: Department or responsible worker (e.g., Intake_Bot, Clinical_Reviewer, Adjudication_Engine)
      - outcome_status: Status outcome at event completion
      - is_override: Boolean flag indicating if manual override occurred
      - override_reason: Optional reason string if override occurred, else None

    Controlled Workflow Patterns & Variants:
      1. Standard Straight-Through Path (~40%):
         Submitted -> Automated Intake -> Initial Review -> Approved -> Payment Processed
      2. Rework / Resubmission Path (~25%):
         Submitted -> Automated Intake -> Initial Review -> Denied -> Appeal Received -> Re-review -> Approved -> Payment Processed
      3. Manual Clinical Override Path (~15%):
         Submitted -> Automated Intake -> Manual Review -> Override -> Approved -> Payment Processed
      4. Missing Information / Pend Path (~15%):
         Submitted -> Automated Intake -> Information Requested -> Pended -> Information Received -> Re-review -> Approved -> Payment Processed
      5. Controlled Data Quality Anomalies (~5%):
         - Single-event cases
         - Out-of-order timestamps
         - Null timestamps
         - Duplicate consecutive events
    """
    rng = np.random.default_rng(seed)
    base_time = datetime(2023, 3, 1, 8, 0, 0)
    all_events = []

    for i in range(n_cases):
        case_id = f"CASE-{10000 + i}"
        case_start = base_time + timedelta(
            days=int(rng.integers(0, 90)),
            hours=int(rng.integers(0, 10)),
            minutes=int(rng.integers(0, 59))
        )
        t = case_start

        # Determine path variant
        rand_val = rng.random()

        if i == 0:
            # Explicit test case: Single-event case
            all_events.append({
                'case_id': case_id,
                'activity': 'Claim Submitted',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Portal_Intake',
                'outcome_status': 'Received',
                'is_override': False,
                'override_reason': None
            })
            continue

        if i == 1:
            # Explicit test case: Out-of-order timestamp injection
            # Activity 1: 08:00, Activity 2: 07:30 (scrambled)
            t1 = t
            t2 = t - timedelta(minutes=30)
            all_events.append({
                'case_id': case_id,
                'activity': 'Claim Submitted',
                'event_timestamp': t1.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Portal_Intake',
                'outcome_status': 'Received',
                'is_override': False,
                'override_reason': None
            })
            all_events.append({
                'case_id': case_id,
                'activity': 'Automated Intake',
                'event_timestamp': t2.strftime('%Y-%m-%d %H:%M:%S'), # Out-of-order!
                'resource': 'Intake_Bot_01',
                'outcome_status': 'Validated',
                'is_override': False,
                'override_reason': None
            })
            continue

        if i == 2:
            # Explicit test case: Missing/Null timestamp
            all_events.append({
                'case_id': case_id,
                'activity': 'Claim Submitted',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Portal_Intake',
                'outcome_status': 'Received',
                'is_override': False,
                'override_reason': None
            })
            all_events.append({
                'case_id': case_id,
                'activity': 'Automated Intake',
                'event_timestamp': None, # Null timestamp!
                'resource': 'Intake_Bot_01',
                'outcome_status': 'Validated',
                'is_override': False,
                'override_reason': None
            })
            continue

        if i == 3:
            # Explicit test case: Duplicate event
            all_events.append({
                'case_id': case_id,
                'activity': 'Claim Submitted',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Portal_Intake',
                'outcome_status': 'Received',
                'is_override': False,
                'override_reason': None
            })
            all_events.append({
                'case_id': case_id,
                'activity': 'Claim Submitted', # Exact duplicate activity and timestamp
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Portal_Intake',
                'outcome_status': 'Received',
                'is_override': False,
                'override_reason': None
            })
            all_events.append({
                'case_id': case_id,
                'activity': 'Initial Review',
                'event_timestamp': (t + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Adjudication_Engine',
                'outcome_status': 'Approved',
                'is_override': False,
                'override_reason': None
            })
            continue

        # Standard variants:
        # Step 1: Claim Submitted
        all_events.append({
            'case_id': case_id,
            'activity': 'Claim Submitted',
            'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
            'resource': 'EDI_Gateway' if rng.random() > 0.3 else 'Web_Portal',
            'outcome_status': 'Received',
            'is_override': False,
            'override_reason': None
        })

        # Step 2: Automated Intake (0.5 to 2 hours later)
        t += timedelta(minutes=int(rng.integers(30, 120)))
        all_events.append({
            'case_id': case_id,
            'activity': 'Automated Intake',
            'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
            'resource': f'Intake_Bot_0{rng.integers(1, 4)}',
            'outcome_status': 'Validated',
            'is_override': False,
            'override_reason': None
        })

        if rand_val < 0.40:
            # Variant 1: Straight-through processing
            t += timedelta(hours=int(rng.integers(2, 8)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Initial Review',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Adjudication_Engine',
                'outcome_status': 'Approved',
                'is_override': False,
                'override_reason': None
            })
            t += timedelta(hours=int(rng.integers(12, 48)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Payment Processed',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Treasury_EFT_System',
                'outcome_status': 'Settled',
                'is_override': False,
                'override_reason': None
            })

        elif rand_val < 0.65:
            # Variant 2: Denial & Appeal Rework Loop
            t += timedelta(hours=int(rng.integers(4, 12)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Initial Review',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Adjudication_Engine',
                'outcome_status': 'Denied',
                'is_override': False,
                'override_reason': None
            })
            # Appeal received 2-5 days later
            t += timedelta(days=int(rng.integers(2, 6)), hours=int(rng.integers(1, 5)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Appeal Received',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Appeals_Intake',
                'outcome_status': 'Resubmitted',
                'is_override': False,
                'override_reason': None
            })
            # Re-review (REWORK)
            t += timedelta(days=int(rng.integers(1, 3)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Initial Review', # Rework loop: Initial Review repeated
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Clinical_Appeals_Nurse',
                'outcome_status': 'Approved',
                'is_override': False,
                'override_reason': None
            })
            t += timedelta(hours=int(rng.integers(24, 72)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Payment Processed',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Treasury_EFT_System',
                'outcome_status': 'Settled',
                'is_override': False,
                'override_reason': None
            })

        elif rand_val < 0.80:
            # Variant 3: Manual Clinical Override
            t += timedelta(hours=int(rng.integers(6, 24)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Manual Clinical Review',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': f'Medical_Director_0{rng.integers(1, 3)}',
                'outcome_status': 'Pending_Override',
                'is_override': False,
                'override_reason': None
            })
            t += timedelta(hours=int(rng.integers(2, 8)))
            reason = rng.choice(['Medical Necessity Exception', 'Peer-to-Peer Concordance', 'Emergency Service Carveout', 'Prior Auth Extenuating Circumstance'])
            all_events.append({
                'case_id': case_id,
                'activity': 'Manager Override',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Medical_Director_Lead',
                'outcome_status': 'Approved',
                'is_override': True,
                'override_reason': reason
            })
            t += timedelta(hours=int(rng.integers(12, 36)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Payment Processed',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Treasury_EFT_System',
                'outcome_status': 'Settled',
                'is_override': False,
                'override_reason': None
            })

        else:
            # Variant 4: Missing Information & Pending Bottleneck
            t += timedelta(hours=int(rng.integers(4, 18)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Information Requested',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Customer_Service_Ops',
                'outcome_status': 'Pending_Provider_Response',
                'is_override': False,
                'override_reason': None
            })
            # Long delay bottleneck (5 to 15 days)
            t += timedelta(days=int(rng.integers(5, 15)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Information Received',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Provider_Portal',
                'outcome_status': 'Ready_For_Adjudication',
                'is_override': False,
                'override_reason': None
            })
            t += timedelta(hours=int(rng.integers(12, 48)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Initial Review',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Adjudication_Engine',
                'outcome_status': 'Approved',
                'is_override': False,
                'override_reason': None
            })
            t += timedelta(hours=int(rng.integers(12, 48)))
            all_events.append({
                'case_id': case_id,
                'activity': 'Payment Processed',
                'event_timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'resource': 'Treasury_EFT_System',
                'outcome_status': 'Settled',
                'is_override': False,
                'override_reason': None
            })

    return pd.DataFrame(all_events)
