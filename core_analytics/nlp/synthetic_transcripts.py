"""
Synthetic Healthcare Call Transcripts Dataset Generator
Generates realistic, strictly fictional customer service and claims call transcripts
with reference annotations for extraction & human-review auditing.
No real patient, member, or provider information is used.
"""

from typing import List, Dict, Any

def get_synthetic_call_transcripts() -> List[Dict[str, Any]]:
    """
    Returns a deterministic list of fictional call transcript records with gold-standard
    reference annotations across:
      - transcript_id
      - call_text
      - reference_intent: 'Claim Status', 'Denial Reason', 'Payment Inquiry', 'Appeal Request', 'Prior Authorization', 'Ambiguous'
      - reference_claim_id: string or None
      - reference_denial_reason: string or None
      - reference_action: 'Check Status', 'Explain Denial', 'Process Appeal', 'Submit Auth', 'Escalate Payment', 'None'
      - reference_urgency: 'High', 'Medium', 'Low'
      - reference_requires_followup: True/False
    """
    return [
        {
            "transcript_id": "CALL-1001",
            "call_text": "Representative: Thanks for calling member services. Caller: Hi, I am calling to check the status of my claim CLM-100234 submitted two weeks ago for my outpatient procedure. Has it been processed yet?",
            "reference_intent": "Claim Status",
            "reference_claim_id": "CLM-100234",
            "reference_denial_reason": None,
            "reference_action": "Check Status",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1002",
            "call_text": "Representative: Support line, how can I help? Caller: Yes, I received an explanation of benefits stating claim CLM-100582 was rejected because Prior Authorization Required. Can you explain why prior authorization was needed for this urgent MRI?",
            "reference_intent": "Denial Reason",
            "reference_claim_id": "CLM-100582",
            "reference_denial_reason": "Prior Authorization Required",
            "reference_action": "Explain Denial",
            "reference_urgency": "High",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1003",
            "call_text": "Caller: Hello, I need to file an appeal immediately for claim CLM-100891. It was denied for Timely Filing Exceeded, but our clinic submitted it within 30 days. We have the fax confirmation receipt ready.",
            "reference_intent": "Appeal Request",
            "reference_claim_id": "CLM-100891",
            "reference_denial_reason": "Timely Filing Exceeded",
            "reference_action": "Process Appeal",
            "reference_urgency": "High",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1004",
            "call_text": "Caller: Good morning, I am calling regarding my doctor bill. My provider said claim CLM-100412 was approved for $450, but they only received a payment check of $120. Why is there a difference?",
            "reference_intent": "Payment Inquiry",
            "reference_claim_id": "CLM-100412",
            "reference_denial_reason": None,
            "reference_action": "Escalate Payment",
            "reference_urgency": "Medium",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1005",
            "call_text": "Representative: Prior auth department. Caller: Hi, I'm calling from Dr. Smith's office. We need to submit a prior authorization request for a scheduled knee replacement next Tuesday. Can you confirm the submission fax number?",
            "reference_intent": "Prior Authorization",
            "reference_claim_id": None,
            "reference_denial_reason": None,
            "reference_action": "Submit Auth",
            "reference_urgency": "Medium",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1006",
            "call_text": "Caller: I got a denial letter for claim CLM-100673 stating Service Not Covered for physical therapy sessions. What does that mean?",
            "reference_intent": "Denial Reason",
            "reference_claim_id": "CLM-100673",
            "reference_denial_reason": "Service Not Covered",
            "reference_action": "Explain Denial",
            "reference_urgency": "Medium",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1007",
            "call_text": "Caller: Hello? Can someone tell me about my insurance card? I lost it yesterday and I don't know what to do.",
            "reference_intent": "Ambiguous",
            "reference_claim_id": None,
            "reference_denial_reason": None,
            "reference_action": "None",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1008",
            "call_text": "Caller: This is urgent! My oncology infusion is scheduled for tomorrow morning at 8 AM and claim CLM-100990 is pending emergency authorization! Please expedite this immediately!",
            "reference_intent": "Prior Authorization",
            "reference_claim_id": "CLM-100990",
            "reference_denial_reason": None,
            "reference_action": "Submit Auth",
            "reference_urgency": "High",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1009",
            "call_text": "Caller: We want to submit an appeal for claim CLM-100115 which was marked as Duplicate Claim in error. These were two separate visits on two consecutive days.",
            "reference_intent": "Appeal Request",
            "reference_claim_id": "CLM-100115",
            "reference_denial_reason": "Duplicate Claim",
            "reference_action": "Process Appeal",
            "reference_urgency": "Medium",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1010",
            "call_text": "Caller: Hi, just calling to see if claim CLM-100344 has cleared and when the check was issued.",
            "reference_intent": "Claim Status",
            "reference_claim_id": "CLM-100344",
            "reference_denial_reason": None,
            "reference_action": "Check Status",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1011",
            "call_text": "Caller: Yeah, uh, someone called me from this number earlier today. I have no idea who called or why.",
            "reference_intent": "Ambiguous",
            "reference_claim_id": None,
            "reference_denial_reason": None,
            "reference_action": "None",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1012",
            "call_text": "Caller: I am checking on payment status for claim CLM-100789. The provider says they haven't received electronic funds transfer yet.",
            "reference_intent": "Payment Inquiry",
            "reference_claim_id": "CLM-100789",
            "reference_denial_reason": None,
            "reference_action": "Escalate Payment",
            "reference_urgency": "Low",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1013",
            "call_text": "Caller: Our claim CLM-100456 was denied citing Invalid Diagnosis code. We have updated the coding to ICD-10 M54.5 and wish to resubmit.",
            "reference_intent": "Denial Reason",
            "reference_claim_id": "CLM-100456",
            "reference_denial_reason": "Invalid Diagnosis",
            "reference_action": "Process Appeal",
            "reference_urgency": "Medium",
            "reference_requires_followup": True
        },
        {
            "transcript_id": "CALL-1014",
            "call_text": "Caller: Can you check if prior authorization is required for CPT code 99283 under my plan?",
            "reference_intent": "Prior Authorization",
            "reference_claim_id": None,
            "reference_denial_reason": None,
            "reference_action": "Submit Auth",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        },
        {
            "transcript_id": "CALL-1015",
            "call_text": "Caller: Static... crackle... [unintelligible audio] ...hello? Can you hear me?",
            "reference_intent": "Ambiguous",
            "reference_claim_id": None,
            "reference_denial_reason": None,
            "reference_action": "None",
            "reference_urgency": "Low",
            "reference_requires_followup": False
        }
    ]
