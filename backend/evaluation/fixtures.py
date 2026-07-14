"""A small labeled "golden set" for the evaluation framework itself — used by the
benchmark/tests to exercise metric providers without needing a live database or
LLM calls to *generate* an answer first (the samples below already carry a
question, the context it should have been answered from, a plausible RAG answer,
and a reference ground truth).

Deliberately includes both well-behaved and defective answers (a fabricated
number, an unhedged answer to a question the context doesn't cover) so the
metrics have something real to catch — an evaluation framework that only ever
sees perfect answers hasn't actually been exercised.
"""

from __future__ import annotations

from backend.evaluation.types import EvalSample

GOLDEN_SAMPLES: list[EvalSample] = [
    EvalSample(
        sample_id="overdraft-fee-grounded",
        question="What is the overdraft fee for premium checking accounts?",
        answer=(
            "For Premium Checking accounts, the first overdraft in a rolling 12-month "
            "period is fee-free; each additional occurrence is charged $20 per item "
            "[Source 1]."
        ),
        contexts=[
            "Premium checking account holders are not charged an overdraft fee for the "
            "first occurrence in a rolling twelve-month period; subsequent occurrences "
            "are charged at $20 per item."
        ],
        ground_truth=(
            "Premium checking has no overdraft fee for the first occurrence in a "
            "12-month period; each additional occurrence costs $20."
        ),
    ),
    EvalSample(
        sample_id="wire-transfer-fabricated-number",
        question="What is the fee for an international wire transfer?",
        answer=(
            "International wire transfers cost $60 plus a 1.5% currency conversion fee "
            "[Source 1]."
        ),
        contexts=[
            "Wire transfers to international accounts are subject to a flat $45 fee plus "
            "any correspondent bank charges."
        ],
        ground_truth="International wire transfers cost a flat $45 fee plus correspondent bank charges.",
    ),
    EvalSample(
        sample_id="business-account-insufficient-info",
        question="What is the overdraft fee for a business checking account?",
        answer=(
            "I do not have sufficient information in the provided documents to answer "
            "this question, specifically: business checking account overdraft fees. "
            "The retrieved context only covers standard and premium checking accounts."
        ),
        contexts=[
            "Standard checking accounts incur a $35 overdraft fee per transaction, capped "
            "at three fees per business day.",
            "Premium checking account holders are not charged an overdraft fee for the "
            "first occurrence in a rolling twelve-month period; subsequent occurrences "
            "are charged at $20 per item.",
        ],
        ground_truth="The provided documents do not state a business checking overdraft fee.",
    ),
    EvalSample(
        sample_id="mortgage-rate-lock-grounded",
        question="How long are mortgage rate locks valid for?",
        answer=(
            "Mortgage rate locks are valid for 60 days from the application date and may "
            "be extended for an additional fee of 0.25 points per 15-day extension "
            "[Source 1]."
        ),
        contexts=[
            "Mortgage rate locks are valid for 60 days from the application date and may "
            "be extended for an additional fee of 0.25 points per 15-day extension."
        ],
        ground_truth="Rate locks last 60 days from application and can be extended at 0.25 points per 15 days.",
    ),
    EvalSample(
        sample_id="kyc-requirements-partial-overreach",
        question="Am I legally required to provide ID to open a business account?",
        answer=(
            "You are legally required to do so. New business account applications "
            "require the entity's formation documents, an EIN confirmation letter, and "
            "government-issued photo identification for each authorized signer "
            "[Source 1]."
        ),
        contexts=[
            "New business account applications require the entity's formation documents, "
            "an EIN confirmation letter, and government-issued photo identification for "
            "each authorized signer."
        ],
        ground_truth=(
            "The bank's policy requires formation documents, an EIN confirmation letter, "
            "and photo ID for authorized signers; whether this is a strict legal "
            "requirement should be confirmed with compliance."
        ),
    ),
]
