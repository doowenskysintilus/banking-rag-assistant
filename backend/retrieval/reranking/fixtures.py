"""Labeled banking query/candidate examples for the reranking benchmark and unit
tests — each query has one or two genuinely relevant candidates mixed in with
plausible-sounding distractors (the kind of thing a bi-encoder or BM25 retriever
would surface as a false positive: same domain, wrong specifics).
"""

from __future__ import annotations

from backend.retrieval.reranking.evaluation import EvaluationExample

EVAL_EXAMPLES: list[EvaluationExample] = [
    EvaluationExample(
        query="What is the overdraft fee for premium checking accounts?",
        candidate_ids=["c1", "c2", "c3", "c4", "c5", "c6"],
        candidate_texts=[
            "Premium checking account holders are not charged an overdraft fee for the "
            "first occurrence in a rolling twelve-month period; subsequent occurrences "
            "are charged at $20 per item.",
            "Standard checking accounts incur a $35 overdraft fee per transaction, capped "
            "at three fees per business day.",
            "Wire transfers to international accounts are subject to a flat $45 fee plus "
            "any correspondent bank charges.",
            "Mortgage rate locks are valid for 60 days from the application date and may "
            "be extended for an additional fee.",
            "The bank's know-your-customer procedure requires government-issued photo "
            "identification and proof of address for all new account openings.",
            "Certificates of deposit under $10,000 earn a fixed rate set at account opening "
            "and cannot be renegotiated before maturity.",
        ],
        relevant_ids={"c1"},
    ),
    EvaluationExample(
        query="How long are wire transfer rate locks valid for mortgages?",
        candidate_ids=["m1", "m2", "m3", "m4", "m5"],
        candidate_texts=[
            "Mortgage rate locks are valid for 60 days from the application date and may "
            "be extended for an additional fee of 0.25 points per 15-day extension.",
            "Overdraft protection transfers from a linked savings account are processed "
            "same-day with no fee for the first three transfers each statement cycle.",
            "Adjustable-rate mortgages reset annually after the initial five-year fixed "
            "period, based on the one-year Treasury index plus a 2.5% margin.",
            "Domestic wire transfers initiated before 3pm local time are typically settled "
            "the same business day.",
            "Home equity lines of credit have a 10-year draw period followed by a 20-year "
            "repayment period.",
        ],
        relevant_ids={"m1"},
    ),
    EvaluationExample(
        query="What identification is required to open a new business account?",
        candidate_ids=["k1", "k2", "k3", "k4", "k5", "k6"],
        candidate_texts=[
            "New business account applications require the entity's formation documents, "
            "an EIN confirmation letter, and government-issued photo identification for "
            "each authorized signer.",
            "Consumer checking accounts require a single government-issued photo ID and a "
            "Social Security number or ITIN.",
            "The bank's fraud monitoring system flags transactions exceeding $10,000 for "
            "manual review under the Bank Secrecy Act.",
            "Loan applications for commercial real estate require two years of business "
            "tax returns and a current rent roll.",
            "Safe deposit box rental requires proof of account relationship and a signed "
            "access agreement.",
            "International wire transfers above $3,000 require additional beneficiary "
            "verification under OFAC screening procedures.",
        ],
        relevant_ids={"k1"},
    ),
    EvaluationExample(
        query="What is the minimum capital ratio required under the bank's prudential regulation?",
        candidate_ids=["r1", "r2", "r3", "r4", "r5"],
        candidate_texts=[
            "Institutions shall at all times maintain a Common Equity Tier 1 capital ratio "
            "of at least 4.5 per cent of total risk-weighted exposure amounts.",
            "Institutions shall hold a capital conservation buffer of Common Equity Tier 1 "
            "capital equal to 2.5 per cent of total risk-weighted exposure amounts, in "
            "addition to the minimum ratio.",
            "The liquidity coverage ratio requires liquid assets sufficient to cover net "
            "outflows over a 30 calendar day stressed period.",
            "An institution's exposure to a single client shall not exceed 25 per cent of "
            "its eligible capital.",
            "Competent authorities shall review institutions' governance arrangements at "
            "least annually.",
        ],
        # Two genuinely relevant passages here (both about the capital ratio
        # requirement) — a good check that reranking recovers *all* relevant
        # results, not just the single most obvious one.
        relevant_ids={"r1", "r2"},
    ),
    EvaluationExample(
        query="How is net interest income trending and what is driving the change?",
        candidate_ids=["f1", "f2", "f3", "f4", "f5"],
        candidate_texts=[
            "Net interest income increased to $482.3 million for the fiscal year, "
            "compared to $451.7 million in the prior year, driven primarily by higher "
            "average loan balances and a wider net interest margin.",
            "Non-interest income decreased slightly to $128.4 million, reflecting lower "
            "fee income from mortgage banking activity.",
            "The allowance for credit losses on the commercial real estate portfolio was "
            "increased by $14.2 million during the fourth quarter.",
            "The Company completed the acquisition of a regional trust company for "
            "approximately $85 million, funded through cash and subordinated debt.",
            "The Company operates through three reportable segments: Consumer Banking, "
            "Commercial Banking, and Wealth Management.",
        ],
        relevant_ids={"f1"},
    ),
]
