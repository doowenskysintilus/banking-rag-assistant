"""Synthetic sample documents used by the chunking benchmark and by unit/integration
tests. Representative of the three document families this chunker targets: banking
contracts, regulations, and financial reports. Content is illustrative, not real
legal/financial text.
"""

from __future__ import annotations

LOAN_AGREEMENT = """LOAN AGREEMENT

This Loan Agreement (the "Agreement") is entered into between the Bank and the Borrower.

Article 1 - Definitions

Article 1.1 - General Terms

For the purposes of this Agreement, the following terms shall have the meanings set out below. "Borrower" means the individual or entity receiving the credit facility described herein. "Bank" means the lending institution extending the facility, together with its successors and permitted assigns. "Business Day" means any day other than a Saturday, Sunday, or public holiday on which banks are open for business in the relevant jurisdiction.

Article 1.2 - Interpretation

Headings are for convenience only and do not affect interpretation. References to Articles and Clauses are references to Articles and Clauses of this Agreement, unless stated otherwise. Words importing the singular include the plural and vice versa, e.g. "the Borrower" may refer to multiple co-borrowers where applicable.

Article 2 - The Facility

Article 2.1 - Amount and Purpose

The Bank agrees to make available to the Borrower a term loan facility in the principal amount set out in the Facility Letter. The facility shall be used exclusively for the purpose described in Schedule A and for no other purpose without the Bank's prior written consent.

Article 2.2 - Drawdown

The Borrower may request a drawdown of the facility, in whole or in part, by delivering a drawdown notice no later than three Business Days prior to the requested drawdown date. Each drawdown notice is irrevocable once given. The Bank shall have no obligation to fund a drawdown request if an Event of Default (as defined in Article 7) has occurred and is continuing.

Article 3 - Interest and Fees

Article 3.1 - Interest Rate

Interest accrues on the outstanding principal balance at a rate equal to the Base Rate plus the Margin, calculated on a daily basis and payable monthly in arrears. The applicable Margin is 2.5 per cent per annum for the first 12 months and 3.0 per cent per annum thereafter.

Article 3.2 - Default Interest

Where any amount due under this Agreement is not paid when due, interest accrues on the overdue amount at the Default Rate, being the Interest Rate plus 4 per cent per annum, from the due date until the date of actual payment, both before and after judgment.

Article 4 - Repayment and Prepayment

Article 4.1 - Scheduled Repayment

The Borrower shall repay the facility in the instalments and on the dates set out in the repayment schedule attached as Schedule B. Any instalment not paid on its due date remains immediately due and payable.

Article 4.2 - Voluntary Prepayment

The Borrower may prepay all or part of the outstanding facility on giving not less than five Business Days' prior written notice to the Bank, provided that any partial prepayment is applied against the remaining instalments in inverse order of maturity, unless the Bank agrees otherwise in writing.

Article 5 - Covenants

Article 5.1 - Financial Covenants

The Borrower shall maintain a debt service coverage ratio of not less than 1.25:1, tested quarterly by reference to the Borrower's most recent management accounts. The Borrower shall promptly notify the Bank in writing if it anticipates a breach of this ratio.

Article 5.2 - Information Covenants

The Borrower shall deliver to the Bank its audited annual financial statements within 120 days of each financial year end, and unaudited quarterly management accounts within 45 days of each quarter end.

Article 6 - Security

The Borrower's obligations under this Agreement are secured by the security interests described in the Security Documents, including a first-ranking charge over the Borrower's principal business assets. The Borrower shall not create or permit to subsist any other security interest over the charged assets without the Bank's prior written consent.

Article 7 - Events of Default

Article 7.1 - Non-Payment

It is an Event of Default if the Borrower fails to pay any amount due under this Agreement within five Business Days of its due date, unless the failure is caused solely by an administrative or technical error and payment is made within three Business Days of the Borrower becoming aware of it.

Article 7.2 - Cross-Default

It is an Event of Default if any other financial indebtedness of the Borrower in excess of the Threshold Amount becomes due and payable prior to its stated maturity as a result of a default, or if the Borrower fails to pay any such indebtedness when due.

Article 8 - Governing Law

This Agreement and any dispute arising out of or in connection with it shall be governed by and construed in accordance with the laws of the relevant jurisdiction, without regard to its conflict of laws principles.
"""

BANKING_REGULATION = """TITLE I - GENERAL PROVISIONS

Chapter 1 - Scope and Purpose

Section 1 - Scope of Application

This Regulation applies to all credit institutions authorized to accept deposits from the public and to all financial holding companies that control one or more such credit institutions. It establishes prudential requirements relating to capital, liquidity, and governance that institutions must satisfy on an ongoing basis.

Section 2 - Objectives

The objectives of this Regulation are to promote the safety and soundness of credit institutions, to protect depositors, and to contribute to the stability of the financial system as a whole. Competent authorities shall have regard to these objectives when exercising any discretion conferred on them by this Regulation.

Chapter 2 - Capital Requirements

Section 3 - Own Funds Requirements

Section 3.1 - Minimum Ratios

Institutions shall at all times maintain a Common Equity Tier 1 capital ratio of at least 4.5 per cent, a Tier 1 capital ratio of at least 6 per cent, and a Total Capital ratio of at least 8 per cent, each expressed as a percentage of total risk-weighted exposure amounts.

Section 3.2 - Capital Buffers

In addition to the minimum ratios in Section 3.1, institutions shall maintain a capital conservation buffer of Common Equity Tier 1 capital equal to 2.5 per cent of total risk-weighted exposure amounts. Competent authorities may require an institution-specific countercyclical buffer of up to 2.5 per cent depending on the credit cycle in the relevant jurisdiction.

Section 4 - Large Exposures

Section 4.1 - General Limit

An institution's exposure to a single client or group of connected clients shall not exceed 25 per cent of its eligible capital after taking into account the effect of credit risk mitigation. Institutions shall report any exposure exceeding 10 per cent of eligible capital to the competent authority without delay.

Section 4.2 - Exposures to Institutions

Where the client is another institution, or a group of connected clients includes one or more institutions, the limit in Section 4.1 shall not exceed the higher of 25 per cent of eligible capital or EUR 150 million, provided the sum of exposure values to all connected clients that are not institutions does not exceed 25 per cent of the institution's eligible capital.

Chapter 3 - Liquidity Requirements

Section 5 - Liquidity Coverage Requirement

Institutions shall hold liquid assets, the sum of the values of which covers the liquidity outflows less the liquidity inflows under stressed conditions over a period of 30 calendar days, such that the ratio of liquid assets to net liquidity outflows is not less than 100 per cent.

Section 6 - Net Stable Funding Requirement

Institutions shall ensure that their available stable funding, in relation to their required stable funding over a one-year horizon, results in a net stable funding ratio of at least 100 per cent, calculated in accordance with the methodology set out in Annex II.

Chapter 4 - Governance and Supervision

Section 7 - Internal Governance

Institutions shall have robust governance arrangements, including a clear organizational structure with well-defined, transparent, and consistent lines of responsibility, effective processes to identify, manage, and report the risks they are or might be exposed to, and adequate internal control mechanisms.

Section 8 - Supervisory Review

The competent authority shall review the arrangements, strategies, processes, and mechanisms implemented by institutions to comply with this Regulation, and shall evaluate the risks to which institutions are or might be exposed, at least annually.
"""

FINANCIAL_REPORT = """ANNUAL REPORT — FISCAL YEAR RESULTS

MANAGEMENT'S DISCUSSION AND ANALYSIS

Item 7. Management's Discussion and Analysis of Financial Condition

Net interest income increased to $482.3 million for the fiscal year, compared to $451.7 million in the prior year, driven primarily by higher average loan balances and a wider net interest margin. Non-interest income decreased slightly to $128.4 million, reflecting lower fee income from mortgage banking activity, partially offset by higher wealth management fees.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

The Company manages interest rate risk through a combination of on-balance-sheet strategies and derivative instruments, including interest rate swaps designated as cash flow hedges. As of year end, a hypothetical 100 basis point parallel increase in interest rates would decrease projected net interest income over the next twelve months by approximately 2.1 per cent, within the Company's board-approved risk tolerance.

RISK FACTORS

Credit risk on the commercial real estate portfolio remains elevated relative to historical averages, reflecting continued softness in office occupancy rates across the Company's primary markets. The allowance for credit losses on this segment was increased by $14.2 million during the fourth quarter in response to updated loss forecasts.

NOTES TO CONSOLIDATED FINANCIAL STATEMENTS

Note 1 — Summary of Significant Accounting Policies

The consolidated financial statements are prepared in accordance with generally accepted accounting principles. The preparation of financial statements requires management to make estimates and assumptions, e.g. for the allowance for credit losses, that affect reported amounts of assets, liabilities, revenues, and expenses.

Note 8 — Allowance for Credit Losses

The allowance for credit losses represents management's estimate of expected credit losses over the contractual life of the loan portfolio. The following table summarizes the changes in the allowance during the year: beginning balance of $312.6 million, provision for credit losses of $46.8 million, net charge-offs of $31.4 million, and an ending balance of $328.0 million.

Note 14 — Segment Reporting

The Company operates through three reportable segments: Consumer Banking, Commercial Banking, and Wealth Management. Segment results are evaluated based on pre-tax, pre-provision net revenue, consistent with how the chief operating decision maker assesses performance and allocates resources.

Note 22 — Subsequent Events

Subsequent to year end, the Company completed the acquisition of a regional trust company for total consideration of approximately $85 million, funded through a combination of cash on hand and the issuance of subordinated debt. Management does not expect the transaction to have a material effect on regulatory capital ratios.
"""

ALL_FIXTURES: dict[str, str] = {
    "loan_agreement": LOAN_AGREEMENT,
    "banking_regulation": BANKING_REGULATION,
    "financial_report": FINANCIAL_REPORT,
}
