"""Lightweight PII redaction for banking data before it hits logs, audit trails, or prompts.

This is a defense-in-depth measure, not a substitute for a proper DLP pipeline.
Patterns are intentionally conservative (favor false positives over leaking real data).
"""

import re

_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4}\b"
    ),
    "account_number": re.compile(
        r"\b(?:account|acct|iban)[\s:#-]*([A-Z0-9]{6,34})\b", re.IGNORECASE
    ),
}


def redact(text: str) -> str:
    """Replace detected PII substrings with a `[REDACTED_<TYPE>]` placeholder."""
    if not text:
        return text

    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PATTERNS.values())
