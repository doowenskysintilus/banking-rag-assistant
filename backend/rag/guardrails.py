"""Rule-based, post-hoc anti-hallucination checks: verifies the model's actual
answer against what `backend.llm.prompts` asked it to do, rather than trusting a
well-written prompt to be self-enforcing. Deterministic and cheap (no extra LLM
call, no network) — a defense-in-depth layer alongside the prompt, not a
replacement for it.

Every check here is a *heuristic* over natural-language text. False positives (an
unnecessary warning surfaced to the user/analyst) are an acceptable cost; the goal
is to flag likely hallucinations for human attention, not to silently block them.
`verify_answer` never raises — a bug in a guardrail must never break the answer
it's supposed to be protecting.
"""

from __future__ import annotations

import re

from backend.core.logging import get_logger
from backend.llm.prompts import HEDGE_PHRASES
from backend.rag.confidence import ConfidenceAssessment
from backend.rag.context_builder import ContextSource

logger = get_logger(__name__)

_SOURCE_MARKER_RE = re.compile(r"\[\s*Source\s+(\d+)\s*\]", re.IGNORECASE)
_CURRENCY_RE = re.compile(
    r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP)\b", re.IGNORECASE
)
_PERCENTAGE_RE = re.compile(r"\d+(?:\.\d+)?\s?%")

# Phrasing that asserts a legal conclusion rather than quoting a source — see
# system_prompt.py section 5. Deliberately narrow (vs. e.g. bare "you must") to
# avoid flagging ordinary instructions ("you must complete form X") as overreach.
_LEGAL_OVERREACH_PATTERNS = (
    r"\byou(?:'re| are) legally\b",
    r"\bit is illegal\b",
    r"\bthis (?:is|would be) (?:a violation|illegal|unlawful)\b",
    r"\byou have (?:the|a) legal right\b",
    r"\byou (?:can|should) sue\b",
    r"\blegally entitled\b",
    r"\bagainst the law\b",
    r"\byou are (?:required|obligated) by law\b",
)
_LEGAL_OVERREACH_RE = re.compile("|".join(_LEGAL_OVERREACH_PATTERNS), re.IGNORECASE)


def _normalize_number(text: str) -> str:
    """Best-effort only — "4.5%" and "4.50%" won't match despite being the same
    value. Good enough to catch a fabricated figure with no basis in context at
    all, which is the failure mode this check targets.
    """
    return re.sub(r"[\s,]", "", text).lower()


def _check_citation_validity(answer: str, num_sources: int) -> list[str]:
    warnings = []
    seen: set[int] = set()
    for match in _SOURCE_MARKER_RE.finditer(answer):
        index = int(match.group(1))
        if index in seen or 1 <= index <= num_sources:
            continue
        seen.add(index)
        warnings.append(
            f"Answer cites [Source {index}], which does not exist among the {num_sources} "
            "retrieved source(s) — possible fabricated citation."
        )
    return warnings


def _check_numeric_grounding(answer: str, context_text: str) -> list[str]:
    answer_numbers = {
        _normalize_number(m.group(0)) for m in _CURRENCY_RE.finditer(answer)
    }
    answer_numbers |= {
        _normalize_number(m.group(0)) for m in _PERCENTAGE_RE.finditer(answer)
    }
    if not answer_numbers:
        return []

    context_normalized = _normalize_number(context_text)
    unverified = sorted(n for n in answer_numbers if n not in context_normalized)
    if not unverified:
        return []

    return [
        f"Answer contains {len(unverified)} numeric value(s) not found verbatim in the retrieved "
        f"context (e.g. '{unverified[0]}') — verify this figure isn't fabricated, miscopied, or "
        "a calculation the source text didn't actually state."
    ]


def _check_confidence_hedging(
    answer: str, confidence: ConfidenceAssessment
) -> list[str]:
    if confidence.level != "low":
        return []
    lowered = answer.lower()
    if any(phrase in lowered for phrase in HEDGE_PHRASES):
        return []
    return [
        "Retrieval confidence is low, but the answer contains no hedging or "
        "'insufficient information' language — check it isn't overstating what the sources support."
    ]


def _check_legal_overreach(answer: str) -> list[str]:
    if not _LEGAL_OVERREACH_RE.search(answer):
        return []
    return [
        "Answer contains language asserting a legal conclusion — confirm this is a direct "
        "quote/paraphrase of the source text, not the model's own legal interpretation."
    ]


def verify_answer(
    answer: str,
    context_text: str,
    sources: list[ContextSource],
    confidence: ConfidenceAssessment,
) -> list[str]:
    """Returns human-readable warnings — an empty list if nothing looked off."""
    try:
        return [
            *_check_citation_validity(answer, len(sources)),
            *_check_numeric_grounding(answer, context_text),
            *_check_confidence_hedging(answer, confidence),
            *_check_legal_overreach(answer),
        ]
    except Exception:  # noqa: BLE001 - a guardrail bug must never break the answer
        logger.exception("guardrail_check_failed")
        return []
