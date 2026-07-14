"""Per-query prompt templates layered on top of `SYSTEM_PROMPT` — small, targeted
reinforcements added to specific user turns rather than bloating the (cached,
but still finite-context) system prompt with instructions only relevant some of
the time.
"""

from __future__ import annotations

import re

_NO_CONTEXT_TEXT = "No relevant internal documents were found for this question."

_LOW_CONFIDENCE_ADDENDUM = (
    "Note: the retrieved context above is only weakly related to this question "
    "(low retrieval confidence). Apply rule 4 and rule 8 strictly — if the context "
    "doesn't clearly answer this, say so; do not stretch a tangential match into a "
    "confident-sounding answer."
)

_LEGAL_QUESTION_ADDENDUM = (
    "Note: this question appears to call for a legal, regulatory, or compliance "
    "judgment. Apply rule 5 strictly — quote what the provided text says, do not "
    "state your own legal conclusion, and flag that interpretation/application to "
    "this specific situation should go through compliance or legal counsel."
)

# Deliberately conservative and English/French-bilingual (the source documents this
# assistant serves — banking policies, EU-style regulations — commonly appear in
# either) — false positives (an extra caution note on a question that wasn't really
# legal) are cheap; false negatives (missing one) are the failure mode rule 5 exists
# to prevent.
_LEGAL_KEYWORDS = (
    r"\blegal(ly)?\b",
    r"\blaw(s)?\b",
    r"\blawsuit\b",
    r"\bsue\b|\bsuing\b",
    r"\bliab(le|ility)\b",
    r"\bregulation(s)?\b",
    r"\bcompliance\b",
    r"\brequired by law\b",
    r"\bviolat(e|ion|ing)\b",
    r"\billegal\b",
    r"\brights?\b",
    r"\bobligated?\b",
    r"\bentitled\b",
    r"\bl[ée]gal(ement)?\b",
    r"\br[èeé]glementation\b",
    r"\bconformit[ée]\b",
    r"\bobligatoire\b",
)
_LEGAL_PATTERN = re.compile("|".join(_LEGAL_KEYWORDS), re.IGNORECASE)


def is_legal_or_regulatory_question(question: str) -> bool:
    """Heuristic, deliberately over-inclusive: a keyword hit is cheap (one extra
    sentence of caution); a miss risks an unqualified legal conclusion.
    """
    return bool(_LEGAL_PATTERN.search(question))


def build_user_turn(
    question: str,
    context_text: str,
    *,
    confidence_level: str | None = None,
    legal_flag: bool | None = None,
) -> str:
    """Assembles the final user turn: CONTEXT, any scenario-specific caution
    addenda, then QUESTION. `confidence_level` and `legal_flag` are optional so
    callers that don't have that information yet still get a working prompt —
    they just lose the extra reinforcement.
    """
    parts = [context_text or _NO_CONTEXT_TEXT]

    if confidence_level in ("low", "medium"):
        parts.append(_LOW_CONFIDENCE_ADDENDUM)

    if legal_flag is None:
        legal_flag = is_legal_or_regulatory_question(question)
    if legal_flag:
        parts.append(_LEGAL_QUESTION_ADDENDUM)

    parts.append(f"QUESTION:\n{question}")
    return "\n\n".join(parts)
