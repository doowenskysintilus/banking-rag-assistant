"""Anti-hallucination prompt system for the banking RAG assistant.

- `system_prompt.SYSTEM_PROMPT` — the static, cached system prompt (grounding
  rules, citation requirements, legal-interpretation caution, worked examples).
- `templates.build_user_turn` — assembles the per-query user turn, adding
  confidence- and legal-question-specific caution addenda on top of the context.
- `backend.rag.guardrails` (a separate module — see there) checks the model's
  actual output against what this prompt asked for, rather than assuming the
  prompt alone is sufficient.
"""

from backend.llm.prompts.system_prompt import (
    HEDGE_PHRASES,
    INSUFFICIENT_INFO_PHRASE,
    SYSTEM_PROMPT,
)
from backend.llm.prompts.templates import (
    build_user_turn,
    is_legal_or_regulatory_question,
)

__all__ = [
    "SYSTEM_PROMPT",
    "INSUFFICIENT_INFO_PHRASE",
    "HEDGE_PHRASES",
    "build_user_turn",
    "is_legal_or_regulatory_question",
]
