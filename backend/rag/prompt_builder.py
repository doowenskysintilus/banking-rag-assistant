"""Prompt Builder stage: assembles the final message list the model sees — prior
conversation turns plus this turn's (context + question), reinforced with
scenario-specific anti-hallucination addenda from `backend.llm.prompts.templates`
(low-confidence caution, legal/regulatory caution) — the last explicit stage
before the LLM call.

The system prompt itself is owned by `backend.llm.client.OpenAIClient` (it's
static and a call-mechanics concern, not something that varies per query).
"""

from __future__ import annotations

from backend.llm.client import ChatTurn
from backend.llm.prompts import build_user_turn


class PromptBuilder:
    def build(
        self,
        question: str,
        context_text: str,
        history: list[ChatTurn] | None = None,
        *,
        confidence_level: str | None = None,
    ) -> list[dict]:
        messages = [
            {"role": turn.role, "content": turn.content} for turn in (history or [])
        ]
        user_turn = build_user_turn(
            question, context_text, confidence_level=confidence_level
        )
        messages.append({"role": "user", "content": user_turn})
        return messages
