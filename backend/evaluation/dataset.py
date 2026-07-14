"""Builds a *live* evaluation dataset by actually running labeled questions
through a real `RAGEngine` — as opposed to `backend.evaluation.fixtures`'
pre-baked golden set, which exists to exercise the metric providers themselves
without needing a database or an LLM call.

Going through `engine.answer()` (the same public entry point real chat traffic
uses) rather than reaching into its private retrieval/prompt-building internals
means an evaluation run sees exactly what a real user would have: current
retrieval config, current reranker, current prompt. It also means each
evaluated question is durably audited via the engine's normal `RagQueryLog`
trail — evaluation traffic is real traffic through the system, not a mock.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation.types import EvalSample
from backend.rag.engine import RAGEngine


@dataclass(frozen=True)
class LabeledQuestion:
    """One row of a hand-labeled evaluation set: a question and the reference
    answer it should produce, used to compute `context_recall`/`answer_relevancy`
    against.
    """

    sample_id: str
    question: str
    ground_truth: str


class EvalDatasetBuilder:
    def __init__(self, engine: RAGEngine) -> None:
        self._engine = engine

    async def build(
        self, db: AsyncSession, owner_id: str, questions: list[LabeledQuestion]
    ) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for labeled in questions:
            response = await self._engine.answer(
                db, labeled.question, owner_id, conversation_id=None
            )
            samples.append(
                EvalSample(
                    sample_id=labeled.sample_id,
                    question=labeled.question,
                    answer=response.answer,
                    contexts=list(response.source_texts),
                    ground_truth=labeled.ground_truth,
                )
            )
        return samples
