"""Fake dependency-free stand-ins for the RAG engine's collaborators."""

from __future__ import annotations

from collections.abc import AsyncIterator

from backend.llm.client import LLMResponse, LLMStreamChunk
from backend.retrieval.reranking.types import RerankerProvider


class FakeRerankerProvider(RerankerProvider):
    """Scores documents by a simple substring-based heuristic (or a fixed score
    map), so reranking behavior in tests is deterministic without a real model.
    """

    name = "fake"

    def __init__(
        self, scores: dict[str, float] | None = None, default_score: float = 0.9
    ) -> None:
        self._scores = scores or {}
        self._default_score = default_score
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self._scores.get(doc, self._default_score) for doc in documents]

    @property
    def is_available(self) -> bool:
        return True


class FakeOpenAIClient:
    """Stands in for `backend.llm.client.OpenAIClient` — records the messages it
    was called with and returns a scripted response (or raises, if configured to).
    """

    def __init__(
        self,
        answer_text: str = "This is the answer.",
        model: str = "fake-model",
        input_tokens: int = 100,
        output_tokens: int = 20,
        stream_chunks: list[str] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._answer_text = answer_text
        self._model = model
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._stream_chunks = (
            stream_chunks
            if stream_chunks is not None
            else ["This ", "is ", "the ", "answer."]
        )
        self._raise_error = raise_error
        self.calls: list[list[dict]] = []

    def _final_response(self) -> LLMResponse:
        return LLMResponse(
            text=self._answer_text,
            model=self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            stop_reason="end_turn",
        )

    async def answer(self, messages: list[dict]) -> LLMResponse:
        self.calls.append(messages)
        if self._raise_error is not None:
            raise self._raise_error
        return self._final_response()

    async def answer_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMStreamChunk]:
        self.calls.append(messages)
        if self._raise_error is not None:
            raise self._raise_error
        for piece in self._stream_chunks:
            yield LLMStreamChunk(text=piece)
        yield LLMStreamChunk(
            text="", is_final=True, final_response=self._final_response()
        )
