"""Thin, production-hardened wrapper around the OpenAI Chat Completions API.

Responsibilities:
- Own the AsyncOpenAI client lifecycle (single instance, reused across requests).
- Expose both a non-streaming call (for simple request/response) and a streaming
  generator (for the chat SSE endpoint).
- Translate every OpenAI SDK exception into our internal `ExternalServiceError`
  hierarchy so routers/services never import the `openai` package directly.

No explicit prompt-caching setup is needed here (unlike an Anthropic-style
`cache_control: ephemeral` block): OpenAI's Chat Completions API caches
repeated prompt prefixes automatically for prompts over ~1024 tokens, which
this large, static system prompt comfortably exceeds — nothing to configure.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ExternalServiceError, RateLimitedError
from backend.core.logging import get_logger
from backend.llm.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None


@dataclass
class LLMStreamChunk:
    text: str
    is_final: bool = False
    final_response: LLMResponse | None = field(default=None)


class OpenAIClient:
    """Wraps `openai.AsyncOpenAI` (Chat Completions) for the RAG assistant's needs."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            logger.warning("openai_api_key_missing")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key or None)
        self._model = settings.openai_chat_model
        self._max_tokens = settings.openai_chat_max_tokens
        self._temperature = settings.openai_chat_temperature

    def _messages(self, messages: list[dict]) -> list[dict]:
        # OpenAI has no separate "system" request field the way Anthropic
        # does — the system prompt is just the first message, role="system".
        return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    def _request_kwargs(self, messages: list[dict], **extra: object) -> dict:
        # `max_completion_tokens` (not the older `max_tokens`) is the current
        # OpenAI parameter name — introduced alongside the o-series reasoning
        # models and now the forward-compatible choice across chat models
        # too; only genuinely old models would reject it in favor of
        # `max_tokens`.
        return {
            "model": self._model,
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": self._messages(messages),
            **extra,
        }

    async def answer(self, messages: list[dict]) -> LLMResponse:
        """Non-streaming call — used by the synchronous /chat/query endpoint.

        `messages` is the fully-built turn list (prior history + this turn's
        context/question) — see `backend.rag.prompt_builder.PromptBuilder`, the
        "Prompt Builder" stage of the RAG engine. This client only knows how to
        actually call the model with it.
        """
        try:
            response = await self._client.chat.completions.create(
                **self._request_kwargs(messages)
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as typed error below
            raise self._map_error(exc) from exc

        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise ExternalServiceError(
                "The assistant declined to answer this request.",
                details={"stop_reason": "content_filter"},
            )

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=choice.finish_reason,
        )

    async def answer_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streaming call — used by the /chat/stream SSE endpoint. See `answer()`
        for where `messages` comes from.
        """
        text_parts: list[str] = []
        model_name = self._model
        finish_reason: str | None = None
        input_tokens = 0
        output_tokens = 0

        try:
            stream = await self._client.chat.completions.create(
                **self._request_kwargs(
                    messages, stream=True, stream_options={"include_usage": True}
                )
            )
            async for event in stream:
                if event.model:
                    model_name = event.model
                if event.choices:
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        text_parts.append(delta.content)
                        yield LLMStreamChunk(text=delta.content)
                    if event.choices[0].finish_reason:
                        finish_reason = event.choices[0].finish_reason
                # Only present on the final chunk when stream_options.include_usage=True.
                if event.usage:
                    input_tokens = event.usage.prompt_tokens
                    output_tokens = event.usage.completion_tokens
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc

        yield LLMStreamChunk(
            text="",
            is_final=True,
            final_response=LLMResponse(
                text="".join(text_parts),
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=finish_reason,
            ),
        )

    @staticmethod
    def _map_error(exc: Exception) -> Exception:
        """Translate OpenAI SDK exceptions into our internal error hierarchy.

        Ordered most-specific-first, per the SDK's typed exception hierarchy.
        """
        import openai

        if isinstance(exc, openai.RateLimitError):
            logger.warning("openai_chat_rate_limited")
            return RateLimitedError(
                "The assistant is temporarily rate limited. Please retry shortly."
            )
        if isinstance(exc, openai.AuthenticationError):
            logger.error("openai_chat_authentication_error")
            return ExternalServiceError("Assistant authentication failed.")
        if isinstance(exc, openai.PermissionDeniedError):
            logger.error("openai_chat_permission_denied")
            return ExternalServiceError(
                "Assistant access is not permitted for this configuration."
            )
        if isinstance(exc, openai.NotFoundError):
            logger.error("openai_chat_model_not_found")
            return ExternalServiceError("Configured OpenAI model was not found.")
        if isinstance(exc, openai.APIStatusError):
            logger.error(
                "openai_chat_api_status_error", extra={"status_code": exc.status_code}
            )
            if exc.status_code >= 500:
                return ExternalServiceError(
                    "The assistant service is temporarily unavailable."
                )
            return ExternalServiceError(f"Assistant request failed: {exc.message}")
        if isinstance(exc, openai.APIConnectionError):
            logger.error("openai_chat_connection_error")
            return ExternalServiceError("Could not reach the assistant service.")
        logger.exception("openai_chat_unexpected_error")
        return ExternalServiceError("Unexpected error calling the assistant service.")


_client_singleton: OpenAIClient | None = None


def get_openai_client() -> OpenAIClient:
    """FastAPI dependency — returns a process-wide singleton `OpenAIClient`."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAIClient(get_settings())
    return _client_singleton
