"""Context Builder stage: turns reranked chunks into the CONTEXT block the LLM
sees, and the structured per-source citation metadata the API response exposes.

Each source gets a `[Source N: filename — header path]` marker in the text, both
so the model can (and is instructed to, per `backend.llm.prompts.SYSTEM_PROMPT`)
cite which source backs each claim, and so the marker's index lines up 1:1 with
the `sources` list returned alongside the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

_EXCERPT_LENGTH = 280
_NO_CONTEXT_TEXT = "No relevant internal documents were found for this question."


@dataclass(frozen=True)
class SourceChunk:
    """Input to the context builder — one reranked, confidence-filtered chunk."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    header_path: str = ""


@dataclass(frozen=True)
class ContextSource:
    """Output: one entry in the citation list the API/UI displays."""

    index: int  # 1-based, matches the "[Source N]" marker in context_text
    document_id: str
    chunk_id: str
    document_filename: str
    header_path: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class ContextBuildResult:
    context_text: str
    sources: list[ContextSource]


class ContextBuilder:
    def build(
        self, chunks: list[SourceChunk], filenames: dict[str, str]
    ) -> ContextBuildResult:
        if not chunks:
            return ContextBuildResult(context_text=_NO_CONTEXT_TEXT, sources=[])

        blocks: list[str] = []
        sources: list[ContextSource] = []

        for position, chunk in enumerate(chunks, start=1):
            filename = filenames.get(chunk.document_id, "Unknown document")
            label = f"[Source {position}: {filename}"
            if chunk.header_path:
                label += f" — {chunk.header_path}"
            label += "]"

            blocks.append(f"{label}\n{chunk.content}")
            sources.append(
                ContextSource(
                    index=position,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    document_filename=filename,
                    header_path=chunk.header_path,
                    score=chunk.score,
                    excerpt=chunk.content[:_EXCERPT_LENGTH],
                )
            )

        context_text = "CONTEXT:\n" + "\n\n".join(blocks)
        return ContextBuildResult(context_text=context_text, sources=sources)
