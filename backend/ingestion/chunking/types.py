"""Shared data types for the chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChunkStrategy(str, Enum):
    FIXED = "fixed"  # legacy fixed-size baseline (backend.ingestion.chunker.chunk_text)
    RECURSIVE = (
        "recursive"  # structure + size-bounded recursive packing, no embeddings needed
    )
    SEMANTIC = (
        "semantic"  # embedding-similarity breakpoints, ignores document structure
    )
    HYBRID = "hybrid"  # structure-aware backbone + semantic refinement within sections


@dataclass
class Section:
    """A structurally-coherent span of the document (e.g. one Article/Clause/Note),
    tagged with its breadcrumb in the document's heading hierarchy.
    """

    breadcrumb: (
        str  # e.g. "Article 5 > Clause 5.2", "" for text before the first heading
    )
    start_char: int
    end_char: int
    text: str


@dataclass
class Chunk:
    """One chunking pipeline output unit, ready for embedding + storage."""

    text: str  # final content (may include a prepended context header)
    raw_text: str  # the span as it appears in the source, no context prefix
    index: int
    char_start: int
    char_end: int
    header_path: str  # breadcrumb this chunk belongs to, "" if the doc has no structure
    strategy: ChunkStrategy
    overlap_sentence_count: int = (
        0  # leading sentences carried over from the previous chunk
    )
    approx_tokens: int = 0
    extra: dict = field(default_factory=dict)
