"""Main entry point for the chunking pipeline: structure detection + recursive
packing + optional semantic refinement + intelligent overlap + context preservation.

    HybridChunker(config).chunk(text)                 -> structure + recursive only
    HybridChunker(config).chunk(text, embedder=...)    -> structure + semantic refinement

Strategy behavior:
    FIXED       not handled here — see `backend.ingestion.chunker.chunk_text` (the
                legacy baseline, kept for benchmarking comparisons).
    RECURSIVE   structural sections, each packed to size by `recursive_chunker`.
                No embeddings needed; fast, deterministic.
    SEMANTIC    structural sections, each regrouped by embedding-similarity
                breakpoints (`semantic_chunker`), then size-bounded as a safety net.
                Requires an embedder; falls back to RECURSIVE per-section if none
                is given.
    HYBRID      like SEMANTIC, but only applies semantic regrouping to sections with
                enough atomic units to make embedding worthwhile
                (`semantic_min_units`); short sections (e.g. a two-sentence clause)
                are just packed directly. This is the recommended default.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.ingestion.chunking.overlap import add_overlap
from backend.ingestion.chunking.recursive_chunker import (
    atomize,
    locate_units,
    pack_units,
)
from backend.ingestion.chunking.semantic_chunker import (
    SentenceEmbedder,
    semantic_group_async,
)
from backend.ingestion.chunking.structure import split_into_sections
from backend.ingestion.chunking.types import Chunk, ChunkStrategy

DEFAULT_MAX_CHARS = 1000
DEFAULT_MIN_CHARS = 200
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_SEMANTIC_PERCENTILE = 95.0
DEFAULT_SEMANTIC_MIN_UNITS = 6


@dataclass
class ChunkerConfig:
    strategy: ChunkStrategy = ChunkStrategy.HYBRID
    max_chars: int = DEFAULT_MAX_CHARS
    min_chars: int = DEFAULT_MIN_CHARS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    semantic_breakpoint_percentile: float = DEFAULT_SEMANTIC_PERCENTILE
    semantic_min_units: int = DEFAULT_SEMANTIC_MIN_UNITS
    include_header_prefix: bool = True

    def __post_init__(self) -> None:
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        if self.min_chars >= self.max_chars:
            raise ValueError("min_chars must be smaller than max_chars")


def _group_len(units: list[str], group: list[int]) -> int:
    if not group:
        return 0
    return sum(len(units[i]) for i in group) + (len(group) - 1)


def _enforce_size_bounds(
    units: list[str], groups: list[list[int]], max_chars: int, min_chars: int
) -> list[list[int]]:
    """Safety net applied after semantic grouping: a semantic breakpoint run can be
    arbitrarily long (split further) or arbitrarily short (merged into a neighbor) —
    unlike `pack_units`, which enforces both bounds by construction.
    """
    expanded: list[list[int]] = []
    for group in groups:
        if _group_len(units, group) > max_chars:
            sub_units = [units[i] for i in group]
            for sub_group in pack_units(sub_units, max_chars, min_chars):
                expanded.append([group[j] for j in sub_group])
        else:
            expanded.append(group)

    if len(expanded) <= 1:
        return expanded

    result: list[list[int]] = []
    pending: list[int] = []
    for group in expanded:
        merged_group = pending + group
        pending = []
        if _group_len(units, merged_group) < min_chars:
            pending = merged_group
            continue
        result.append(merged_group)

    if pending:
        if result:
            result[-1] = result[-1] + pending
        else:
            result.append(pending)

    return result


class HybridChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()

    async def chunk(
        self, text: str, embedder: SentenceEmbedder | None = None
    ) -> list[Chunk]:
        cfg = self._config
        sections = split_into_sections(text)
        chunks: list[Chunk] = []
        index = 0

        for section in sections:
            units = atomize(section.text, cfg.max_chars)
            if not units:
                continue

            use_semantic = (
                cfg.strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.HYBRID)
                and embedder is not None
                and len(units) >= cfg.semantic_min_units
            )

            if use_semantic:
                groups = await semantic_group_async(
                    units, embedder, cfg.semantic_breakpoint_percentile
                )
                groups = _enforce_size_bounds(
                    units, groups, cfg.max_chars, cfg.min_chars
                )
                strategy_used = ChunkStrategy.SEMANTIC
            else:
                groups = pack_units(units, cfg.max_chars, cfg.min_chars)
                strategy_used = ChunkStrategy.RECURSIVE

            unit_spans = locate_units(section.text, units)
            groups_with_overlap = add_overlap(groups, cfg.overlap_sentences)

            for indices, overlap_count in groups_with_overlap:
                raw_text = " ".join(units[i] for i in indices)
                char_start = section.start_char + unit_spans[indices[0]][0]
                char_end = section.start_char + unit_spans[indices[-1]][1]

                content = raw_text
                if cfg.include_header_prefix and section.breadcrumb:
                    content = f"[{section.breadcrumb}]\n{raw_text}"

                chunks.append(
                    Chunk(
                        text=content,
                        raw_text=raw_text,
                        index=index,
                        char_start=char_start,
                        char_end=char_end,
                        header_path=section.breadcrumb,
                        strategy=strategy_used,
                        overlap_sentence_count=overlap_count,
                        approx_tokens=max(1, len(content) // 4),
                    )
                )
                index += 1

        return chunks
