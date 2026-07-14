"""Chunk quality metrics — used by the benchmark script and available to call
directly after any ingestion run to catch a chunking regression before it hurts
retrieval quality.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.ingestion.chunking.sentence_split import split_sentences
from backend.ingestion.chunking.types import Chunk

_SENTENCE_END_CHARS = set(".!?\"')]”’")


@dataclass
class ChunkingMetrics:
    num_chunks: int
    avg_chars: float
    median_chars: float
    min_chars: int
    max_chars: int
    stdev_chars: float

    pct_within_target_range: float
    pct_ending_at_sentence_boundary: float
    pct_with_context_header: float

    avg_overlap_sentences: float
    pct_chunks_with_overlap: float

    avg_approx_tokens: float

    # Optional — only populated when an `embed_fn` is supplied to `compute_metrics`.
    avg_intra_chunk_coherence: float | None = None
    avg_boundary_coherence: float | None = None
    coherence_gap: float | None = None


def _ends_at_sentence_boundary(raw_text: str) -> bool:
    stripped = raw_text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _coherence_scores(
    chunks: list[Chunk], embed_fn: Callable[[list[str]], np.ndarray]
) -> tuple[float | None, float | None]:
    intra_scores: list[float] = []
    boundary_scores: list[float] = []

    boundary_endpoints: list[
        tuple[np.ndarray, np.ndarray]
    ] = []  # (last-of-i, first-of-i+1)

    for chunk in chunks:
        sentences = split_sentences(chunk.raw_text)
        if len(sentences) < 2:
            boundary_endpoints.append((None, None))  # type: ignore[arg-type]
            continue
        embeddings = embed_fn(sentences)
        for i in range(len(embeddings) - 1):
            intra_scores.append(_cosine_similarity(embeddings[i], embeddings[i + 1]))
        boundary_endpoints.append((embeddings[-1], embeddings[0]))

    for i in range(len(boundary_endpoints) - 1):
        last_of_current = boundary_endpoints[i][0]
        first_of_next = boundary_endpoints[i + 1][1]
        if last_of_current is None or first_of_next is None:
            continue
        boundary_scores.append(_cosine_similarity(last_of_current, first_of_next))

    avg_intra = sum(intra_scores) / len(intra_scores) if intra_scores else None
    avg_boundary = (
        sum(boundary_scores) / len(boundary_scores) if boundary_scores else None
    )
    return avg_intra, avg_boundary


def compute_metrics(
    chunks: list[Chunk],
    *,
    target_min_chars: int,
    target_max_chars: int,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> ChunkingMetrics:
    if not chunks:
        return ChunkingMetrics(
            num_chunks=0,
            avg_chars=0.0,
            median_chars=0.0,
            min_chars=0,
            max_chars=0,
            stdev_chars=0.0,
            pct_within_target_range=0.0,
            pct_ending_at_sentence_boundary=0.0,
            pct_with_context_header=0.0,
            avg_overlap_sentences=0.0,
            pct_chunks_with_overlap=0.0,
            avg_approx_tokens=0.0,
        )

    lengths = [len(c.raw_text) for c in chunks]
    n = len(chunks)

    within_range = sum(
        1 for length in lengths if target_min_chars <= length <= target_max_chars
    )
    at_boundary = sum(1 for c in chunks if _ends_at_sentence_boundary(c.raw_text))
    with_header = sum(1 for c in chunks if c.header_path)
    with_overlap = sum(1 for c in chunks if c.overlap_sentence_count > 0)

    avg_intra = avg_boundary = gap = None
    if embed_fn is not None:
        avg_intra, avg_boundary = _coherence_scores(chunks, embed_fn)
        if avg_intra is not None and avg_boundary is not None:
            gap = avg_intra - avg_boundary

    return ChunkingMetrics(
        num_chunks=n,
        avg_chars=statistics.mean(lengths),
        median_chars=statistics.median(lengths),
        min_chars=min(lengths),
        max_chars=max(lengths),
        stdev_chars=statistics.pstdev(lengths) if n > 1 else 0.0,
        pct_within_target_range=within_range / n,
        pct_ending_at_sentence_boundary=at_boundary / n,
        pct_with_context_header=with_header / n,
        avg_overlap_sentences=sum(c.overlap_sentence_count for c in chunks) / n,
        pct_chunks_with_overlap=with_overlap / n,
        avg_approx_tokens=sum(c.approx_tokens for c in chunks) / n,
        avg_intra_chunk_coherence=avg_intra,
        avg_boundary_coherence=avg_boundary,
        coherence_gap=gap,
    )
