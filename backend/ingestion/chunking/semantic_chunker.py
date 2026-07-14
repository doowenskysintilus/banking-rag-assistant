"""Semantic chunking: group sentences by embedding-similarity breakpoints instead of
raw size, so a chunk boundary falls where the *topic* actually shifts (e.g. between
a definitions clause and a covenants clause) rather than at an arbitrary character
count.

Method (percentile breakpoint, as popularized by Greg Kamradt / LlamaIndex's semantic
splitter): embed each sentence, compute the cosine distance between every pair of
consecutive sentences, and cut wherever that distance is unusually large relative to
the rest of the document (above the Nth percentile). Sentences between two cuts form
one semantic group.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SentenceEmbedder(Protocol):
    async def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise distance between consecutive rows. Assumes embeddings are already
    L2-normalized (as `backend.retrieval.embeddings.EmbeddingService` guarantees),
    so cosine similarity is just the dot product.
    """
    if len(embeddings) < 2:
        return np.empty((0,), dtype="float32")
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return 1.0 - similarities


def group_by_semantic_breakpoints(
    units: list[str],
    embeddings: np.ndarray,
    breakpoint_percentile: float = 95.0,
) -> list[list[int]]:
    """Group atomic unit indices into semantically-coherent runs.

    `embeddings` must have one row per entry in `units`, in the same order.
    """
    n = len(units)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    if len(embeddings) != n:
        raise ValueError(
            f"Expected {n} embeddings (one per unit), got {len(embeddings)}"
        )

    distances = _cosine_distances(embeddings)
    threshold = float(np.percentile(distances, breakpoint_percentile))

    groups: list[list[int]] = []
    current = [0]
    for i, distance in enumerate(distances):
        # distances[i] is the distance between unit i and unit i+1.
        if distance > threshold:
            groups.append(current)
            current = []
        current.append(i + 1)
    groups.append(current)

    return groups


async def semantic_group_async(
    units: list[str],
    embedder: SentenceEmbedder,
    breakpoint_percentile: float = 95.0,
) -> list[list[int]]:
    """Async convenience wrapper: embeds `units` then delegates to the pure grouping
    function above. Kept separate so the grouping logic is unit-testable without a
    real (or even fake-async) embedding model.
    """
    if not units:
        return []
    embeddings = await embedder.embed_texts(units)
    return group_by_semantic_breakpoints(
        units, embeddings, breakpoint_percentile=breakpoint_percentile
    )
