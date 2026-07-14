"""Shared fixtures for chunking tests.

`FakeEmbedder` stands in for `backend.retrieval.embeddings.EmbeddingService` so
semantic/hybrid chunking logic can be tested without downloading a real
sentence-transformers model. It's a simple normalized bag-of-words hashing
embedding: sentences sharing vocabulary land close together in cosine space,
sentences about unrelated topics land near-orthogonal — enough signal to exercise
the breakpoint-detection algorithm meaningfully.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

_DIM = 64
_WORD_RE = re.compile(r"[\w]+")


def _hash_embed(text: str) -> np.ndarray:
    vector = np.zeros(_DIM, dtype="float32")
    for word in _WORD_RE.findall(text.lower()):
        # zlib.crc32 (not the builtin hash()) so results are stable across runs/
        # processes — Python randomizes str hash() per-process by default.
        index = zlib.crc32(word.encode("utf-8")) % _DIM
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class FakeEmbedder:
    """Duck-types `backend.ingestion.chunking.semantic_chunker.SentenceEmbedder`."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return (
            np.stack([_hash_embed(t) for t in texts])
            if texts
            else np.empty((0, _DIM), dtype="float32")
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
