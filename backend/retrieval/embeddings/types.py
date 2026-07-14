"""Shared types for the embeddings module."""

from __future__ import annotations

import abc

import numpy as np


class EmbeddingProvider(abc.ABC):
    """One implementation per backend (OpenAI API, BGE-Large, Instructor-XL, ...).

    Providers are asymmetric-aware: `is_query=True` lets a provider apply whatever
    instruction/prefix its model expects for search queries as opposed to indexed
    documents (BGE and Instructor-XL both recommend different text for each side of
    retrieval; OpenAI's embeddings are symmetric and ignore the flag).
    """

    name: str
    dimension: int

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        """Embed a single batch of texts. Returned array is L2-normalized,
        shape (len(texts), self.dimension), dtype float32.
        """

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this provider's dependencies/credentials are usable right now
        (without necessarily having loaded the model yet). Used by the automatic
        provider selector.
        """


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")
