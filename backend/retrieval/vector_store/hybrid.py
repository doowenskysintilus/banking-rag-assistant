"""Reciprocal Rank Fusion — combines a dense (FAISS) and sparse (BM25) ranking.

RRF fuses by *rank*, not raw score, which sidesteps the fact that cosine similarity
(roughly 0-1) and BM25 scores (unbounded, corpus-dependent) live on incomparable
scales — a problem a naive weighted-sum fusion would have to calibrate for.
"""

from __future__ import annotations

_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = _DEFAULT_RRF_K
) -> dict[int, float]:
    """`rankings` is a list of ranked-id lists (best first), e.g. [dense_ids, sparse_ids].
    An id absent from a given ranking simply doesn't contribute a term from it.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, vector_id in enumerate(ranking):
            scores[vector_id] = scores.get(vector_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
