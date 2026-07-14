"""A small from-scratch Okapi BM25 index — the lexical half of hybrid search.

Dense embeddings are good at semantic similarity but can miss exact terms that
matter a lot in banking documents: account/clause numbers, defined terms, ISO
currency codes, regulation citations. BM25 catches those by term overlap instead.

Implemented in pure Python (no extra dependency) since a from-scratch BM25 is
compact and this index's scale already matches the single-instance FAISS store it
sits next to (see `backend.retrieval.vector_store.faiss_store`).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._term_freqs: dict[int, Counter[str]] = {}
        self._doc_lengths: dict[int, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_length = 0.0

    def __len__(self) -> int:
        return len(self._term_freqs)

    def add(self, vector_id: int, text: str) -> None:
        if vector_id in self._term_freqs:
            self.remove(vector_id)

        tokens = tokenize(text)
        freqs = Counter(tokens)
        self._term_freqs[vector_id] = freqs
        self._doc_lengths[vector_id] = len(tokens)
        for term in freqs:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._recompute_avg_doc_length()

    def remove(self, vector_id: int) -> None:
        freqs = self._term_freqs.pop(vector_id, None)
        if freqs is None:
            return
        self._doc_lengths.pop(vector_id, None)
        for term in freqs:
            remaining = self._doc_freq.get(term, 0) - 1
            if remaining <= 0:
                self._doc_freq.pop(term, None)
            else:
                self._doc_freq[term] = remaining
        self._recompute_avg_doc_length()

    def _recompute_avg_doc_length(self) -> None:
        n = len(self._doc_lengths)
        self._avg_doc_length = (sum(self._doc_lengths.values()) / n) if n else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._term_freqs)
        doc_freq = self._doc_freq.get(term, 0)
        # +1 smoothing keeps idf non-negative even for terms in >50% of documents.
        return math.log(1 + (n - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns (vector_id, bm25_score) pairs, highest score first. Documents
        that share no term with the query are omitted (score would be 0).
        """
        if not self._term_freqs:
            return []

        query_terms = tokenize(query)
        scores: dict[int, float] = {}

        for vector_id, freqs in self._term_freqs.items():
            doc_length = self._doc_lengths[vector_id]
            score = 0.0
            for term in query_terms:
                term_freq = freqs.get(term)
                if not term_freq:
                    continue
                idf = self._idf(term)
                denominator = term_freq + self._k1 * (
                    1 - self._b + self._b * doc_length / (self._avg_doc_length or 1)
                )
                score += idf * (term_freq * (self._k1 + 1)) / denominator
            if score > 0:
                scores[vector_id] = score

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
