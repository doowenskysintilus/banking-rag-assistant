"""Duplicate detection: exact (content hash) and near-duplicate (simhash)."""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_SIMHASH_BITS = 64
_DEFAULT_NEAR_DUPLICATE_THRESHOLD = 3  # max Hamming distance out of 64 bits


def compute_content_hash(text: str) -> str:
    """SHA-256 hex digest of the cleaned text — used for exact-duplicate detection.

    Callers must pass already-cleaned text (see `backend.ingestion.cleaning.clean_text`)
    so that whitespace/formatting differences from re-extracting the same source
    don't produce different hashes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - non-cryptographic use
    return int.from_bytes(digest[:8], byteorder="big")


def compute_simhash(text: str) -> int:
    """64-bit simhash fingerprint for near-duplicate detection.

    Documents that differ only by minor edits (a changed paragraph, updated date,
    reformatted whitespace) produce simhashes with a small Hamming distance;
    unrelated documents produce ~32-bit average distance (random).
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0

    weights = [0] * _SIMHASH_BITS
    for token in tokens:
        token_hash = _token_hash(token)
        for bit in range(_SIMHASH_BITS):
            weights[bit] += 1 if (token_hash >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(_SIMHASH_BITS):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def to_signed_64(value: int) -> int:
    """Reinterprets an unsigned 64-bit simhash as a signed 64-bit int.

    `compute_simhash` can set bit 63, producing values up to 2**64-1, which
    overflow a SQL INTEGER column (signed 64-bit) — this must be applied
    before persisting `content_simhash`.
    """
    return value - (1 << 64) if value & (1 << 63) else value


def from_signed_64(value: int) -> int:
    """Inverse of `to_signed_64` — restores the unsigned 64-bit fingerprint
    read back from storage so `hamming_distance`'s bit operations stay correct.
    """
    return value & ((1 << 64) - 1)


def is_near_duplicate(
    simhash_a: int, simhash_b: int, threshold: int = _DEFAULT_NEAR_DUPLICATE_THRESHOLD
) -> bool:
    return hamming_distance(simhash_a, simhash_b) <= threshold


def find_near_duplicate(
    simhash: int,
    candidates: list[tuple[str, int]],
    threshold: int = _DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> str | None:
    """Return the id of the closest candidate within `threshold`, or None.

    `candidates` is a list of (id, simhash) pairs — callers fetch these from the
    database (e.g. other current documents owned by the same user) so this
    function stays pure and easy to unit test.
    """
    best_id: str | None = None
    best_distance = threshold + 1

    for candidate_id, candidate_simhash in candidates:
        distance = hamming_distance(simhash, candidate_simhash)
        if distance <= threshold and distance < best_distance:
            best_id = candidate_id
            best_distance = distance

    return best_id
