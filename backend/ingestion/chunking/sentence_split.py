"""Abbreviation-aware sentence segmentation.

Deliberately regex-based rather than a statistical/NLP-model splitter (no spaCy/NLTK
model download) — banking contracts, regulations, and financial reports are dense with
abbreviations, decimal numbers, and clause numbering that a naive
`re.split(r"(?<=[.!?])\\s+")` would shatter mid-sentence (e.g. "Art. 5", "12.5%",
'the Borrower (the "Client")', "e.g. overdraft fees").
"""

from __future__ import annotations

import re

# Private-use-area character used to temporarily stand in for a period that must NOT
# be treated as a sentence boundary. Restored to "." before returning. Chosen from the
# Unicode Private Use Area (U+E000) so it can never collide with real document text.
_PERIOD_PLACEHOLDER = ""

# Lowercase, single-token abbreviations (no internal period) likely to appear in
# banking contracts, regulatory text, and financial reports.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "mx",
    "dr",
    "prof",
    "sr",
    "jr",
    "inc",
    "corp",
    "ltd",
    "co",
    "llc",
    "llp",
    "plc",
    "gmbh",
    "vs",
    "etc",
    "cf",
    "viz",
    "al",
    "art",
    "arts",
    "sec",
    "secs",
    "no",
    "nos",
    "para",
    "paras",
    "ch",
    "chs",
    "fig",
    "figs",
    "p",
    "pp",
    "vol",
    "vols",
    "op",
    "cit",
    "fol",
    "st",
    "ave",
    "blvd",
    "approx",
    "est",
    "rev",
    "m",
    "mme",
    "mlle",
    "pr",
    "cie",
    "sarl",
    "sas",
}

# Multi-part abbreviations that contain their own internal periods — must be matched
# as literal phrases (longest first, so "u.s.a." isn't pre-empted by "u.s.").
_MULTI_PART_ABBREVIATIONS = sorted(
    ["e.g.", "i.e.", "u.s.a.", "u.s.", "u.k."], key=len, reverse=True
)
_MULTI_PART_RE = re.compile(
    "|".join(re.escape(p) for p in _MULTI_PART_ABBREVIATIONS), re.IGNORECASE
)

# "12.5", "5.2.1" — internal periods between digits are numbering/decimals, never
# sentence boundaries.
_DECIMAL_NUMBER_RE = re.compile(r"(?<=\d)\.(?=\d)")

_ABBREV_PERIOD_RE = re.compile(r"([A-Za-zÀ-ÿ]+)\.")

# A period/!/? that ends a sentence: optionally followed by a closing quote/bracket,
# then whitespace, then something that looks like the start of a new sentence
# (capital letter, digit, quote, or opening bracket).
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])(?:[\'"”\)\]])?\s+(?=[A-Z0-9À-Ý"\'“(])')


def _protect_multi_part_abbreviations(text: str) -> str:
    return _MULTI_PART_RE.sub(
        lambda m: m.group(0).replace(".", _PERIOD_PLACEHOLDER), text
    )


def _protect_single_word_abbreviations(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        word = match.group(1)
        if word.lower() in _ABBREVIATIONS:
            return f"{word}{_PERIOD_PLACEHOLDER}"
        return match.group(0)

    return _ABBREV_PERIOD_RE.sub(_replace, text)


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, preserving decimal numbers and common
    legal/financial abbreviations. Returns non-empty, stripped sentences in
    document order.
    """
    stripped = text.strip()
    if not stripped:
        return []

    protected = _DECIMAL_NUMBER_RE.sub(_PERIOD_PLACEHOLDER, stripped)
    protected = _protect_multi_part_abbreviations(protected)
    protected = _protect_single_word_abbreviations(protected)

    raw_sentences = _SENTENCE_BOUNDARY_RE.split(protected)

    sentences = [s.replace(_PERIOD_PLACEHOLDER, ".").strip() for s in raw_sentences]
    return [s for s in sentences if s]
