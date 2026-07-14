"""Recursive, size-bounded chunk packing.

"Recursive" in the LangChain sense: split on the coarsest separator first (already
done one level up, by `backend.ingestion.chunking.structure`, into per-Article/
Clause/Note sections), then recurse into finer separators — sentences, then raw
words — only for the spans that are still too large to fit in one chunk.
"""

from __future__ import annotations

from backend.ingestion.chunking.sentence_split import split_sentences

_JOINER = " "


def atomize(text: str, max_chars: int) -> list[str]:
    """Split `text` into sentence-level atomic units, force-splitting on whitespace
    any single sentence that alone exceeds `max_chars` (e.g. a long run-on clause or
    a wide table row with no sentence punctuation).
    """
    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(force_split_long_unit(sentence, max_chars))
    return units


def force_split_long_unit(unit: str, max_chars: int) -> list[str]:
    """Word-boundary fallback for a single unit that exceeds `max_chars` on its own."""
    words = unit.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
            added_len = len(word)
        current.append(word)
        current_len += added_len

    if current:
        pieces.append(" ".join(current))

    # A single word longer than max_chars (pathological input) is kept whole rather
    # than butchered mid-token — better to slightly exceed the target than emit
    # meaningless fragments.
    return pieces or [unit]


def _units_length(units: list[str], indices: list[int]) -> int:
    if not indices:
        return 0
    return sum(len(units[i]) for i in indices) + len(_JOINER) * (len(indices) - 1)


def pack_units(units: list[str], max_chars: int, min_chars: int) -> list[list[int]]:
    """Greedily bin-pack atomic units, in order, into groups whose joined length
    stays within `max_chars`.

    If that leaves a too-small trailing group, units are shifted one at a time
    from the end of its predecessor into its front (rebalancing the boundary
    between the last two groups) until the trailing group reaches `min_chars` —
    or until the predecessor would be emptied or `max_chars` would be exceeded,
    whichever comes first. (A straight merge of the whole trailing group into its
    predecessor — the obvious first idea — can never actually succeed: the
    predecessor was, by construction, already as full as it could get without
    exceeding `max_chars`, which is exactly why the trailing group had to start
    in the first place.)

    Returns groups of unit *indices* (not text) so callers can reuse the grouping
    for overlap bookkeeping and metadata.
    """
    if not units:
        return []

    groups: list[list[int]] = []
    current: list[int] = []
    current_len = 0

    for i, unit in enumerate(units):
        sep_len = len(_JOINER) if current else 0
        if current and current_len + sep_len + len(unit) > max_chars:
            groups.append(current)
            current = []
            current_len = 0
            sep_len = 0
        current.append(i)
        current_len += sep_len + len(unit)

    if current:
        groups.append(current)

    if len(groups) > 1:
        while len(groups[-2]) > 1 and _units_length(units, groups[-1]) < min_chars:
            borrowed_index = groups[-2][-1]
            sep_len = len(_JOINER) if groups[-1] else 0
            new_last_len = (
                len(units[borrowed_index]) + sep_len + _units_length(units, groups[-1])
            )
            if new_last_len > max_chars:
                break
            groups[-2].pop()
            groups[-1].insert(0, borrowed_index)

    return groups


def join_units(units: list[str], indices: list[int]) -> str:
    return _JOINER.join(units[i] for i in indices)


def locate_units(haystack: str, units: list[str]) -> list[tuple[int, int]]:
    """Find each unit's (start, end) character offsets within `haystack`, in order.

    Units come from `atomize()`, which only ever slices/strips `haystack` (via
    `sentence_split.split_sentences`) — so each one is normally an exact substring
    and a monotonically-advancing `str.find` locates it precisely. The one exception
    is a `force_split_long_unit` fragment re-joined with single spaces: if the source
    used different internal whitespace (newlines, double spaces), the fragment won't
    match verbatim and we fall back to a zero-width span at the current cursor rather
    than raising — losing precise offsets for that one oversized-sentence fallback
    chunk is an acceptable trade-off over crashing ingestion.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        idx = haystack.find(unit, cursor)
        if idx == -1:
            idx = haystack.find(unit)
        if idx == -1:
            spans.append((cursor, cursor))
            continue
        end = idx + len(unit)
        spans.append((idx, end))
        cursor = end
    return spans
