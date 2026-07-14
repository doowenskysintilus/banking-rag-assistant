"""Intelligent overlap: carries whole trailing sentences from one chunk into the
start of the next, instead of a raw character-count overlap that can cut a sentence
(or a defined term, or a monetary amount) in half.

Operates on the same unit-index groups produced by `recursive_chunker.pack_units`
and `semantic_chunker.group_by_semantic_breakpoints`, so it composes with either.
"""

from __future__ import annotations


def add_overlap(
    groups: list[list[int]], overlap_sentences: int
) -> list[tuple[list[int], int]]:
    """Prepend the last `overlap_sentences` unit indices of each group onto the next
    group.

    Returns (indices, overlap_count) pairs, where `overlap_count` is how many of the
    leading indices in that group are carried-over context rather than new content —
    callers use this to report how much of a chunk is repeated from its predecessor.
    """
    if overlap_sentences <= 0 or len(groups) <= 1:
        return [(group, 0) for group in groups]

    result: list[tuple[list[int], int]] = []
    for position, group in enumerate(groups):
        if position == 0:
            result.append((list(group), 0))
            continue

        previous_group = groups[position - 1]
        carried = previous_group[-overlap_sentences:]
        # Never duplicate a unit that is already the start of this group (can happen
        # if a previous merge — see `pack_units`'s trailing-group merge — already
        # pulled those sentences in).
        carried = [i for i in carried if i not in group]

        result.append((carried + list(group), len(carried)))

    return result
