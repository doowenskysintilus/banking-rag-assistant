from backend.ingestion.chunking.overlap import add_overlap


def test_no_overlap_when_overlap_sentences_is_zero():
    groups = [[0, 1], [2, 3], [4, 5]]
    result = add_overlap(groups, overlap_sentences=0)
    assert result == [([0, 1], 0), ([2, 3], 0), ([4, 5], 0)]


def test_single_group_has_no_overlap():
    assert add_overlap([[0, 1, 2]], overlap_sentences=2) == [([0, 1, 2], 0)]


def test_carries_trailing_sentences_from_previous_group():
    groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    result = add_overlap(groups, overlap_sentences=1)

    assert result[0] == ([0, 1, 2], 0)
    assert result[1] == ([2, 3, 4, 5], 1)
    assert result[2] == ([5, 6, 7, 8], 1)


def test_overlap_larger_than_previous_group_is_capped():
    groups = [[0], [1, 2]]
    result = add_overlap(groups, overlap_sentences=5)
    # Only one sentence exists in the previous group, so only one can be carried.
    assert result[1] == ([0, 1, 2], 1)


def test_does_not_duplicate_a_unit_already_present_in_the_next_group():
    # Simulates the case where `pack_units` already merged a small trailing group
    # backward, so the "previous" group's tail already appears at the front of
    # this group.
    groups = [[0, 1, 2], [2, 3]]
    result = add_overlap(groups, overlap_sentences=2)
    assert result[1] == ([1, 2, 3], 1)


def test_empty_groups_returns_empty():
    assert add_overlap([], overlap_sentences=2) == []
