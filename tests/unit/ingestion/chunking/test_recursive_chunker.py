from backend.ingestion.chunking.recursive_chunker import (
    atomize,
    force_split_long_unit,
    join_units,
    locate_units,
    pack_units,
)


class TestForceSplitLongUnit:
    def test_splits_on_word_boundaries_within_limit(self):
        unit = " ".join(["word"] * 20)  # 20*4 + 19 spaces = 99 chars
        pieces = force_split_long_unit(unit, max_chars=30)
        assert all(len(p) <= 30 for p in pieces)
        assert " ".join(pieces) == unit

    def test_keeps_single_oversized_word_whole(self):
        unit = "a" * 100
        pieces = force_split_long_unit(unit, max_chars=30)
        assert pieces == [unit]

    def test_short_unit_returns_single_piece(self):
        assert force_split_long_unit("short text", max_chars=100) == ["short text"]


class TestAtomize:
    def test_splits_into_sentences(self):
        text = "First sentence here. Second sentence here."
        units = atomize(text, max_chars=1000)
        assert len(units) == 2

    def test_force_splits_a_sentence_longer_than_max_chars(self):
        long_sentence = " ".join(["word"] * 50) + "."
        units = atomize(long_sentence, max_chars=50)
        assert len(units) > 1
        assert all(len(u) <= 50 for u in units)

    def test_empty_text_returns_no_units(self):
        assert atomize("", max_chars=100) == []


class TestPackUnits:
    def test_packs_units_within_max_chars(self):
        units = ["a" * 40, "b" * 40, "c" * 40]
        groups = pack_units(units, max_chars=90, min_chars=10)
        for group in groups:
            joined_len = sum(len(units[i]) for i in group) + (len(group) - 1)
            assert joined_len <= 90

    def test_preserves_order_and_all_units(self):
        units = ["one", "two", "three", "four"]
        groups = pack_units(units, max_chars=1000, min_chars=0)
        flattened = [i for group in groups for i in group]
        assert flattened == [0, 1, 2, 3]

    def test_rebalances_small_trailing_group_by_borrowing_from_predecessor(self):
        # max_chars=105 sits between "a"*50+"b"*50 (101, fits in one group) and all
        # three combined (107, doesn't) — so greedy packing first produces
        # [[0, 1], [2]], then the trailing group (5 chars) borrows unit 1 from its
        # predecessor to reach a healthier size, without ever exceeding max_chars.
        units = ["a" * 50, "b" * 50, "c" * 5]
        groups = pack_units(units, max_chars=105, min_chars=20)
        assert groups == [[0], [1, 2]]

    def test_does_not_rebalance_if_predecessor_has_only_one_unit_to_give(self):
        # Predecessor groups of a single unit can't be shrunk further (that unit
        # would just become its own new trailing fragment), so no borrowing happens
        # even though the trailing group is below min_chars.
        units = ["a" * 50, "b" * 50, "c" * 5]
        groups = pack_units(units, max_chars=55, min_chars=20)
        assert groups[-1] == [2]

    def test_does_not_rebalance_if_borrowing_would_exceed_max_chars(self):
        # Predecessor group [0, 1] ("a"*2 + "b"*80 = 83 chars) fits under
        # max_chars=85, so greedy packing splits here. Its last unit ("b"*80) is
        # itself too big to move into the trailing 5-char group without blowing
        # past max_chars (80 + 1 + 5 = 86 > 85), so no borrowing happens.
        units = ["a" * 2, "b" * 80, "c" * 5]
        groups = pack_units(units, max_chars=85, min_chars=20)
        assert groups == [[0, 1], [2]]

    def test_empty_units_returns_no_groups(self):
        assert pack_units([], max_chars=100, min_chars=10) == []

    def test_single_unit_returns_single_group(self):
        assert pack_units(["only unit"], max_chars=100, min_chars=0) == [[0]]


def test_join_units_reconstructs_text_with_single_space():
    units = ["First sentence.", "Second sentence."]
    assert join_units(units, [0, 1]) == "First sentence. Second sentence."


class TestLocateUnits:
    def test_locates_units_in_order(self):
        text = "First sentence. Second sentence. Third sentence."
        units = ["First sentence.", "Second sentence.", "Third sentence."]
        spans = locate_units(text, units)
        for (start, end), unit in zip(spans, units):
            assert text[start:end] == unit

    def test_spans_are_monotonically_increasing(self):
        text = "Alpha beta. Alpha beta. Gamma delta."
        units = ["Alpha beta.", "Alpha beta.", "Gamma delta."]
        spans = locate_units(text, units)
        assert spans[0][0] < spans[1][0] < spans[2][0]

    def test_missing_unit_falls_back_to_zero_width_span_without_crashing(self):
        text = "Some text."
        spans = locate_units(text, ["not present in text"])
        assert spans == [(0, 0)]
