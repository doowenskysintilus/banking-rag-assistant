import pytest

from backend.retrieval.reranking.metrics import (
    average_precision,
    dcg_at_k,
    grounding_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b"], {"c"}, k=2) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1 / 3)

    def test_k_smaller_than_list(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=1) == 1.0

    def test_empty_ranked_list(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestRecallAtK:
    def test_finds_all_relevant_within_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_misses_relevant_beyond_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=1) == pytest.approx(0.5)

    def test_no_relevant_ids_returns_zero(self):
        assert recall_at_k(["a"], set(), k=5) == 0.0


class TestReciprocalRank:
    def test_relevant_at_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_relevant_at_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found_is_zero(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestAveragePrecision:
    def test_perfect_ranking(self):
        assert average_precision(["a", "b"], {"a", "b"}) == 1.0

    def test_relevant_docs_scattered(self):
        # relevant at positions 1 and 3: precision@1=1.0, precision@3=2/3
        ap = average_precision(["a", "x", "b"], {"a", "b"})
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_relevant_ids_returns_zero(self):
        assert average_precision(["a"], set()) == 0.0


class TestDcgAndNdcg:
    def test_dcg_higher_when_relevant_ranked_first(self):
        dcg_first = dcg_at_k(["a", "x"], {"a"}, k=2)
        dcg_second = dcg_at_k(["x", "a"], {"a"}, k=2)
        assert dcg_first > dcg_second

    def test_ndcg_of_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_ndcg_of_empty_relevant_set_is_zero(self):
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_ndcg_is_between_zero_and_one(self):
        value = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
        assert 0.0 <= value <= 1.0


class TestGroundingPrecision:
    def test_all_scores_above_threshold(self):
        assert grounding_precision([0.9, 0.8, 0.7], threshold=0.5, k=3) == 1.0

    def test_none_above_threshold(self):
        assert grounding_precision([0.1, 0.2], threshold=0.5, k=2) == 0.0

    def test_partial(self):
        assert grounding_precision(
            [0.9, 0.1, 0.6], threshold=0.5, k=3
        ) == pytest.approx(2 / 3)

    def test_respects_k_truncation(self):
        # Only the top-1 (0.9) should count when k=1, ignoring the low score after it.
        assert grounding_precision([0.9, 0.1], threshold=0.5, k=1) == 1.0

    def test_empty_scores_returns_zero(self):
        assert grounding_precision([], threshold=0.5, k=5) == 0.0
