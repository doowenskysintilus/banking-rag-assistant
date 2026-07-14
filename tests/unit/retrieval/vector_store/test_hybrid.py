from backend.retrieval.vector_store.hybrid import reciprocal_rank_fusion


def test_single_ranking_preserves_relative_order():
    scores = reciprocal_rank_fusion([[10, 20, 30]])
    assert scores[10] > scores[20] > scores[30]


def test_ids_present_in_both_rankings_score_higher():
    dense = [1, 2, 3]
    sparse = [3, 1, 2]
    scores = reciprocal_rank_fusion([dense, sparse])
    # id 1 is top-of-dense and 2nd-of-sparse; id 2 is 2nd-of-dense but last-of-sparse
    assert scores[1] > scores[2]


def test_id_only_in_one_ranking_still_gets_a_score():
    scores = reciprocal_rank_fusion([[1, 2], [3]])
    assert 1 in scores
    assert 2 in scores
    assert 3 in scores


def test_empty_rankings_produce_no_scores():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_smaller_k_amplifies_rank_differences():
    ranking = [1, 2]
    scores_small_k = reciprocal_rank_fusion([ranking], k=1)
    scores_large_k = reciprocal_rank_fusion([ranking], k=1000)

    gap_small_k = scores_small_k[1] - scores_small_k[2]
    gap_large_k = scores_large_k[1] - scores_large_k[2]
    assert gap_small_k > gap_large_k
