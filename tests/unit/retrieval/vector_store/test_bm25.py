from backend.retrieval.vector_store.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("The Overdraft Limit!") == ["the", "overdraft", "limit"]


class TestBM25Index:
    def test_empty_index_returns_no_results(self):
        index = BM25Index()
        assert index.search("anything", top_k=5) == []

    def test_finds_document_containing_query_terms(self):
        index = BM25Index()
        index.add(1, "The overdraft limit is five hundred dollars.")
        index.add(2, "Vacation request process for part-time employees.")

        results = index.search("overdraft limit", top_k=5)
        assert results
        assert results[0][0] == 1

    def test_documents_with_no_matching_terms_are_excluded(self):
        index = BM25Index()
        index.add(1, "The overdraft limit is five hundred dollars.")
        results = index.search("vacation request", top_k=5)
        assert results == []

    def test_more_term_overlap_ranks_higher(self):
        index = BM25Index()
        index.add(1, "overdraft overdraft overdraft policy fees")
        index.add(
            2,
            "a single mention of overdraft in a long unrelated passage about policy and fees",
        )
        results = index.search("overdraft policy fees", top_k=5)
        ranked_ids = [vector_id for vector_id, _ in results]
        assert ranked_ids[0] == 1

    def test_respects_top_k(self):
        index = BM25Index()
        for i in range(10):
            index.add(i, "overdraft policy fees apply")
        results = index.search("overdraft", top_k=3)
        assert len(results) == 3

    def test_remove_excludes_document_from_future_searches(self):
        index = BM25Index()
        index.add(1, "overdraft policy text")
        index.remove(1)
        assert index.search("overdraft", top_k=5) == []

    def test_remove_unknown_id_is_a_no_op(self):
        index = BM25Index()
        index.remove(999)  # must not raise

    def test_re_adding_same_id_replaces_previous_text(self):
        index = BM25Index()
        index.add(1, "overdraft policy")
        index.add(1, "completely different vacation content")

        assert index.search("overdraft", top_k=5) == []
        results = index.search("vacation", top_k=5)
        assert results and results[0][0] == 1

    def test_len_reflects_indexed_document_count(self):
        index = BM25Index()
        assert len(index) == 0
        index.add(1, "text one")
        index.add(2, "text two")
        assert len(index) == 2
        index.remove(1)
        assert len(index) == 1
