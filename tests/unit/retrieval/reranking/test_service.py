from backend.retrieval.reranking.service import RerankingService
from backend.retrieval.reranking.types import RerankerProvider


class FakeProvider(RerankerProvider):
    name = "fake"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self._scores.get(doc, 0.0) for doc in documents]

    @property
    def is_available(self) -> bool:
        return True


class TestRerank:
    async def test_empty_documents_returns_empty_result_without_scoring(self):
        provider = FakeProvider({})
        service = RerankingService(provider)
        result = await service.rerank("query", [])
        assert result.documents == []
        assert result.low_confidence is False
        assert provider.calls == []

    async def test_sorts_documents_by_descending_relevance_score(self):
        provider = FakeProvider({"low": 0.1, "high": 0.9, "mid": 0.5})
        service = RerankingService(provider, min_relevance_score=0.0)

        result = await service.rerank("query", ["low", "high", "mid"])

        assert [d.content for d in result.documents] == ["high", "mid", "low"]

    async def test_preserves_original_index_for_traceability(self):
        provider = FakeProvider({"a": 0.2, "b": 0.8})
        service = RerankingService(provider, min_relevance_score=0.0)

        result = await service.rerank("query", ["a", "b"])

        by_content = {d.content: d.original_index for d in result.documents}
        assert by_content == {"a": 0, "b": 1}

    async def test_truncates_to_top_n(self):
        provider = FakeProvider({f"doc{i}": float(i) for i in range(10)})
        service = RerankingService(provider, top_n=3, min_relevance_score=0.0)

        result = await service.rerank("query", [f"doc{i}" for i in range(10)])

        assert len(result.documents) == 3
        assert [d.content for d in result.documents] == ["doc9", "doc8", "doc7"]

    async def test_metadata_is_attached_per_document(self):
        provider = FakeProvider({"a": 0.9, "b": 0.1})
        service = RerankingService(provider, min_relevance_score=0.0)

        result = await service.rerank(
            "query", ["a", "b"], metadatas=[{"chunk_id": "id-a"}, {"chunk_id": "id-b"}]
        )

        assert result.documents[0].metadata == {"chunk_id": "id-a"}


class TestConfidenceThreshold:
    async def test_documents_below_threshold_are_dropped_entirely(self):
        provider = FakeProvider({"relevant": 0.9, "irrelevant": 0.05})
        service = RerankingService(provider, min_relevance_score=0.5)

        result = await service.rerank("query", ["relevant", "irrelevant"])

        assert [d.content for d in result.documents] == ["relevant"]
        assert result.low_confidence is False

    async def test_low_confidence_flag_set_when_nothing_clears_the_threshold(self):
        provider = FakeProvider({"a": 0.1, "b": 0.2})
        service = RerankingService(provider, min_relevance_score=0.5)

        result = await service.rerank("query", ["a", "b"])

        assert result.documents == []
        assert result.low_confidence is True

    async def test_zero_threshold_keeps_non_negative_scores(self):
        provider = FakeProvider({"a": 0.0, "b": 0.3})
        service = RerankingService(provider, min_relevance_score=0.0)

        result = await service.rerank("query", ["a", "b"])

        assert len(result.documents) == 2
        assert result.low_confidence is False

    async def test_negative_score_is_still_filtered_by_a_zero_threshold(self):
        # min_relevance_score=0.0 isn't "no filtering" — it's "require non-negative
        # relevance." Real providers always return sigmoid-normalized [0, 1] scores,
        # so this only matters for hypothetical/test scoring functions.
        provider = FakeProvider({"a": 0.0, "b": -1.0})
        service = RerankingService(provider, min_relevance_score=0.0)

        result = await service.rerank("query", ["a", "b"])

        assert [d.content for d in result.documents] == ["a"]


def test_provider_name_is_exposed():
    service = RerankingService(FakeProvider({}))
    assert service.provider_name == "fake"
