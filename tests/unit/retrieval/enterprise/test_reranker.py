from langchain_core.documents import Document

from backend.retrieval.enterprise.reranker import LangChainRerankerCompressor
from backend.retrieval.reranking.service import RerankingService
from backend.retrieval.reranking.types import RerankerProvider


class FakeProvider(RerankerProvider):
    name = "fake"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def score(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores.get(doc, 0.0) for doc in documents]

    @property
    def is_available(self) -> bool:
        return True


def _compressor(
    scores: dict[str, float], **service_kwargs
) -> LangChainRerankerCompressor:
    service = RerankingService(FakeProvider(scores), **service_kwargs)
    return LangChainRerankerCompressor(reranking_service=service)


class TestCompressDocuments:
    def test_reorders_by_relevance_score(self):
        compressor = _compressor(
            {"unrelated vacation policy": 0.1, "overdraft fee schedule": 0.9},
            min_relevance_score=0.0,
        )
        documents = [
            Document(
                page_content="unrelated vacation policy", metadata={"chunk_id": "a"}
            ),
            Document(page_content="overdraft fee schedule", metadata={"chunk_id": "b"}),
        ]

        result = compressor.compress_documents(documents, query="overdraft fees")

        assert result[0].metadata["chunk_id"] == "b"
        assert result[0].metadata["rerank_score"] == 0.9

    def test_truncates_to_top_n(self):
        scores = {f"doc {i}": float(i) for i in range(5)}
        compressor = _compressor(scores, top_n=2, min_relevance_score=0.0)
        documents = [Document(page_content=f"doc {i}") for i in range(5)]

        result = compressor.compress_documents(documents, query="anything")

        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        compressor = _compressor({})
        assert compressor.compress_documents([], query="anything") == []

    def test_original_documents_are_not_mutated(self):
        compressor = _compressor({"text": 1.0}, min_relevance_score=0.0)
        original = Document(page_content="text", metadata={"chunk_id": "a"})

        compressor.compress_documents([original], query="q")

        assert "rerank_score" not in original.metadata

    def test_low_confidence_documents_are_dropped(self):
        compressor = _compressor({"weak match": 0.05}, min_relevance_score=0.5)
        documents = [Document(page_content="weak match")]

        result = compressor.compress_documents(documents, query="q")

        assert result == []

    async def test_async_entry_point_produces_the_correct_order(self):
        compressor = _compressor({"a": 0.9, "b": 0.1}, min_relevance_score=0.0)
        documents = [Document(page_content="a"), Document(page_content="b")]

        result = await compressor.acompress_documents(documents, query="q")

        assert [d.page_content for d in result] == ["a", "b"]

    def test_sync_entry_point_produces_the_correct_order(self):
        # Called directly with no event loop running (e.g. a standalone script) —
        # the realistic scenario for the sync path. LangChain's own async
        # invocation instead runs this same sync method inside a thread-pool
        # executor (see BaseDocumentCompressor.acompress_documents), which is
        # exercised by test_async_entry_point_produces_the_correct_order above via
        # acompress_documents directly, not by calling compress_documents from
        # already-async test code (that would fight over the running loop, which
        # isn't how LangChain itself invokes this path).
        compressor = _compressor({"a": 0.9, "b": 0.1}, min_relevance_score=0.0)
        documents = [Document(page_content="a"), Document(page_content="b")]

        result = compressor.compress_documents(documents, query="q")

        assert [d.page_content for d in result] == ["a", "b"]
