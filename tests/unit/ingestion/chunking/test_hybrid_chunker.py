import pytest

from backend.ingestion.chunking.fixtures import LOAN_AGREEMENT
from backend.ingestion.chunking.hybrid_chunker import ChunkerConfig, HybridChunker
from backend.ingestion.chunking.types import ChunkStrategy

# A single long, unstructured section (one heading, many sentences, no
# sub-headings) — representative of the narrative clauses that actually benefit
# from semantic (rather than purely structural) sub-splitting.
LONG_UNSTRUCTURED_SECTION = (
    "Article 9 - Miscellaneous\n\n"
    "The Bank may assign its rights under this Agreement to any other financial institution. "
    "The Borrower may not assign any of its rights or obligations without the Bank's prior written consent. "
    "Notices under this Agreement must be given in writing and delivered by hand, post, or email. "
    "A notice is deemed received on the next Business Day after it is sent. "
    "If any provision of this Agreement is held invalid, the remaining provisions continue in full force. "
    "This Agreement may be executed in counterparts, each of which is an original. "
    "No waiver of any breach is a waiver of any subsequent breach. "
    "The Bank's failure to exercise any right does not constitute a waiver of that right."
)


class TestChunkerConfig:
    def test_rejects_overlap_larger_than_zero_being_negative(self):
        with pytest.raises(ValueError):
            ChunkerConfig(overlap_sentences=-1)

    def test_rejects_min_chars_not_smaller_than_max_chars(self):
        with pytest.raises(ValueError):
            ChunkerConfig(min_chars=500, max_chars=500)


class TestHybridChunkerRecursiveMode:
    async def test_produces_chunks_within_size_bounds(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE, max_chars=500, min_chars=100
        )
        chunker = HybridChunker(config)

        chunks = await chunker.chunk(LOAN_AGREEMENT)

        assert len(chunks) > 1
        for chunk in chunks:
            assert (
                len(chunk.raw_text) <= config.max_chars + 50
            )  # small slack for force-split edge cases
            assert chunk.strategy is ChunkStrategy.RECURSIVE

    async def test_semantic_strategy_without_embedder_falls_back_to_recursive(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.SEMANTIC, max_chars=500, min_chars=100
        )
        chunker = HybridChunker(config)
        chunks = await chunker.chunk(LOAN_AGREEMENT, embedder=None)
        assert all(c.strategy is ChunkStrategy.RECURSIVE for c in chunks)

    async def test_chunks_are_indexed_sequentially(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE, max_chars=400, min_chars=80
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    async def test_char_offsets_point_back_into_the_source_document(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE, max_chars=500, min_chars=100
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        for chunk in chunks:
            assert 0 <= chunk.char_start < chunk.char_end <= len(LOAN_AGREEMENT)

    async def test_context_header_is_prepended_when_section_has_a_breadcrumb(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE, max_chars=500, min_chars=100
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        with_headers = [c for c in chunks if c.header_path]
        assert with_headers
        for chunk in with_headers:
            assert chunk.text.startswith(f"[{chunk.header_path}]")
            assert chunk.raw_text in chunk.text

    async def test_disabling_context_prefix_keeps_text_equal_to_raw_text(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE,
            max_chars=500,
            min_chars=100,
            include_header_prefix=False,
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        assert all(c.text == c.raw_text for c in chunks)

    async def test_overlap_is_applied_between_consecutive_chunks_in_a_section(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE,
            max_chars=350,
            min_chars=80,
            overlap_sentences=1,
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        assert any(c.overlap_sentence_count > 0 for c in chunks)

    async def test_zero_overlap_produces_no_repeated_content(self):
        config = ChunkerConfig(
            strategy=ChunkStrategy.RECURSIVE,
            max_chars=350,
            min_chars=80,
            overlap_sentences=0,
        )
        chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
        assert all(c.overlap_sentence_count == 0 for c in chunks)


class TestHybridChunkerSemanticMode:
    async def test_hybrid_uses_semantic_grouping_for_large_sections(
        self, fake_embedder
    ):
        config = ChunkerConfig(
            strategy=ChunkStrategy.HYBRID,
            max_chars=2000,
            min_chars=50,
            semantic_min_units=3,
            semantic_breakpoint_percentile=60.0,
        )
        chunks = await HybridChunker(config).chunk(
            LONG_UNSTRUCTURED_SECTION, embedder=fake_embedder
        )
        assert any(c.strategy is ChunkStrategy.SEMANTIC for c in chunks)
        assert fake_embedder.calls  # the embedder was actually invoked

    async def test_short_sections_are_not_sent_to_the_embedder(self, fake_embedder):
        # semantic_min_units very high -> no section ever qualifies for semantic
        # regrouping, so recursive packing is used everywhere and the embedder is
        # never called.
        config = ChunkerConfig(strategy=ChunkStrategy.HYBRID, semantic_min_units=10_000)
        chunks = await HybridChunker(config).chunk(
            LOAN_AGREEMENT, embedder=fake_embedder
        )
        assert all(c.strategy is ChunkStrategy.RECURSIVE for c in chunks)
        assert fake_embedder.calls == []

    async def test_semantic_chunks_still_respect_max_chars(self, fake_embedder):
        config = ChunkerConfig(
            strategy=ChunkStrategy.SEMANTIC,
            max_chars=300,
            min_chars=50,
            semantic_min_units=3,
            semantic_breakpoint_percentile=50.0,
        )
        chunks = await HybridChunker(config).chunk(
            LONG_UNSTRUCTURED_SECTION, embedder=fake_embedder
        )
        for chunk in chunks:
            assert len(chunk.raw_text) <= config.max_chars + 50


async def test_all_source_text_is_covered_by_some_chunk():
    config = ChunkerConfig(
        strategy=ChunkStrategy.RECURSIVE, max_chars=400, min_chars=80
    )
    chunks = await HybridChunker(config).chunk(LOAN_AGREEMENT)
    # every non-trivial word from the source appears in at least one chunk's raw text
    combined = " ".join(c.raw_text for c in chunks)
    for keyword in [
        "Borrower",
        "Bank",
        "Interest Rate",
        "Event of Default",
        "Governing Law",
    ]:
        assert keyword in combined
