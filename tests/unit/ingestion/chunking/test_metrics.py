import numpy as np

from backend.ingestion.chunking.metrics import compute_metrics
from backend.ingestion.chunking.types import Chunk, ChunkStrategy


def _make_chunk(
    raw_text: str, index: int, header_path: str = "", overlap: int = 0
) -> Chunk:
    text = f"[{header_path}]\n{raw_text}" if header_path else raw_text
    return Chunk(
        text=text,
        raw_text=raw_text,
        index=index,
        char_start=0,
        char_end=len(raw_text),
        header_path=header_path,
        strategy=ChunkStrategy.RECURSIVE,
        overlap_sentence_count=overlap,
        approx_tokens=max(1, len(text) // 4),
    )


def test_empty_chunk_list_returns_zeroed_metrics():
    metrics = compute_metrics([], target_min_chars=100, target_max_chars=500)
    assert metrics.num_chunks == 0
    assert metrics.avg_chars == 0.0
    assert metrics.pct_within_target_range == 0.0


def test_computes_basic_size_statistics():
    chunks = [
        _make_chunk("a" * 100, 0),
        _make_chunk("b" * 200, 1),
        _make_chunk("c" * 300, 2),
    ]
    metrics = compute_metrics(chunks, target_min_chars=50, target_max_chars=1000)
    assert metrics.num_chunks == 3
    assert metrics.avg_chars == 200.0
    assert metrics.median_chars == 200.0
    assert metrics.min_chars == 100
    assert metrics.max_chars == 300


def test_pct_within_target_range():
    chunks = [
        _make_chunk("a" * 50, 0),
        _make_chunk("b" * 500, 1),
        _make_chunk("c" * 150, 2),
    ]
    metrics = compute_metrics(chunks, target_min_chars=100, target_max_chars=200)
    # only the 150-char chunk falls within [100, 200]
    assert metrics.pct_within_target_range == 1 / 3


def test_pct_ending_at_sentence_boundary():
    chunks = [
        _make_chunk("This ends properly.", 0),
        _make_chunk("This does not end properly", 1),
        _make_chunk('This ends with a quote."', 2),
    ]
    metrics = compute_metrics(chunks, target_min_chars=0, target_max_chars=1000)
    assert metrics.pct_ending_at_sentence_boundary == 2 / 3


def test_pct_with_context_header():
    chunks = [
        _make_chunk("Body one.", 0, header_path="Article 1"),
        _make_chunk("Body two.", 1, header_path=""),
    ]
    metrics = compute_metrics(chunks, target_min_chars=0, target_max_chars=1000)
    assert metrics.pct_with_context_header == 0.5


def test_overlap_metrics():
    chunks = [
        _make_chunk("Body one.", 0, overlap=0),
        _make_chunk("Body two.", 1, overlap=2),
        _make_chunk("Body three.", 2, overlap=1),
    ]
    metrics = compute_metrics(chunks, target_min_chars=0, target_max_chars=1000)
    assert metrics.avg_overlap_sentences == 1.0
    assert metrics.pct_chunks_with_overlap == 2 / 3


def test_coherence_metrics_are_none_without_embed_fn():
    chunks = [_make_chunk("First sentence. Second sentence.", 0)]
    metrics = compute_metrics(chunks, target_min_chars=0, target_max_chars=1000)
    assert metrics.avg_intra_chunk_coherence is None
    assert metrics.avg_boundary_coherence is None
    assert metrics.coherence_gap is None


def test_coherence_metrics_are_populated_with_embed_fn():
    def embed_fn(texts: list[str]) -> np.ndarray:
        # deterministic: embed by text length parity, so "coherent" pairs are
        # trivially identical vectors and everything is computable without ML deps.
        return np.array(
            [[1.0, 0.0] if len(t) % 2 == 0 else [0.0, 1.0] for t in texts],
            dtype="float32",
        )

    chunks = [
        _make_chunk("First sentence. Second sentence here now.", 0),
        _make_chunk("Third sentence appears. Fourth sentence appears.", 1),
    ]
    metrics = compute_metrics(
        chunks, target_min_chars=0, target_max_chars=1000, embed_fn=embed_fn
    )
    assert metrics.avg_intra_chunk_coherence is not None
    assert metrics.avg_boundary_coherence is not None
    assert (
        metrics.coherence_gap
        == metrics.avg_intra_chunk_coherence - metrics.avg_boundary_coherence
    )
