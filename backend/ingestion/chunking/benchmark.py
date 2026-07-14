"""Compares chunking strategies (fixed-size baseline vs recursive vs semantic vs
hybrid) on representative banking documents, reporting quality metrics and timing.

Run as a script:
    python -m backend.ingestion.chunking.benchmark

Semantic/hybrid rows are skipped automatically if the embedding backend
(sentence-transformers + model weights) isn't available in the current
environment — the recursive/fixed comparison still runs everywhere.
"""

from __future__ import annotations

import asyncio
import time

from backend.ingestion.chunker import chunk_text
from backend.ingestion.chunking.fixtures import ALL_FIXTURES
from backend.ingestion.chunking.hybrid_chunker import ChunkerConfig, HybridChunker
from backend.ingestion.chunking.metrics import ChunkingMetrics, compute_metrics
from backend.ingestion.chunking.types import Chunk, ChunkStrategy

_DEFAULT_CONFIG = ChunkerConfig()
_DEFAULT_STRATEGIES = [
    ChunkStrategy.FIXED,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.SEMANTIC,
    ChunkStrategy.HYBRID,
]


def _fixed_baseline_chunks(
    text: str, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    raw_chunks = chunk_text(text, chunk_size=max_chars, chunk_overlap=overlap_chars)
    return [
        Chunk(
            text=raw,
            raw_text=raw,
            index=i,
            char_start=0,
            char_end=len(raw),
            header_path="",
            strategy=ChunkStrategy.FIXED,
            approx_tokens=max(1, len(raw) // 4),
        )
        for i, raw in enumerate(raw_chunks)
    ]


def _try_load_embedder():
    """Best-effort: returns a live EmbeddingService, or None if the optional
    ML dependency/model isn't available in this environment.
    """
    try:
        from backend.retrieval.embeddings import get_embedding_service

        service = get_embedding_service()
        asyncio.run(service.embed_texts(["warm-up"]))
        return service
    except Exception:
        return None


def _build_chunks(
    strategy: ChunkStrategy, text: str, embedder, config: ChunkerConfig
) -> list[Chunk]:
    if strategy is ChunkStrategy.FIXED:
        # Convert the "N sentences of overlap" config into an approximate character
        # count so the legacy fixed-size baseline has a roughly comparable overlap.
        approx_overlap_chars = config.overlap_sentences * 40
        return _fixed_baseline_chunks(text, config.max_chars, approx_overlap_chars)

    chunker = HybridChunker(
        ChunkerConfig(
            strategy=strategy,
            max_chars=config.max_chars,
            min_chars=config.min_chars,
            overlap_sentences=config.overlap_sentences,
            semantic_breakpoint_percentile=config.semantic_breakpoint_percentile,
            semantic_min_units=config.semantic_min_units,
        )
    )
    use_embedder = embedder if strategy is not ChunkStrategy.RECURSIVE else None
    return asyncio.run(chunker.chunk(text, embedder=use_embedder))


def run_benchmark(
    strategies: list[ChunkStrategy] | None = None,
    config: ChunkerConfig = _DEFAULT_CONFIG,
) -> dict[str, dict[str, tuple[ChunkingMetrics, float]]]:
    """Returns `{document_name: {strategy_value: (metrics, elapsed_seconds)}}`."""
    strategies = strategies or _DEFAULT_STRATEGIES
    embedder = _try_load_embedder()

    results: dict[str, dict[str, tuple[ChunkingMetrics, float]]] = {}

    for doc_name, text in ALL_FIXTURES.items():
        results[doc_name] = {}
        for strategy in strategies:
            if (
                strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.HYBRID)
                and embedder is None
            ):
                continue

            start = time.perf_counter()
            chunks = _build_chunks(strategy, text, embedder, config)
            elapsed = time.perf_counter() - start

            embed_fn = (
                (lambda texts: asyncio.run(embedder.embed_texts(texts)))
                if embedder
                else None
            )
            metrics = compute_metrics(
                chunks,
                target_min_chars=config.min_chars,
                target_max_chars=config.max_chars,
                embed_fn=embed_fn,
            )
            results[doc_name][strategy.value] = (metrics, elapsed)

    return results


def format_report(results: dict[str, dict[str, tuple[ChunkingMetrics, float]]]) -> str:
    header = (
        f"{'strategy':<10} {'n':>4} {'avg_chars':>9} {'p_in_range':>10} "
        f"{'p_boundary':>10} {'p_header':>9} {'coh_gap':>8} {'ms':>8}"
    )
    lines: list[str] = []
    for doc_name, per_strategy in results.items():
        lines.append(f"\n=== {doc_name} ===")
        lines.append(header)
        lines.append("-" * len(header))
        for strategy_name, (metrics, elapsed) in per_strategy.items():
            gap = (
                f"{metrics.coherence_gap:.3f}"
                if metrics.coherence_gap is not None
                else "n/a"
            )
            lines.append(
                f"{strategy_name:<10} {metrics.num_chunks:>4} {metrics.avg_chars:>9.1f} "
                f"{metrics.pct_within_target_range:>10.1%} {metrics.pct_ending_at_sentence_boundary:>10.1%} "
                f"{metrics.pct_with_context_header:>9.1%} {gap:>8} {elapsed * 1000:>8.1f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(run_benchmark()))
