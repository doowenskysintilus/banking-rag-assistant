from backend.ingestion.chunking.benchmark import format_report, run_benchmark
from backend.ingestion.chunking.fixtures import ALL_FIXTURES
from backend.ingestion.chunking.types import ChunkStrategy


def test_run_benchmark_covers_every_fixture_document():
    results = run_benchmark(strategies=[ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE])
    assert set(results.keys()) == set(ALL_FIXTURES.keys())


def test_run_benchmark_reports_metrics_and_timing_per_strategy():
    results = run_benchmark(strategies=[ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE])
    for per_strategy in results.values():
        assert "fixed" in per_strategy
        assert "recursive" in per_strategy
        for metrics, elapsed in per_strategy.values():
            assert metrics.num_chunks > 0
            assert elapsed >= 0.0


def test_recursive_strategy_preserves_more_context_than_fixed_baseline():
    results = run_benchmark(strategies=[ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE])
    for per_strategy in results.values():
        fixed_metrics, _ = per_strategy["fixed"]
        recursive_metrics, _ = per_strategy["recursive"]
        # The fixed baseline never attaches a section breadcrumb; recursive does
        # whenever the source document has detectable structure.
        assert fixed_metrics.pct_with_context_header == 0.0
        assert recursive_metrics.pct_with_context_header > 0.0


def test_format_report_produces_a_readable_table():
    results = run_benchmark(strategies=[ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE])
    report = format_report(results)
    assert "loan_agreement" in report
    assert "banking_regulation" in report
    assert "financial_report" in report
    assert "fixed" in report
    assert "recursive" in report
