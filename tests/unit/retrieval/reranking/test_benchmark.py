from backend.retrieval.reranking.benchmark import format_report, run_benchmark
from backend.retrieval.reranking.fixtures import EVAL_EXAMPLES


def test_run_benchmark_always_includes_the_baseline():
    results = run_benchmark()
    assert "baseline" in results
    assert results["baseline"].num_examples == len(EVAL_EXAMPLES)


def test_run_benchmark_baseline_metrics_are_in_valid_ranges():
    results = run_benchmark()
    baseline = results["baseline"]
    for value in (
        baseline.precision_at_k,
        baseline.recall_at_k,
        baseline.mrr,
        baseline.map_score,
        baseline.ndcg_at_k,
        baseline.grounding_precision,
    ):
        assert 0.0 <= value <= 1.0


def test_format_report_includes_every_result_and_a_caveat_note():
    results = run_benchmark()
    report = format_report(results)
    assert "baseline" in report
    assert "grounding" in report.lower()
    assert "calibrated confidence" in report.lower()
