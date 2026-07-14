import pytest

from backend.retrieval.reranking.evaluation import EvaluationExample, RerankEvaluator


def _example() -> EvaluationExample:
    return EvaluationExample(
        query="overdraft fee",
        candidate_ids=["c1", "c2", "c3"],
        candidate_texts=[
            "overdraft policy text",
            "unrelated vacation text",
            "wire transfer text",
        ],
        relevant_ids={"c1"},
    )


class TestBaselineEvaluation:
    async def test_baseline_preserves_original_candidate_order(self):
        evaluator = RerankEvaluator(k=3)
        report = await evaluator.evaluate([_example()])

        # c1 (relevant) is already first in candidate order, so a no-op baseline
        # should get a perfect score on this single example.
        assert report.precision_at_k == pytest.approx(1 / 3)
        assert report.mrr == 1.0
        assert report.provider_name == "baseline"

    async def test_baseline_report_reflects_example_count(self):
        evaluator = RerankEvaluator(k=3)
        report = await evaluator.evaluate([_example(), _example()])
        assert report.num_examples == 2


class TestCustomScoreFn:
    async def test_perfect_reranker_achieves_perfect_metrics(self):
        async def perfect_score_fn(query: str, documents: list[str]) -> list[float]:
            # Always ranks the known-relevant text first.
            return [1.0 if "overdraft" in doc else 0.0 for doc in documents]

        evaluator = RerankEvaluator(k=1, relevance_threshold=0.5)
        report = await evaluator.evaluate(
            [_example()], score_fn=perfect_score_fn, provider_name="perfect"
        )

        assert report.precision_at_k == 1.0
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.map_score == 1.0
        assert report.ndcg_at_k == 1.0
        assert report.grounding_precision == 1.0
        assert report.provider_name == "perfect"

    async def test_worst_case_reranker_scores_zero_on_precision_and_recall_at_k(self):
        async def worst_score_fn(query: str, documents: list[str]) -> list[float]:
            # Ranks the irrelevant documents above the relevant one (which ends up
            # last, at rank 3 of 3 — not excluded, so MRR is 1/3, not 0).
            return [0.0 if "overdraft" in doc else 1.0 for doc in documents]

        evaluator = RerankEvaluator(k=1)
        report = await evaluator.evaluate([_example()], score_fn=worst_score_fn)

        assert report.precision_at_k == 0.0
        assert report.recall_at_k == 0.0
        assert report.mrr == pytest.approx(1 / 3)

    async def test_grounding_precision_reflects_score_fn_calibration(self):
        async def low_confidence_score_fn(
            query: str, documents: list[str]
        ) -> list[float]:
            return [0.05 if "overdraft" in doc else 0.01 for doc in documents]

        evaluator = RerankEvaluator(k=3, relevance_threshold=0.5)
        report = await evaluator.evaluate(
            [_example()], score_fn=low_confidence_score_fn
        )

        # Even though ranking is correct, nothing clears the confidence threshold.
        assert report.grounding_precision == 0.0
        assert report.precision_at_k == pytest.approx(1 / 3)


async def test_evaluate_multiple_examples_averages_metrics():
    good_example = _example()
    bad_example = EvaluationExample(
        query="wire transfer limit",
        candidate_ids=["d1", "d2"],
        candidate_texts=["unrelated text", "also unrelated"],
        relevant_ids={"d1"},
    )

    async def score_fn(query: str, documents: list[str]) -> list[float]:
        return [1.0 if "overdraft" in doc else 0.0 for doc in documents]

    evaluator = RerankEvaluator(k=1)
    report = await evaluator.evaluate([good_example, bad_example], score_fn=score_fn)

    # good_example: relevant doc correctly identified (precision@1 = 1.0).
    # bad_example: score_fn has no signal for either candidate (both score 0.0);
    # the stable sort preserves original order, and "d1" (the relevant one)
    # already came first — so this also scores precision@1 = 1.0.
    assert report.num_examples == 2
    assert report.precision_at_k == pytest.approx(1.0)
