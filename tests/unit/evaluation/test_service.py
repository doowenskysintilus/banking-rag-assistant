from backend.evaluation.service import EvaluationService
from backend.evaluation.types import EvalSample
from tests.unit.evaluation.fakes import FakeEvaluationProvider


def _samples(n: int = 2) -> list[EvalSample]:
    return [
        EvalSample(
            sample_id=f"s{i}",
            question=f"question {i}",
            answer=f"answer {i}",
            contexts=["ctx"],
            ground_truth="gt",
        )
        for i in range(n)
    ]


class TestEvaluationServiceHappyPath:
    async def test_merges_scores_from_multiple_providers_into_one_report(self):
        ragas = FakeEvaluationProvider(
            "ragas",
            ("faithfulness", "context_precision", "context_recall"),
            scores={
                "faithfulness": [0.9, 0.7],
                "context_precision": [1.0, 0.5],
                "context_recall": [0.8, 0.6],
            },
        )
        deepeval = FakeEvaluationProvider(
            "deepeval",
            ("answer_relevancy", "hallucination_rate"),
            scores={"answer_relevancy": [0.95, 0.85], "hallucination_rate": [1.0, 0.9]},
        )
        service = EvaluationService(providers=[ragas, deepeval])

        report = await service.evaluate(_samples(), dataset_name="unit-test")

        assert report.num_samples == 2
        assert report.dataset_name == "unit-test"
        assert report.aggregate_scores["faithfulness"] == 0.8  # mean(0.9, 0.7)
        assert report.aggregate_scores["hallucination_rate"] == 0.95  # mean(1.0, 0.9)
        assert report.providers_used == {
            "faithfulness": "ragas",
            "context_precision": "ragas",
            "context_recall": "ragas",
            "answer_relevancy": "deepeval",
            "hallucination_rate": "deepeval",
        }
        assert report.warnings == []
        assert len(report.sample_evaluations) == 2
        assert report.sample_evaluations[0].score_for("faithfulness") == 0.9

    async def test_calls_each_provider_with_only_the_metrics_assigned_to_it(self):
        ragas = FakeEvaluationProvider(
            "ragas", ("faithfulness", "context_precision", "context_recall")
        )
        deepeval = FakeEvaluationProvider(
            "deepeval", ("answer_relevancy", "hallucination_rate")
        )
        service = EvaluationService(providers=[ragas, deepeval])

        await service.evaluate(_samples(1), dataset_name="unit-test")

        assert set(ragas.calls[0][1]) == {
            "faithfulness",
            "context_precision",
            "context_recall",
        }
        assert set(deepeval.calls[0][1]) == {"answer_relevancy", "hallucination_rate"}


class TestEvaluationServiceDegradedProviders:
    async def test_unavailable_provider_metrics_are_skipped_with_a_warning(self):
        ragas = FakeEvaluationProvider(
            "ragas",
            ("faithfulness", "context_precision", "context_recall"),
            available=False,
        )
        deepeval = FakeEvaluationProvider(
            "deepeval", ("answer_relevancy", "hallucination_rate")
        )
        service = EvaluationService(providers=[ragas, deepeval])

        report = await service.evaluate(_samples(), dataset_name="unit-test")

        assert "faithfulness" not in report.aggregate_scores
        assert "answer_relevancy" in report.aggregate_scores
        assert any("ragas" in w and "unavailable" in w for w in report.warnings)
        assert any("faithfulness" in w for w in report.warnings)

    async def test_per_sample_none_scores_are_excluded_from_the_samples_scores_but_kept_in_aggregate_mean(
        self,
    ):
        deepeval = FakeEvaluationProvider(
            "deepeval",
            ("answer_relevancy", "hallucination_rate"),
            scores={"answer_relevancy": [0.9, None]},
        )
        service = EvaluationService(providers=[deepeval])

        report = await service.evaluate(_samples(2), dataset_name="unit-test")

        assert (
            report.aggregate_scores["answer_relevancy"] == 0.9
        )  # mean of only the non-None score
        assert report.sample_evaluations[0].score_for("answer_relevancy") == 0.9
        assert report.sample_evaluations[1].score_for("answer_relevancy") is None

    async def test_every_sample_failing_a_metric_produces_a_warning_and_no_aggregate_entry(
        self,
    ):
        deepeval = FakeEvaluationProvider(
            "deepeval",
            ("answer_relevancy", "hallucination_rate"),
            scores={"answer_relevancy": [None, None]},
        )
        service = EvaluationService(providers=[deepeval])

        report = await service.evaluate(_samples(2), dataset_name="unit-test")

        assert "answer_relevancy" not in report.aggregate_scores
        assert any("answer_relevancy" in w and "failed" in w for w in report.warnings)


class TestEvaluationServiceEmptyInput:
    async def test_empty_sample_list_returns_an_empty_report_with_a_warning(self):
        service = EvaluationService(providers=[])
        report = await service.evaluate([], dataset_name="unit-test")

        assert report.num_samples == 0
        assert report.aggregate_scores == {}
        assert report.sample_evaluations == []
        assert report.warnings
