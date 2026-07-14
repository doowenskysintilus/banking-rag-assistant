from backend.evaluation.selector import probe_availability, select_metric_providers
from tests.unit.evaluation.fakes import FakeEvaluationProvider


def _providers(ragas_available=True, deepeval_available=True):
    ragas = FakeEvaluationProvider(
        "ragas",
        ("faithfulness", "context_precision", "context_recall"),
        available=ragas_available,
    )
    deepeval = FakeEvaluationProvider(
        "deepeval",
        ("answer_relevancy", "hallucination_rate"),
        available=deepeval_available,
    )
    return [ragas, deepeval]


def test_probe_availability_reflects_each_providers_is_available():
    providers = _providers(ragas_available=True, deepeval_available=False)
    assert probe_availability(providers) == {"ragas": True, "deepeval": False}


def test_assigns_each_metric_to_its_preferred_provider_when_both_available():
    assignment = select_metric_providers(_providers())
    assert assignment == {
        "faithfulness": "ragas",
        "context_precision": "ragas",
        "context_recall": "ragas",
        "answer_relevancy": "deepeval",
        "hallucination_rate": "deepeval",
    }


def test_hallucination_rate_has_no_fallback_and_is_skipped_if_deepeval_unavailable():
    assignment = select_metric_providers(_providers(deepeval_available=False))
    assert "hallucination_rate" not in assignment
    assert (
        "answer_relevancy" not in assignment
    )  # deepeval is its only listed provider too


def test_ragas_metrics_are_skipped_entirely_if_ragas_unavailable_and_deepeval_has_no_fallback_for_them():
    assignment = select_metric_providers(_providers(ragas_available=False))
    assert "faithfulness" not in assignment
    assert "context_precision" not in assignment
    assert "context_recall" not in assignment
    # answer_relevancy still resolves to deepeval regardless
    assert assignment["answer_relevancy"] == "deepeval"


def test_no_providers_available_yields_empty_assignment():
    assignment = select_metric_providers(
        _providers(ragas_available=False, deepeval_available=False)
    )
    assert assignment == {}


def test_precomputed_availability_can_be_passed_in_to_avoid_recomputing():
    providers = _providers()
    availability = {"ragas": False, "deepeval": True}
    assignment = select_metric_providers(providers, availability=availability)
    assert "faithfulness" not in assignment
    assert assignment["answer_relevancy"] == "deepeval"
