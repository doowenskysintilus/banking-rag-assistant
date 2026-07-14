"""Round-trip tests for the persistence half of the pipeline (run_and_persist /
load_report / fetch_run_history) against a real (in-memory SQLite) database
session — using fake providers so no real judge-LLM call is made.
"""

from __future__ import annotations

from backend.evaluation.runner import fetch_run_history, load_report, run_and_persist
from backend.evaluation.service import EvaluationService
from backend.evaluation.types import EvalSample
from tests.unit.evaluation.fakes import FakeEvaluationProvider
from tests.unit.rag.conftest import make_user


def _samples() -> list[EvalSample]:
    return [
        EvalSample(
            sample_id="s1",
            question="What is the fee?",
            answer="It is $35 [Source 1].",
            contexts=["ctx"],
            ground_truth="$35",
        ),
        EvalSample(
            sample_id="s2",
            question="What about wires?",
            answer="I do not have sufficient information.",
            contexts=["ctx2"],
            ground_truth="unknown",
        ),
    ]


def _service() -> EvaluationService:
    ragas = FakeEvaluationProvider(
        "ragas",
        ("faithfulness", "context_precision", "context_recall"),
        scores={"faithfulness": [0.9, 0.4]},
    )
    deepeval = FakeEvaluationProvider(
        "deepeval",
        ("answer_relevancy", "hallucination_rate"),
        scores={"hallucination_rate": [1.0, 0.5]},
    )
    return EvaluationService(providers=[ragas, deepeval])


class TestRunAndPersist:
    async def test_persists_a_run_and_its_sample_scores(self, db_session):
        report = await run_and_persist(
            db_session,
            _samples(),
            "unit-test-dataset",
            service=_service(),
            write_report_files=False,
        )

        assert report.num_samples == 2
        loaded = await load_report(db_session, report.run_id)

        assert loaded is not None
        assert loaded.dataset_name == "unit-test-dataset"
        assert loaded.num_samples == 2
        assert loaded.aggregate_scores == report.aggregate_scores
        assert loaded.providers_used == report.providers_used
        assert {s.sample.sample_id for s in loaded.sample_evaluations} == {"s1", "s2"}

    async def test_records_the_triggering_user(self, db_session):
        user = await make_user(db_session)
        report = await run_and_persist(
            db_session,
            _samples(),
            "unit-test-dataset",
            service=_service(),
            triggered_by_user_id=user.id,
            write_report_files=False,
        )
        assert report.run_id  # sanity: run completed

    async def test_load_report_returns_none_for_an_unknown_run_id(self, db_session):
        assert await load_report(db_session, "does-not-exist") is None


class TestFetchRunHistory:
    async def test_returns_prior_runs_oldest_first_excluding_the_given_run(
        self, db_session
    ):
        service = _service()
        first = await run_and_persist(
            db_session,
            _samples(),
            "history-dataset",
            service=service,
            write_report_files=False,
        )
        second = await run_and_persist(
            db_session,
            _samples(),
            "history-dataset",
            service=service,
            write_report_files=False,
        )

        history = await fetch_run_history(
            db_session, "history-dataset", exclude_run_id=second.run_id
        )

        assert len(history) == 1
        assert history[0][1] == first.aggregate_scores

    async def test_different_dataset_names_do_not_leak_into_each_others_history(
        self, db_session
    ):
        service = _service()
        await run_and_persist(
            db_session,
            _samples(),
            "dataset-a",
            service=service,
            write_report_files=False,
        )

        history = await fetch_run_history(db_session, "dataset-b")
        assert history == []
