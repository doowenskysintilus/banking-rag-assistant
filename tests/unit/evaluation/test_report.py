from datetime import datetime, timezone

from backend.evaluation.report import format_markdown_report, render_dashboard_html
from backend.evaluation.types import (
    EvalReport,
    EvalSample,
    MetricScore,
    SampleEvaluation,
)


def _report(**overrides) -> EvalReport:
    defaults = dict(
        run_id="run-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        dataset_name="golden-fixture-set",
        num_samples=1,
        aggregate_scores={"faithfulness": 0.9, "hallucination_rate": 0.2},
        providers_used={"faithfulness": "ragas", "hallucination_rate": "deepeval"},
        sample_evaluations=[
            SampleEvaluation(
                sample=EvalSample(
                    sample_id="s1",
                    question="What is <b>the fee</b>?",
                    answer="a",
                    contexts=["c"],
                    ground_truth="g",
                ),
                scores=[
                    MetricScore(metric="faithfulness", score=0.9, provider="ragas")
                ],
            )
        ],
        warnings=[],
    )
    defaults.update(overrides)
    return EvalReport(**defaults)


class TestMarkdownReport:
    def test_includes_run_metadata_and_scores(self):
        md = format_markdown_report(_report())
        assert "run-1" in md
        assert "golden-fixture-set" in md
        assert "0.900" in md
        assert "s1" in md

    def test_missing_metric_renders_as_skipped(self):
        md = format_markdown_report(_report(aggregate_scores={}))
        assert "skipped" in md

    def test_warnings_section_only_appears_when_there_are_warnings(self):
        clean_md = format_markdown_report(_report(warnings=[]))
        assert "## Warnings" not in clean_md

        warned_md = format_markdown_report(_report(warnings=["something looked off"]))
        assert "## Warnings" in warned_md
        assert "something looked off" in warned_md


class TestHtmlDashboard:
    def test_renders_metric_cards_and_escapes_question_text(self):
        html = render_dashboard_html(_report())
        assert "Faithfulness" in html
        assert "&lt;b&gt;the fee&lt;/b&gt;" in html
        assert "<b>the fee</b>" not in html  # must not be injected unescaped

    def test_skipped_metric_gets_the_skipped_band(self):
        html = render_dashboard_html(_report(aggregate_scores={}))
        assert 'class="card skipped"' in html

    def test_low_score_gets_the_critical_band(self):
        html = render_dashboard_html(_report(aggregate_scores={"faithfulness": 0.1}))
        assert 'class="card critical"' in html

    def test_warnings_render_as_a_list_when_present(self):
        html = render_dashboard_html(_report(warnings=["check this answer"]))
        assert "check this answer" in html
        assert "warnings" in html.lower()

    def test_history_is_included_in_the_sparkline_when_provided(self):
        history = [
            (datetime(2025, 12, 1, tzinfo=timezone.utc), {"faithfulness": 0.5}),
            (datetime(2025, 12, 15, tzinfo=timezone.utc), {"faithfulness": 0.7}),
        ]
        html = render_dashboard_html(_report(), history=history)
        assert "sparkline" in html
